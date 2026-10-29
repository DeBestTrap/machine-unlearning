"""Report semantics, retained evaluation data and model-free regeneration."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from reporting import classification_evaluation, digest, load_bundle, manifest, step_record, write_bundle
from reporting.core import classification_metrics, read_jsonl, record_evaluation
from reporting.reports import refresh
from reporting.timing import OperationTimer, total_timing


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        self.run = manifest(run_id="fixture", method="sisa", dataset="fixture", model="tiny",
                            config={"epochs": 1}, split_sha256="split", workload_sha256="workload",
                            requested_steps=3, precision="fp32")
        self.steps = [step_record("fixture", 0, update_time_sec=100, training_time_sec=80, timing_scope="update_v1")]
        for index, seconds in enumerate((1., 3., 5.), 1):
            self.steps.append(step_record("fixture", index, deleted_count=1, deleted_total=index,
                                          source_ids=[index], update_time_sec=seconds, training_time_sec=0,
                                          timing_scope="update_v1", status="requires_retraining" if index == 3 else "ready"))
        self.evaluations = [self.evaluation(index) for index in range(4)]
        write_bundle(self.directory, self.run, self.steps, self.evaluations)

    def evaluation(self, index, split="test", state_id=None):
        return classification_evaluation(
            self.directory, run_id="fixture", state_id=state_id or f"step-{index}", step=index,
            split=split, dataset="fixture", labels=[0, 0, 1, 1],
            predictions=[0, 0, 0, 0] if index < 3 else None, num_classes=3,
            source_sha256="raw-input-sha", split_sha256="split",
            status="ready" if index < 3 else "requires_retraining")

    def rows(self, path):
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def test_initial_training_excluded_and_exhaustion_keeps_request_cost(self):
        refresh([self.directory], self.root / "reports", plots=False)
        row = self.rows(self.root / "reports/summary.csv")[0]
        self.assertEqual(float(row["initial_train_time_sec"]), 100)
        self.assertEqual(float(row["mean_update_time_sec"]), 3)
        self.assertEqual(float(row["median_update_time_sec"]), 3)
        self.assertAlmostEqual(float(row["p95_update_time_sec"]), 4.8)
        self.assertEqual(float(row["total_update_time_sec"]), 9)
        self.assertEqual(float(row["total_training_time_sec"]), 0)
        self.assertEqual(row["ready_steps"], "2")
        self.assertEqual(row["scored_steps"], "2")
        self.assertEqual(row["measured_steps"], "3")
        self.assertEqual(row["final_metric"], "")
        self.assertEqual(self.rows(self.root / "reports/steps.csv")[-1]["metric_value"], "")

    def test_refresh_recomputes_metric_and_figures_after_models_removed(self):
        checkpoints = self.directory / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "old.pt").write_bytes(b"disposable")
        shutil.rmtree(checkpoints)
        output = self.root / "reports"
        refresh([self.directory], output)
        original = (output / "comparison-1.png").read_bytes()
        sources = {name: (self.directory / name).read_bytes() for name in ("run.json", "steps.jsonl", "evaluations.jsonl")}
        refresh([self.directory], output, metric="macro_f1", x="p95_update_time_sec", y="initial_metric")
        self.assertNotEqual(original, (output / "comparison-1.png").read_bytes())
        self.assertAlmostEqual(float(self.rows(output / "summary.csv")[0]["initial_metric"]), 1 / 3)
        self.assertEqual(json.loads((output / "report_settings.json").read_text())["x"], "p95_update_time_sec")
        for name, contents in sources.items():
            self.assertEqual((self.directory / name).read_bytes(), contents)
        record = read_jsonl(self.directory / "evaluations.jsonl")[0]
        with np.load(self.directory / record["prediction_artifact"]) as arrays:
            np.testing.assert_array_equal(arrays["labels"], [0, 0, 1, 1])

    def test_missing_timing_does_not_become_zero_or_full_workload_total(self):
        self.steps[2]["update_time_sec"] = None
        write_bundle(self.directory, self.run, self.steps, self.evaluations)
        refresh([self.directory], self.root / "reports", plots=False)
        row = self.rows(self.root / "reports/summary.csv")[0]
        self.assertEqual(row["measured_steps"], "2")
        self.assertEqual(row["total_update_time_sec"], "")

    def test_stopped_workload_does_not_use_last_available_score_as_final(self):
        self.run["requested_steps"] = 5
        write_bundle(self.directory, self.run, self.steps[:3], self.evaluations[:3])
        refresh([self.directory], self.root / "reports", plots=False)
        row = self.rows(self.root / "reports/summary.csv")[0]
        self.assertEqual(row["final_metric"], "")
        self.assertEqual(row["total_update_time_sec"], "")
        self.assertEqual(row["final_step"], "2")

    def test_final_state_selects_post_retrain_evaluation_once(self):
        self.steps[3]["state_id"] = "generation-1-step-3"
        self.steps[3]["status"] = "ready"
        ready = self.evaluation(2, state_id="generation-1-step-3")
        ready["step"] = 3
        write_bundle(self.directory, self.run, self.steps, self.evaluations + [ready])
        refresh([self.directory], self.root / "reports", plots=False)
        rows = self.rows(self.root / "reports/steps.csv")
        self.assertEqual(len(rows), 4)
        self.assertEqual(float(rows[-1]["metric_value"]), .5)
        self.assertEqual(float(self.rows(self.root / "reports/summary.csv")[0]["total_update_time_sec"]), 9)

    def test_splits_are_never_substituted_and_unknown_metrics_require_measurements(self):
        with self.assertRaisesRegex(ValueError, "No saved val"):
            refresh([self.directory], self.root / "reports", split="val", plots=False)
        with self.assertRaisesRegex(ValueError, "No saved test/auroc"):
            refresh([self.directory], self.root / "reports", metric="auroc", plots=False)

    def test_conflicting_protocols_are_reported_and_strict_mode_rejects(self):
        other = self.root / "other"
        run = {**self.run, "run_id": "other", "workload_sha256": "different"}
        steps = [{**step, "run_id": "other"} for step in self.steps]
        evaluations = [{**item, "run_id": "other"} for item in self.evaluations]
        write_bundle(other, run, steps, evaluations)
        result = refresh([self.directory, other], self.root / "reports", plots=False)
        self.assertTrue(any("different workload_sha256" in note for note in result["notes"]))
        with self.assertRaisesRegex(ValueError, "Comparison validation failed"):
            refresh([self.directory, other], self.root / "strict", plots=False, strict=True)

    def test_invalid_counts_nonfinite_metrics_and_duplicate_states_rejected(self):
        steps = [dict(item) for item in self.steps]
        steps[1]["deleted_total"] = 7
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            write_bundle(self.directory, self.run, steps, self.evaluations)
        with self.assertRaisesRegex(ValueError, "Duplicate evaluation"):
            write_bundle(self.directory, self.run, self.steps, self.evaluations + [self.evaluations[0]])
        self.evaluations[0]["metrics"]["custom"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            write_bundle(self.directory, self.run, self.steps, self.evaluations)

    def test_repeated_evaluation_replaces_state_without_duplicate_rows(self):
        record_evaluation(self.directory, self.evaluations[0])
        self.assertEqual(len(read_jsonl(self.directory / "evaluations.jsonl")), 4)
        self.assertEqual(load_bundle(self.directory)[0]["run_id"], "fixture")

    def test_cli_saved_settings_refreshes_without_model_imports(self):
        settings = self.root / "settings.json"
        settings.write_text(json.dumps({"metric": "balanced_accuracy", "plots": False}))
        completed = subprocess.run([sys.executable, "-m", "reporting", "refresh", str(self.directory),
                                    "--output", str(self.root / "reports"), "--settings", str(settings)],
                                   capture_output=True, text=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(self.rows(self.root / "reports/summary.csv")[0]["metric_name"], "balanced_accuracy")

    def test_macro_metrics_handle_absent_classes(self):
        metrics = classification_metrics([[2, 0, 0], [2, 0, 0], [0, 0, 0]])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertAlmostEqual(metrics["macro_f1"], 1 / 3)
        self.assertEqual(metrics["balanced_accuracy"], .5)

    def test_study_refresh_reuses_external_inputs_and_updates_settings(self):
        import contextlib
        import io
        from reporting.study import main
        study = self.root / "study"
        study.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            main(study, ["--tag", "comparison", "--include", str(self.directory), "--no-plots"])
            main(study, ["--tag", "comparison", "--metric", "macro_f1", "--no-plots"])
        row = self.rows(study / "runs/comparison/reports/comparison/summary.csv")[0]
        self.assertEqual(row["metric_name"], "macro_f1")

    def test_legacy_sisa_export_keeps_accuracy_but_does_not_invent_f1_or_update_time(self):
        from comparison_export import export_sisa
        directory = self.root / "legacy"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"dataset": "datasets/fixture/datasetfile",
            "model": "tiny", "shards": 1, "deletion_steps": 1, "dataset_metadata": {"nb_test": 10}}))
        (directory / "report.csv").write_text("nb_requests,accuracy\n0,0.9\n1,0.8\n")
        for label in (0, 1):
            requests = np.empty(1, dtype=object)
            requests[0] = np.array([5] if label else [], dtype=np.int64)
            np.save(directory / f"requestfile:{label}.npy", requests, allow_pickle=True)
        export_sisa(directory)
        run, steps, evaluations = load_bundle(directory)
        self.assertEqual(run["measurement_origin"], "historical_export")
        self.assertEqual(steps[-1]["timing_scope"], "legacy_batch_last_slice")
        self.assertIsNone(steps[-1]["update_time_sec"])
        self.assertEqual(evaluations[-1]["metrics"], {"accuracy": .8})
        with self.assertRaisesRegex(ValueError, "No saved test/macro_f1"):
            refresh([directory], self.root / "legacy-reports", metric="macro_f1", plots=False)

    def test_timing_excludes_diagnostics_accumulates_and_marks_unclean_intervals(self):
        value = [0.]
        path = self.root / "timer.json"
        with OperationTimer(path, lambda: value[0]) as timer:
            value[0] += 10
            timer.exclude(3)
            timer.add_training(2)
        with OperationTimer(path, lambda: value[0]):
            value[0] += 4
        self.assertEqual(total_timing([path])["update_time_sec"], 11)
        record = json.loads(path.read_text())
        record["status"] = "running"
        path.write_text(json.dumps(record))
        with OperationTimer(path, lambda: value[0]):
            value[0] += 1
        self.assertIsNone(total_timing([path])["update_time_sec"])


class S3TComparisonTests(unittest.TestCase):
    def test_retraining_cost_counted_once_and_export_survives_checkpoint_deletion(self):
        import torch
        from s3t.config import Config
        from s3t.engine import Run
        from s3t.artifacts import atomic_json
        from deletion_requests import generate_workload
        from comparison_export import export_s3t
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config(dataset="synthetic", model="tiny_vit", pretrained=False, shards=1, slices=1,
                            budget=1, rank=2, alpha=4, epochs_per_slice=1, batch_size=16, eval_batch_size=24,
                            num_workers=0, lora_dropout=0)
            run = Run.create(root / "run", config)
            run.train("cpu")
            before = {path.name: path.read_bytes() for path in (run.directory / "comparison_timing").glob("*.json")}
            run.train("cpu")
            self.assertEqual(before, {path.name: path.read_bytes() for path in (run.directory / "comparison_timing").glob("*.json")})
            requests = root / "requests.json"
            atomic_json(requests, generate_workload(run.split, steps=1))
            run.replay(requests, "cpu", on_exhaustion="retrain")
            meta, steps, evaluations = load_bundle(run.directory)
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[-1]["status"], "ready")
            self.assertEqual(steps[-1]["method_details"]["generation"], 1)
            journals = list((run.directory / "comparison_timing").glob("*.json"))
            expected = sum(json.loads(path.read_text())["update_time_sec"] for path in journals
                           if not path.name.startswith(("g0-", "initial-")))
            self.assertAlmostEqual(steps[-1]["update_time_sec"], expected)
            self.assertGreater(steps[-1]["update_time_sec"], steps[-1]["method_details"]["planning_time_sec"])
            self.assertEqual(len([item for item in evaluations if item["step"] == 1]), 2)
            for path in run.directory.rglob("*.pt"):
                path.unlink()
            export_s3t(run.directory)
            result = refresh([run.directory], root / "reports", metric="macro_f1", plots=False)
            self.assertEqual(result["steps"], 2)


if __name__ == "__main__":
    unittest.main()
