"""Validate shared comparison plans without GPUs, model loading, or training."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.shared_comparison.run import STUDY, STUDY_RELATIVE, build_plan, load_protocol


class SharedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.protocol = load_protocol(STUDY / "configs/quick25.json", ROOT)

    def test_sisa_and_s3t_replay_same_workload(self):
        plan = build_plan(self.protocol, ROOT, ["sisa", "s3t"], list(self.protocol["datasets"]), "unit", "cuda:2")
        self.assertEqual(len(plan), 12)
        for name in self.protocol["datasets"]:
            sisa, train, replay, audit = [item for item in plan if item["dataset"] == name]
            self.assertEqual(sisa["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(sisa["env"]["DELETION_REQUEST_MANIFEST"],
                             replay["argv"][replay["argv"].index("--requests") + 1])
            self.assertIn("max_train_examples=null", train["argv"])
            self.assertIn("max_eval_examples=null", train["argv"])
            self.assertEqual(sisa["argv"][8], "1")
            self.assertEqual(Path(sisa["argv"][6]), STUDY / "runs/unit" / f"sisa-{name}")
            self.assertIn("epochs_per_slice=1", train["argv"])
            self.assertEqual(audit["argv"][3], "audit")
            self.assertTrue(Path(train["argv"][train["argv"].index("--output") + 1]).is_relative_to(STUDY / "runs"))

    def test_mismatched_workload_checksum_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["datasets"]["cifar10"]["workload_sha256"] = "bad"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "workload checksum differs"):
                load_protocol(path, ROOT)

    def test_protosisa_uses_paper_shards_safe_weighting_and_shared_workload(self):
        plan = build_plan(self.protocol, ROOT, ["protosisa"], list(self.protocol["datasets"]), "unit", "cuda:2")
        for item, shards, slices in zip(plan, (10, 100, 2), (1, 1, 2)):
            argv = item["argv"]
            def value(flag):
                return argv[argv.index(flag) + 1]
            self.assertEqual(item["method"], "protosisa")
            self.assertEqual(item["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertEqual(value("--device"), "cuda:0")
            self.assertEqual(int(value("--shards")), shards)
            self.assertEqual(int(value("--slices")), slices)
            self.assertEqual(value("--prototype-weighting"), "safe")
            self.assertNotIn("--prototype-weight", argv)
            self.assertNotIn("--prototype-temperature", argv)
            self.assertEqual(int(value("--seed")), self.protocol["training_seed"])
            self.assertEqual(Path(value("--requests")), ROOT / "deletions" / self.protocol["datasets"][item["dataset"]]["workload"])
            self.assertEqual(Path(value("--container")), STUDY / "runs/unit" / f"protosisa-{item['dataset']}")

    def test_protosisa_fixed_variants_are_explicit_and_safe_rejects_tuning(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["protosisa"].update(prototype_weighting="fixed", prototype_weight=0.5)
        argv = build_plan(protocol, ROOT, ["protosisa"], ["cifar10"], "unit", "0")[0]["argv"]
        self.assertEqual(argv[argv.index("--prototype-weight") + 1], "0.5")
        self.assertNotIn("--prototype-temperature", argv)
        protocol["protosisa"]["prototype_weighting"] = "safe"
        with self.assertRaisesRegex(ValueError, "computes its own weight"):
            build_plan(protocol, ROOT, ["protosisa"], ["cifar10"], "unit", "0")

    def test_obsolete_temperature_setting_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["protosisa"]["prototype_temperature"] = 0.05
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            build_plan(protocol, ROOT, ["protosisa"], ["cifar10"], "unit", "0")

    def test_dry_run_from_another_directory_has_no_side_effects(self):
        tag = "unit_dry_run_no_artifacts"
        output = STUDY / "runs" / tag
        self.assertFalse(output.exists())
        result = subprocess.run([sys.executable, str(STUDY / "run.py"),
                                 "--dry-run", "--tag", tag], cwd="/tmp", capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("25 shared deletions", result.stdout)
        self.assertFalse(output.exists())

    def test_methods_from_other_repository_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Methods for this repository"):
            build_plan(self.protocol, ROOT, ["fcm"], ["cifar10"], "unit", "0")

    def test_existing_record_directory_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Dataset metadata still resolves the production shared split.
            dataset_dir = root / "datasets/cifar10"
            dataset_dir.mkdir(parents=True)
            original = json.loads((ROOT / "datasets/cifar10/datasetfile").read_text())
            original["split_manifest"] = str(ROOT / "splits" / self.protocol["datasets"]["cifar10"]["split"])
            (dataset_dir / "datasetfile").write_text(json.dumps(original))
            (root / STUDY_RELATIVE / "runs/unit/sisa-cifar10").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "records exist"):
                build_plan(self.protocol, root, ["sisa"], ["cifar10"], "unit", "0")


if __name__ == "__main__":
    unittest.main()
