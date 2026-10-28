import json
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from container_storage import clear_models, initialize_container, record_directory
from studies.artifacts import capture_run, create_sisa_records, execute_plan, study_directory, verify_run_source
from studies.migrate import SHARED, migrate, remove_source_links


ROOT = Path(__file__).resolve().parents[1]


class StudyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)

    def test_launch_snapshot_and_resume_are_bound_to_source_and_settings(self):
        study = self.root / "studies/2026-10-28_1200_fixture"
        study.mkdir(parents=True)
        source = self.root / "engine.py"
        source.write_text("VALUE = 1\n")
        (study / "run.py").write_text("# fixture driver\n")
        output = study / "runs/first"
        settings = {"epochs": 2, "split_sha256": "saved-split"}
        first = capture_run(output, study, settings, self.root)
        manifest = (output / "study_run.json").read_bytes()
        self.assertEqual(first["kind"], "launch")
        self.assertEqual((output / "launch_source/engine.py").read_text(), "VALUE = 1\n")
        self.assertEqual(capture_run(output, study, settings, self.root), first)
        verify_run_source(output / "s3t-fixture", self.root)
        with self.assertRaisesRegex(ValueError, "source or settings changed"):
            capture_run(output, study, {"epochs": 3}, self.root)
        source.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "source or settings changed"):
            capture_run(output, study, settings, self.root)
        with self.assertRaisesRegex(ValueError, "source changed since launch"):
            verify_run_source(output / "s3t-fixture", self.root)
        self.assertEqual((output / "study_run.json").read_bytes(), manifest)
        self.assertEqual((output / "launch_source/engine.py").read_text(), "VALUE = 1\n")

    def test_legacy_output_is_not_assigned_current_training_provenance(self):
        study = self.root / "studies/2026-10-28_1200_fixture"
        output = study / "runs/old"
        output.mkdir(parents=True)
        (output / "results.csv").write_text("accuracy\n0.9\n")
        with self.assertRaisesRegex(ValueError, "no study launch provenance"):
            capture_run(output, study, {}, self.root)
        (output / "study_run.json").write_text(json.dumps({"kind": "historical_import"}))
        with self.assertRaisesRegex(ValueError, "historical study run"):
            capture_run(output, study, {}, self.root)
        self.assertFalse((output / "launch_source").exists())

    def test_sisa_records_can_live_in_a_study_without_moving_external_models(self):
        external = self.root / "model-drive"
        external.mkdir()
        Path("saved_models").symlink_to(external, target_is_directory=True)
        destination = self.root / SHARED / "runs/test/sisa-fixture"
        create_sisa_records(destination, self.root)
        directory = initialize_container(destination)
        model_path = external / destination.relative_to(self.root) / "cache"
        self.assertEqual(directory.resolve(), destination)
        self.assertEqual((directory / "cache").resolve(), model_path)
        (directory / "cache/weights.pt").write_bytes(b"weights")
        (directory / "report.csv").write_text("accuracy\n0.9\n")
        clear_models([destination])
        self.assertEqual((destination / "report.csv").read_text(), "accuracy\n0.9\n")
        self.assertEqual(list(model_path.iterdir()), [])
        self.assertFalse((self.root / "containers").exists())

    def test_migration_preserves_files_checkpoint_links_and_historical_identity(self):
        legacy = self.root / "runs/shared_comparison/comparison_v1"
        legacy.mkdir(parents=True)
        (legacy / "plan.json").write_text('{"old_plan":true}\n')
        models = self.root / "saved_models/comparison_v1_sisa_cifar10/cache"
        models.mkdir(parents=True)
        (models / "slice.pt").write_bytes(b"large-model-placeholder")
        container = self.root / "containers/comparison_v1_sisa_cifar10"
        container.mkdir(parents=True)
        (container / "cache").symlink_to("../../saved_models/comparison_v1_sisa_cifar10/cache", target_is_directory=True)
        (container / "report.csv").write_text("accuracy\n0.8\n")
        report = self.root / "report-shared25-comparison_v1-cifar10.csv"
        report.write_text("accuracy\n0.8\n")
        original_inode = (container / "report.csv").stat().st_ino
        preview = migrate(self.root)
        self.assertEqual(len(preview), 3)
        self.assertFalse(container.is_symlink())
        result = migrate(self.root, apply=True)
        self.assertEqual(len(result), 3)
        canonical = self.root / SHARED / "runs/comparison_v1"
        self.assertEqual((canonical / "plan.json").read_text(), '{"old_plan":true}\n')
        self.assertFalse(legacy.exists())
        self.assertEqual((canonical / "sisa-cifar10/report.csv").stat().st_ino, original_inode)
        self.assertEqual((canonical / "sisa-cifar10/cache").resolve(), models)
        self.assertEqual((canonical / "sisa-cifar10/cache/slice.pt").read_bytes(), b"large-model-placeholder")
        self.assertEqual(record_directory(canonical / "sisa-cifar10"), canonical / "sisa-cifar10")
        self.assertFalse(container.exists())
        manifest = json.loads((canonical / "study_run.json").read_text())
        self.assertEqual(manifest["kind"], "historical_import")
        self.assertEqual(manifest["compatibility"], "unverified")
        self.assertNotIn("source_sha256", manifest)
        self.assertEqual((canonical / "reports/sisa-cifar10.csv").read_text(), "accuracy\n0.8\n")
        self.assertFalse(report.exists())
        self.assertEqual(migrate(self.root, apply=True), [])

    def test_stable_imports_use_dated_studies_without_legacy_entry_points(self):
        from studies.shared_comparison import run as comparison
        from studies.s3t_lr_screening import run as screening
        from studies.s3t_validation import run as validation
        self.assertEqual(Path(comparison.__file__).parent, study_directory("shared_comparison"))
        self.assertEqual(Path(screening.__file__).parent, study_directory("s3t_lr_screening"))
        self.assertEqual(Path(validation.__file__).parent, study_directory("s3t_validation"))
        for path in (ROOT / "run_shared_comparison.py", ROOT / "s3t/lr_search.py"):
            self.assertFalse(path.exists())
            self.assertFalse(path.is_symlink())
        for script in ("shared_comparison", "s3t_lr_screening", "s3t_validation"):
            result = subprocess.run(["bash", str(study_directory(script) / "run.sh"), "--help"],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_active_run_lock_prevents_migration(self):
        legacy = self.root / "runs/s3t-validation"
        legacy.mkdir(parents=True)
        with (legacy / ".lifecycle.lock").open("w") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "active process"):
                migrate(self.root, apply=True)
        self.assertTrue(legacy.is_dir())
        self.assertFalse(legacy.is_symlink())

    def test_source_cleanup_removes_only_retired_entry_points(self):
        study = self.root / SHARED
        study.mkdir(parents=True)
        (study / "run.py").write_text("# canonical study\n")
        old = self.root / "run_shared_comparison.py"
        old.symlink_to(Path(SHARED) / "run.py")
        models = self.root / "model-drive"
        models.mkdir()
        (self.root / "saved_models").symlink_to(models, target_is_directory=True)
        self.assertEqual(len(remove_source_links(self.root, apply=True)), 1)
        self.assertFalse(old.exists())
        self.assertFalse(old.is_symlink())
        self.assertEqual((study / "run.py").read_text(), "# canonical study\n")
        self.assertTrue((self.root / "saved_models").is_symlink())
        self.assertEqual(remove_source_links(self.root, apply=True), [])

    def test_study_executor_records_the_plan_and_subprocess_result(self):
        study = self.root / "studies/2026-10-28_1200_fixture"
        study.mkdir(parents=True)
        output = study / "runs/first"
        script = "import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_text(os.environ['STUDY_TEST'])"
        commands = [{"argv": [sys.executable, "-c", script, str(output / "result.txt")],
                     "env": {"STUDY_TEST": "recorded"}}]
        settings = {"commands": commands, "epochs": 1}
        self.assertEqual(execute_plan(output, study, commands, settings, self.root), 0)
        self.assertEqual((output / "result.txt").read_text(), "recorded")
        self.assertEqual(json.loads((output / "plan.json").read_text()), settings)
        self.assertEqual(json.loads((output / "execution.jsonl").read_text())["returncode"], 0)
        self.assertEqual(json.loads((output / "study_run.json").read_text())["settings"], settings)


if __name__ == "__main__":
    unittest.main()
