import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from container_storage import (checkpoint_summary, clear_models, initialize_container,
                               record_directory, separate_storage)


ROOT = Path(__file__).resolve().parents[1]


class ContainerStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        self.external = self.root / "external"
        self.external.mkdir()

    def test_checkpoints_are_external_and_cleanup_keeps_records_and_size(self):
        Path("saved_models").symlink_to(self.external, target_is_directory=True)
        directory = initialize_container("fixture")
        self.assertFalse(Path("containers").is_symlink())
        self.assertEqual((directory / "cache").resolve(), self.external / "fixture/cache")
        (directory / "cache/slice.pt").write_bytes(b"weights")
        (directory / "cache/shard-0:0.pt").symlink_to("slice.pt")
        (directory / "times/slice.time").write_text("2.5\n")
        (directory / "times/shard-0:0.time").symlink_to("slice.time")
        for name in ("config.json", "report.csv", "outputs/predictions.npy", "splitfile.npy", "deletion_requests.json"):
            (directory / name).write_bytes(b"retained")
        before = {str(path): path.read_bytes() for path in directory.rglob("*")
                  if path.is_file() and "cache" not in path.parts}
        self.assertEqual(checkpoint_summary("fixture")["checkpoint_bytes"], 7)
        self.assertEqual(clear_models(["fixture"]), 2)
        self.assertEqual(list((directory / "cache").iterdir()), [])
        self.assertEqual(checkpoint_summary("fixture")["checkpoint_bytes"], 7)
        for path, content in before.items():
            self.assertEqual(Path(path).read_bytes(), content)
        initialize_container("fixture")
        self.assertEqual((directory / "times/slice.time").read_text(), "2.5\n")

    def test_legacy_separation_copies_records_without_changing_external_files(self):
        legacy = self.external / "fixture"
        for name in ("cache", "times", "outputs"):
            (legacy / name).mkdir(parents=True)
        (legacy / "cache/slice.pt").write_bytes(b"weights")
        (legacy / "cache/shard-0:0.pt").symlink_to("slice.pt")
        (legacy / "times/slice.time").write_text("1.5\n")
        (legacy / "times/shard-0:0.time").symlink_to((legacy / "times/slice.time").resolve())
        (legacy / "outputs/scores.npy").write_bytes(b"predictions")
        (legacy / "splitfile.npy").write_bytes(b"split")
        Path("containers").symlink_to(self.external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "separate first"):
            initialize_container("new")
        separate_storage()
        self.assertFalse(Path("containers").is_symlink())
        self.assertEqual(Path("saved_models").resolve(), self.external)
        self.assertEqual((legacy / "cache/slice.pt").read_bytes(), b"weights")
        self.assertTrue((legacy / "times/shard-0:0.time").is_symlink())
        local = Path("containers/fixture")
        self.assertFalse((local / "times/shard-0:0.time").is_symlink())
        initialize_container("fixture")
        # Deleting the entire old drive tree must leave local records usable.
        shutil.rmtree(self.external)
        self.assertEqual((local / "times/shard-0:0.time").read_text(), "1.5\n")
        self.assertEqual((local / "outputs/scores.npy").read_bytes(), b"predictions")
        self.assertEqual(checkpoint_summary("fixture")["checkpoint_bytes"], 7)
        separate_storage()  # Repeating setup must not change the local layout.

    def test_conflicts_are_rejected_before_replacing_legacy_link(self):
        Path("containers").symlink_to(self.external, target_is_directory=True)
        Path("saved_models").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            separate_storage()
        self.assertEqual(Path("containers").resolve(), self.external)

    def test_invalid_names_and_redirected_records_are_rejected(self):
        for name in ("", "..", "../outside", "/absolute"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                record_directory(name)
        Path("containers").mkdir()
        Path("containers/fixture").symlink_to(self.external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "records must be local"):
            clear_models(["fixture"])

    def test_real_training_reporting_and_reporting_again_after_cleanup(self):
        Path("saved_models").symlink_to(self.external, target_is_directory=True)
        for name in ("sisa.py", "distribution.py", "aggregation.py", "time.py", "container_storage.py", "sisa_timing.py"):
            Path(name).symlink_to(ROOT / name)
        fixture = Path("fixture_data")
        fixture.mkdir()
        (fixture / "__init__.py").write_text("")
        (fixture / "datasetfile").write_text(json.dumps({
            "nb_train": 8, "nb_test": 4, "nb_classes": 2,
            "input_shape": [2], "dataloader": "dataloader",
        }))
        (fixture / "dataloader.py").write_text(
            "import numpy as np\n"
            "def load(indices, category='train'):\n"
            "    labels = np.asarray(indices, dtype=np.int64) % 2\n"
            "    return np.eye(2, dtype=np.float32)[labels], labels\n"
        )
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "PYTHONPATH": os.pathsep.join((str(self.root), str(ROOT)))}
        environment.pop("DELETION_REQUEST_MANIFEST", None)
        command = ["bash", str(ROOT / "example-scripts/common/run_all.sh"), "purchase",
                   "fixture_data/datasetfile", "2", "1", "fixture", "2", "2", "2",
                   "0.01", "sgd", "1", "2", "combined-report"]
        completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=90)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        directory = Path("containers/fixture")
        config = json.loads((directory / "config.json").read_text())
        self.assertEqual(config["hyperparams"]["slices"], 2)
        self.assertGreater(checkpoint_summary("fixture")["checkpoint_bytes"], 0)
        report = (directory / "report.csv").read_bytes()
        with (directory / "report.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["nb_requests"] for row in rows], ["0", "1"])
        cleanup = subprocess.run(["bash", str(ROOT / "clear_containers.sh"), "fixture"],
                                 env=environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(cleanup.returncode, 0, cleanup.stderr)
        self.assertEqual(list((directory / "cache").glob("*.pt")), [])
        regenerated = subprocess.run(["bash", str(ROOT / "example-scripts/common/data.sh"),
                                     "fixture_data/datasetfile", "2", "1", "fixture", "regenerated-report",
                                     json.dumps(config["hyperparams"], separators=(",", ":"))],
                                    env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(regenerated.returncode, 0, regenerated.stderr)
        self.assertEqual((directory / "report.csv").read_bytes(), report)
        self.assertEqual(Path("regenerated-report.csv").read_bytes(), report)


if __name__ == "__main__":
    unittest.main()
