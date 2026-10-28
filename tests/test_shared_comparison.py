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
