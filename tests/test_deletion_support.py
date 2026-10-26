import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset_splits import load_manifest
from deletion_requests import generate_workload, local_batches
from deletion_support import export_deletion_requests


class DeletionExportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        self.split = load_manifest(ROOT / "splits/cifar100-val2500-seed42.json")
        self.metadata = self.root / "datasetfile"
        self.metadata.write_text(json.dumps(dict(nb_train=len(self.split["train_indices"]),
            nb_val=len(self.split["val_indices"]),
            split_manifest=str(ROOT / "splits/cifar100-val2500-seed42.json"))))
        self.workload = generate_workload(self.split, steps=4, batch_size=3)
        self.workload_path = self.root / "requests.json"
        self.workload_path.write_text(json.dumps(self.workload))
        self.container = self.root / "containers/fixture"
        result = self.cli("--shards", "5", "--label", "0")
        self.assertEqual(result.returncode, 0, result.stderr)

    def cli(self, *arguments):
        return subprocess.run([sys.executable, str(ROOT / "distribution.py"),
                               "--container", "fixture", "--dataset", str(self.metadata), *arguments],
                              capture_output=True, text=True)

    def test_cumulative_labels_match_batches_and_export_is_repeatable(self):
        batches = local_batches(self.workload, self.split)
        for _ in range(2):
            export_deletion_requests("fixture", self.metadata, self.workload_path)
        cumulative = set()
        for label in range(5):
            if label:
                cumulative.update(batches[label - 1])
            requests = np.load(self.container / f"requestfile:{label}.npy", allow_pickle=True)
            actual = [int(idx) for shard in requests for idx in shard]
            self.assertEqual(len(actual), label * 3)
            self.assertEqual(set(actual), cumulative)

    def test_different_workload_random_overwrite_and_resharding_are_rejected(self):
        export_deletion_requests("fixture", self.metadata, self.workload_path)
        self.workload_path.write_text(json.dumps(generate_workload(self.split, seed=999, steps=4)))
        with self.assertRaisesRegex(ValueError, "different deletion workload"):
            export_deletion_requests("fixture", self.metadata, self.workload_path)
        for arguments in (("--requests", "1"), ("--shards", "2")):
            result = self.cli(*arguments)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shared deletion", result.stderr.lower())

    def test_conflicting_request_labels_and_invalid_shards_are_rejected(self):
        requests = np.empty(5, dtype=object)
        requests[:] = [np.array([1]), np.array([], dtype=int), np.array([], dtype=int),
                       np.array([], dtype=int), np.array([], dtype=int)]
        np.save(self.container / "requestfile:1.npy", requests, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "Existing requests differ"):
            export_deletion_requests("fixture", self.metadata, self.workload_path)
        self.assertFalse((self.container / "deletion_requests.json").exists())
        original_shards = np.load(self.container / "splitfile.npy", allow_pickle=True)
        shards = np.empty(len(original_shards), dtype=object)
        shards[:] = [np.asarray(shard, dtype=int) for shard in original_shards]
        shards[0] = shards[0][:-1]
        np.save(self.container / "splitfile.npy", shards, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "full training pool"):
            export_deletion_requests("fixture", self.metadata, self.workload_path)

    def test_cli_exports_shared_batches_and_rejects_ambiguous_options(self):
        result = self.cli("--request-manifest", str(self.workload_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["workload_sha256"], self.workload["workload_sha256"])
        result = self.cli("--request-manifest", str(self.workload_path), "--requests", "1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cannot be combined", result.stderr)


if __name__ == "__main__":
    unittest.main()
