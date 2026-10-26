import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_splits import get_manifest, fingerprint_records
from deletion_requests import checksum, generate_workload, load_workload, local_batches, validate_workload


class DeletionWorkloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = patch.dict(os.environ, DATASET_SPLIT_DIR=str(self.root / "splits"))
        environment.start()
        self.addCleanup(environment.stop)
        labels = [0, 1] * 50
        self.split, _ = get_manifest("fixture", labels, 20, fingerprint_records(labels),
                                     path=None)

    def workload(self):
        return generate_workload(self.split, seed=123, steps=5, batch_size=3)

    def test_independent_rng_and_unique_nonvalidation_source_ids(self):
        random.seed(99)
        before = random.getstate()
        first = self.workload()
        self.assertEqual(before, random.getstate())
        random.seed(1)
        self.assertEqual(first, self.workload())
        ids = sum(first["batches"], [])
        self.assertEqual(len(ids), 15)
        self.assertEqual(len(set(ids)), 15)
        self.assertFalse(set(ids) & set(self.split["val_indices"]))

    def test_roundtrip_and_reordered_local_training_pool(self):
        workload = self.workload()
        path = self.root / "requests.json"
        path.write_text(json.dumps(workload))
        self.assertEqual(load_workload(path, self.split), workload)
        sources = self.split["train_indices"][::-1]
        batches = local_batches(workload, self.split, sources)
        self.assertEqual([[sources[idx] for idx in batch] for batch in batches], workload["batches"])

    def test_rejects_corruption_and_wrong_split(self):
        for key, value in (("dataset_id", "other"), ("source_sha256", "other"),
                           ("split_manifest_sha256", "other"), ("schema_version", 2)):
            workload = dict(self.workload(), **{key: value})
            workload["workload_sha256"] = checksum(workload)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_workload(workload, self.split)
        workload = dict(self.workload(), deletion_seed=0)
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_workload(workload, self.split)

    def test_rejects_duplicate_validation_empty_and_noninteger_requests(self):
        original = self.workload()
        invalid_batches = [[], [[]], [[self.split["val_indices"][0]]], [[True]],
                           [[original["batches"][0][0]]] * 2]
        for batches in invalid_batches:
            workload = copy.deepcopy(original)
            workload["batches"] = batches
            workload["workload_sha256"] = checksum(workload)
            with self.subTest(batches=batches), self.assertRaises(ValueError):
                validate_workload(workload, self.split)

    def test_rejects_caps_duplicate_pool_and_excessive_generation(self):
        for sources in (self.split["train_indices"][:-1], [self.split["train_indices"][0]] * 80):
            with self.assertRaisesRegex(ValueError, "full training pool"):
                local_batches(self.workload(), self.split, sources)
        for steps, batch_size in ((81, 1), (0, 1), (1, 0), (True, 1)):
            with self.assertRaises(ValueError):
                generate_workload(self.split, steps=steps, batch_size=batch_size)


if __name__ == "__main__":
    unittest.main()
