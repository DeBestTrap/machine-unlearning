import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from dataset_loading import load_dataset_module
from sharded import fetchShardBatch, fetchTestBatch


ROOT = Path(__file__).resolve().parents[1]


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original_cwd)
        self.directory = self.root / "machine-unlearning with spaces/datasets/fixture"
        self.directory.mkdir(parents=True)
        self.metadata = {"dataloader": "dataloader", "nb_train": 5, "nb_test": 3, "nb_val": 2}
        self.dataset = self.directory / "datasetfile"
        self.dataset.write_text(json.dumps(self.metadata))
        (self.directory / "__init__.py").write_text("FACTOR = 2\n")
        (self.directory / "helper.py").write_text("BIAS = 0\n")
        (self.directory / "dataloader.py").write_text(
            "import numpy as np\n"
            "from . import FACTOR\n"
            "from .helper import BIAS\n"
            "def load(indices, category='train'):\n"
            "    indices = np.asarray(indices, dtype=np.int64)\n"
            "    return indices, (indices + BIAS) % FACTOR\n"
        )

    def test_absolute_relative_and_normalized_paths_share_a_loader(self):
        absolute = load_dataset_module(self.dataset, self.metadata)
        for path in (self.dataset, str(self.dataset), self.dataset.relative_to(self.root),
                     self.directory / "../fixture/datasetfile"):
            with self.subTest(path=path):
                self.assertIs(load_dataset_module(path, self.metadata), absolute)
        self.assertEqual(Path(absolute.__file__).resolve(), self.directory / "dataloader.py")
        np.testing.assert_array_equal(absolute.load([0, 1, 2])[1], [0, 1, 0])

    def test_identically_named_loaders_in_different_directories_stay_separate(self):
        other = self.root / "another-dataset"
        other.mkdir()
        (other / "datasetfile").write_text(json.dumps(self.metadata))
        (other / "dataloader.py").write_text("VALUE = 123\n")
        first = load_dataset_module(self.dataset, self.metadata)
        second = load_dataset_module(other / "datasetfile", self.metadata)
        self.assertIsNot(first, second)
        self.assertEqual(second.VALUE, 123)

    def test_training_and_evaluation_batches_accept_absolute_metadata(self):
        directory = Path("containers/fixture")
        directory.mkdir(parents=True)
        shards = np.empty(1, dtype=object)
        shards[0] = np.arange(5)
        requests = np.empty(1, dtype=object)
        requests[0] = np.array([1])
        np.save(directory / "splitfile.npy", shards, allow_pickle=True)
        np.save(directory / "requestfile:0.npy", requests, allow_pickle=True)
        batches = list(fetchShardBatch("fixture", "0", 0, 2, self.dataset))
        np.testing.assert_array_equal(np.concatenate([batch[0] for batch in batches]), [0, 2, 3, 4])
        for split, count in (("test", 3), ("val", 2)):
            with self.subTest(split=split):
                batches = list(fetchTestBatch(self.dataset, 2, category=split))
                np.testing.assert_array_equal(np.concatenate([batch[0] for batch in batches]), np.arange(count))

    def test_aggregation_accepts_absolute_metadata_in_a_separate_process(self):
        outputs = Path("containers/fixture/outputs")
        outputs.mkdir(parents=True)
        predictions = np.eye(2)[np.arange(3) % 2]
        np.save(outputs / "shard-0:0.npy", predictions)
        result = subprocess.run([sys.executable, str(ROOT / "aggregation.py"), "--dataset", str(self.dataset),
                                 "--container", "fixture", "--shards", "1", "--label", "0"],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(float(result.stdout), 1.0)


if __name__ == "__main__":
    unittest.main()
