import argparse
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from container_storage import clear_models, initialize_container
from sisa_timing import TIMING_SCOPE, UpdateTimer, summarize, write_reports
from sisa_training import synchronized_time, train_shard


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value


class SISATimingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        self.directory = initialize_container("fixture")
        shards = np.empty(1, dtype=object)
        shards[0] = np.arange(6)
        requests = np.empty(1, dtype=object)
        requests[0] = np.array([], dtype=int)
        np.save(self.directory / "splitfile.npy", shards, allow_pickle=True)
        for label in ("0", "1"):
            np.save(self.directory / f"requestfile:{label}.npy", requests, allow_pickle=True)
        Path("datasetfile").write_text(json.dumps({"dataloader": "fixture_loader"}))
        self.args = argparse.Namespace(container="fixture", model="fixture", dataset="datasetfile",
            shard=0, label="0", slices=3, epochs=4, batch_size=2, learning_rate=0.01,
            optimizer="sgd", dropout_rate=0.0, chkpt_interval=1)
        self.clock = Clock()
        self.save = torch.save
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def run_training(self, fail_step=None):
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        step = optimizer.step
        steps = 0
        def timed_step():
            nonlocal steps
            steps += 1
            if steps == fail_step:
                raise RuntimeError("interrupted training")
            step()
            self.clock.value += 1.0
        optimizer.step = timed_step
        def timed_save(*args, **kwargs):
            self.save(*args, **kwargs)
            self.clock.value += 0.5
        loader = Mock()
        loader.load.side_effect = lambda indices: (
            np.eye(2, dtype=np.float32)[np.asarray(indices, dtype=int) % 2],
            np.asarray(indices, dtype=np.int64) % 2)
        with patch("sisa_training.synchronized_time", side_effect=lambda device: self.clock()), \
             patch("sisa_training.torch.save", side_effect=timed_save), \
             patch("sharded.importlib.import_module", return_value=loader):
            return train_shard(self.args, model, optimizer, torch.nn.CrossEntropyLoss(), torch.device("cpu"))

    def test_every_slice_and_checkpoint_write_is_included(self):
        result = self.run_training()
        self.assertEqual([item["training_time_sec"] for item in result["slices"]], [2, 4, 6])
        self.assertEqual(result["training_time_sec"], 12)
        self.assertEqual(result["update_time_sec"], 16.5)
        self.assertEqual(float((self.directory / "times/shard-0:0.time").read_text()), 16.5)
        self.assertFalse((self.directory / "times/shard-0:0.time").is_symlink())
        self.assertEqual(summarize("fixture", "0", 1)["update_time_sec"], 16.5)
        self.assertEqual(self.run_training(), result)  # Reusing the label preserves its measurement.
        clear_models(["fixture"])
        self.assertEqual(summarize("fixture", "0", 1)["update_time_sec"], 16.5)

    def test_reused_slices_do_not_reuse_an_earlier_requests_training_cost(self):
        self.run_training()
        self.args.label = "1"
        result = self.run_training()
        self.assertEqual(result["training_time_sec"], 0)
        self.assertTrue(all(item["reused"] for item in result["slices"]))
        self.assertEqual(summarize("fixture", "0", 1)["training_time_sec"], 12)

    def test_interrupted_training_resumes_after_last_completed_epoch(self):
        # Two batches finish slice 0; two more finish the first epoch of slice 1.
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.run_training(fail_step=5)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize("fixture", "0", 1)
        result = self.run_training()
        self.assertEqual(result["training_time_sec"], 12)
        self.assertEqual(sum(item["epochs_trained"] for item in result["slices"]), 6)
        self.assertTrue(result["measurement_complete"])
        self.assertEqual(result["update_time_sec"], 16.5)

    def test_missing_shard_or_unclean_timing_is_not_reported_as_zero(self):
        self.run_training()
        with self.assertRaisesRegex(ValueError, "Missing complete"):
            summarize("fixture", "0", 2)
        path = self.directory / "times/shard-0:0.json"
        record = json.loads(path.read_text())
        record["status"] = "running"
        path.write_text(json.dumps(record))
        timer = UpdateTimer(self.directory, 0, "0", record["binding"], self.clock)
        with timer:
            pass
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            summarize("fixture", "0", 1)

    def test_report_keeps_legacy_scope_separate_and_sums_all_shards(self):
        self.run_training()
        record = json.loads((self.directory / "times/shard-0:0.json").read_text())
        record["shard"] = 1
        record["update_time_sec"] = 7.0
        record["training_time_sec"] = 5.0
        (self.directory / "times/shard-1:0.json").write_text(json.dumps(record))
        output = Path("combined.csv")
        output.write_text('nb_shards,nb_requests,accuracy,retraining_time,hyperparams_json\n2,0,0.5,3.0,"{}"\n')
        with patch("sisa_timing.subprocess.check_output", return_value="0.75\n"):
            write_reports("fixture", "datasetfile", 2, 0, output, "{}")
        with output.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["timing_scope"], "legacy_batch_last_slice")
        self.assertEqual(rows[0]["update_time_sec"], "")
        self.assertEqual(float(rows[1]["update_time_sec"]), 23.5)
        self.assertEqual(float(rows[1]["retraining_time"]), 23.5)
        self.assertEqual(float(rows[1]["training_time_sec"]), 17)
        self.assertEqual(rows[1]["timing_scope"], TIMING_SCOPE)

    def test_cuda_clock_synchronizes_before_reading_time(self):
        device = torch.device("cuda:0")
        with patch("sisa_training.torch.cuda.synchronize") as synchronize, \
             patch("sisa_training.perf_counter", return_value=42):
            self.assertEqual(synchronized_time(device), 42)
            synchronize.assert_called_once_with(device)


if __name__ == "__main__":
    unittest.main()
