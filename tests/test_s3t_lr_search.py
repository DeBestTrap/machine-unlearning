import json
from pathlib import Path
import tempfile
import unittest

from s3t.config import Config
from s3t.lr_search import make_plan, run_search


class LearningRateSearchTests(unittest.TestCase):
    def test_screening_plan_uses_bms_first_slice_and_intended_training_regime(self):
        plan = make_plan(["cifar10", "cifar100", "sst2"], "cuda:0")
        self.assertEqual(len(plan["trials"]), 9)
        for trial in plan["trials"]:
            config = Config.from_dict(trial["config"])
            self.assertEqual(config.head_policy, "first_slice")
            self.assertEqual(config.sequence_strategy, "bms")
            self.assertEqual(config.shards, 5)
            self.assertEqual(config.slices, 6)
            self.assertEqual(config.budget, 1)
            self.assertIsNone(config.max_train_examples)
            self.assertIsNone(config.max_eval_examples)
            self.assertEqual(config.epochs_per_slice, 15 if config.dataset.startswith("cifar") else 10)

    def test_subprocess_queue_produces_validation_results_and_skips_completed_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config(dataset="synthetic", model="tiny_vit", pretrained=False,
                            shards=1, slices=2, budget=1, rank=4, alpha=8, sequence_strategy="bms",
                            deletion_probabilities=[0.5, 0.5], epochs_per_slice=1,
                            num_workers=0, batch_size=16, eval_batch_size=16).validate()
            plan = {"schema_version": 1, "device": "cpu", "shard": 0, "sequence": 0,
                    "selection_metric": "final_prefix_validation_accuracy",
                    "trials": [{"name": "smoke", "dataset": "synthetic",
                                "learning_rate": config.learning_rate, "config": config.to_dict()}]}
            result = run_search(root, plan)
            self.assertEqual(result["status"], "completed")
            row = result["trials"][0]
            self.assertEqual(row["completed_epochs"], 2)
            self.assertIsNotNone(row["validation_accuracy"])
            self.assertGreater(row["wall_seconds"], 0)
            saved = root / "smoke/pilot_result.json"
            before = saved.read_bytes()
            self.assertFalse((root / "smoke/evaluations").exists())
            self.assertEqual(json.loads(before)["head_policy"], "first_slice")
            resumed = run_search(root, plan)
            self.assertEqual(resumed["status"], "completed")
            self.assertEqual(saved.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
