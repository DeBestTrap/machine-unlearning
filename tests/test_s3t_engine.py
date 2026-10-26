from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from deletion_requests import generate_workload
from s3t.artifacts import atomic_json, file_hash, lock
from s3t.config import Config
from s3t.engine import Run


torch.set_num_threads(1)


def config(**kwargs):
    values = dict(dataset="synthetic", model="tiny_vit", pretrained=False, shards=2, slices=2, budget=2,
                  rank=4, alpha=8, lora_dropout=0.1, epochs_per_slice=2, batch_size=8, eval_batch_size=16,
                  num_workers=0, learning_rate=0.01, scheduler="linear", warmup_ratio=0.2)
    values.update(kwargs)
    return Config.from_dict(values)


class EngineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_epoch_resume_matches_uninterrupted_training_bit_for_bit(self):
        cfg = config(shards=1, budget=1)
        complete = Run.create(self.root / "complete", cfg)
        complete.train("cpu")
        first = complete.checkpoint(0, 0, 0)
        for interrupted_at in (1, 2, 3):
            with self.subTest(interrupted_at=interrupted_at):
                partial = Run.create(self.root / f"partial-{interrupted_at}", cfg)
                partial.train("cpu", max_epochs=interrupted_at)
                self.assertEqual(partial.status()["status"], "training")
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    partial.delete([partial.metadata["source_ids"][0]])
                resumed = Run(partial.directory)
                resumed.train("cpu")
                second = resumed.checkpoint(0, 0, 0)
                for name, tensors in first["model"]["adapters"].items():
                    for key, value in tensors.items():
                        self.assertTrue(torch.equal(value, second["model"]["adapters"][name][key]), name)
                for head1, head2 in zip(first["model"]["head_history"], second["model"]["head_history"]):
                    for key, value in head1.items():
                        self.assertTrue(torch.equal(value, head2[key]), key)
                self.assertEqual(first["validation_scores"], second["validation_scores"])
                checkpoint_hash = file_hash(resumed.member_path(0, 0, 0))
                resumed.train("cpu")
                self.assertEqual(checkpoint_hash, file_hash(resumed.member_path(0, 0, 0)))

    def test_saved_replay_resumes_and_deletes_original_source_ids(self):
        run = Run.create(self.root / "run", config(epochs_per_slice=1))
        run.train("cpu")
        workload = generate_workload(run.split, seed=123, steps=6)
        requests = self.root / "requests.json"
        atomic_json(requests, workload)
        run.replay(requests, "cpu", steps=2, on_exhaustion="continue")
        resumed = Run(run.directory)
        resumed.replay(requests, "cpu", steps=6, on_exhaustion="continue")
        state = resumed.state()
        self.assertEqual(state["deleted_source_ids"], sorted(sum(workload["batches"], [])))
        self.assertEqual([event["source_ids"] for event in state["history"]], workload["batches"])
        partition = resumed.partition(0)
        for event in state["history"]:
            for member in event["selected"]:
                if member:
                    active_ids = partition.training_ids(member["shard"], member["sequence"], member["prefix"])
                    self.assertFalse(set(active_ids) & set(event["cumulative_source_ids"]))
        cached = resumed.evaluate("cpu")
        uncached = resumed.evaluate("cpu", cache=False)
        self.assertEqual(cached["accuracy"], uncached["accuracy"])
        self.assertEqual(resumed.evaluate("cpu", step=0)["deleted_examples"], 0)
        self.assertTrue(Path(resumed.export_csv()).exists())
        self.assertEqual(resumed.audit()["steps_checked"], 6)
        other_requests = self.root / "other-requests.json"
        atomic_json(other_requests, generate_workload(run.split, seed=456, steps=6))
        with self.assertRaisesRegex(ValueError, "differs|different"):
            resumed.replay(other_requests, "cpu", on_exhaustion="continue")
        with self.assertRaisesRegex(ValueError, "already deleted"):
            resumed.delete(workload["batches"][0])

    def test_exhaustion_and_retraining_only_use_retained_examples(self):
        run = Run.create(self.root / "run", config(epochs_per_slice=1))
        run.train("cpu")
        partition = run.partition(0)
        deleted = [part[0] for shard in partition.shards for part in shard]
        run.delete(deleted)
        result = run.evaluate("cpu")
        self.assertEqual(result["status"], "requires_retraining")
        self.assertIsNone(result["accuracy"])
        old_backbone = file_hash(run.directory / "backbone.pt")
        pending = run.retrain("cpu", max_epochs=1)
        self.assertEqual(pending["status"], "retraining")
        self.assertEqual(run.state()["generation"], 0)
        result = Run(run.directory).retrain("cpu")
        self.assertEqual(result["generation"], 1)
        self.assertEqual(run.state()["generation"], 1)
        self.assertFalse(set(run.partition(1).locations) & set(deleted))
        self.assertEqual(set(run.partition(1).locations), set(run.metadata["source_ids"]) - set(deleted))
        self.assertEqual(run.evaluate("cpu")["active_shards"], 2)
        self.assertEqual(file_hash(run.directory / "backbone.pt"), old_backbone)
        self.assertEqual(run.retrain("cpu"), result)
        self.assertEqual(run.audit()["deleted_examples"], len(deleted))

    def test_replay_rejects_caps_wrong_workloads_and_reinitialization(self):
        run = Run.create(self.root / "run", config(max_train_examples=32))
        workload = generate_workload(run.split, seed=123, steps=3)
        requests = self.root / "requests.json"
        atomic_json(requests, workload)
        with self.assertRaisesRegex(ValueError, "full training pool"):
            run.replay(requests, "cpu")
        with self.assertRaisesRegex(ValueError, "different configuration"):
            Run.create(run.directory, replace(run.config, rank=8))
        value = json.loads((run.directory / "config.json").read_text())
        value["seed"] += 1
        atomic_json(run.directory / "config.json", value)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            Run(run.directory)

    def test_independent_member_order_and_validation_selection(self):
        cfg = config(epochs_per_slice=1, selection="validation")
        run = Run.create(self.root / "run", cfg)
        for shard in (1, 0):
            for sequence in (1, 0):
                run.train("cpu", shard=shard, sequence=sequence)
        result = run.evaluate("cpu", "val")
        self.assertEqual(result["active_shards"], 2)
        for member in result["selected"]:
            scores = [run.checkpoint(0, member["shard"], sequence)["validation_scores"][-1]
                      for sequence in range(cfg.budget)]
            self.assertEqual(scores[member["sequence"]], max(scores))
        ordered = Run.create(self.root / "ordered", cfg)
        ordered.train("cpu")
        for shard in range(cfg.shards):
            for sequence in range(cfg.budget):
                a = run.checkpoint(0, shard, sequence)["model"]
                b = ordered.checkpoint(0, shard, sequence)["model"]
                for name in a["adapters"]:
                    for key in a["adapters"][name]:
                        self.assertTrue(torch.equal(a["adapters"][name][key], b["adapters"][name][key]))

    def test_automatic_retraining_and_audit_detect_trace_corruption(self):
        run = Run.create(self.root / "run", config(shards=1, epochs_per_slice=1))
        run.train("cpu")
        requests = self.root / "requests.json"
        atomic_json(requests, generate_workload(run.split, seed=123, steps=10))
        run.replay(requests, "cpu", on_exhaustion="retrain")
        self.assertGreater(run.state()["generation"], 0)
        self.assertEqual(run.audit()["steps_checked"], 10)
        state = run.state()
        state["history"][0]["prefixes"][0][0] = 999
        atomic_json(run.directory / "state.json", state)
        with self.assertRaisesRegex(ValueError, "prefix masks"):
            run.audit()

    def test_pending_retraining_blocks_new_deletions(self):
        run = Run.create(self.root / "run", config(shards=1, budget=1, epochs_per_slice=1))
        run.train("cpu")
        source_ids = run.metadata["source_ids"]
        run.delete([source_ids[0]])
        run.retrain("cpu", max_epochs=1)
        with self.assertRaisesRegex(ValueError, "pending"):
            run.delete([source_ids[1]])

    def test_lifecycle_and_member_locks_reject_conflicting_writers(self):
        run = Run.create(self.root / "run", config(shards=1, budget=1))
        with lock(run.directory / ".lifecycle.lock"):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                run.train("cpu")
        with lock(run.member_path(0, 0, 0).with_suffix(".lock")):
            with self.assertRaisesRegex(RuntimeError, "Another process"):
                run.train("cpu")


if __name__ == "__main__":
    unittest.main()
