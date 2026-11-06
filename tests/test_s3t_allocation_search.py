"""Development memberships, candidate coverage, and ResNet S3T adapter checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from dataset_splits import load_manifest
from s3t.adapters import S3TModel
from s3t.config import Config
from s3t.data import development_split, load_saved_split
from s3t.models import build_backbone
from studies.s3t_allocation_search import run as search


class S3TAllocationSearchTests(unittest.TestCase):
    def test_candidate_plan_is_reproducible_and_covers_feasible_levels(self):
        for name in ("vit_cifar100", "resnet50_cifar10"):
            settings = json.loads((search.STUDY / "configs" / f"{name}.json").read_text())
            rows = search.candidate_plan(settings)
            self.assertEqual(rows, search.candidate_plan(settings))
            self.assertEqual(len(rows), settings["screening_trials"])
            self.assertEqual(len({tuple(row[key] for key in ("shards", "slices", "budget",
                                                            "epochs_per_slice", "learning_rate"))
                                  for row in rows}), len(rows))
            for band in range(len(settings["shard_bands"])):
                self.assertEqual(sum(row["shard_band"] == band for row in rows), 12)
            for row in rows:
                self.assertLessEqual(row["member_count"], settings["max_members"])
                self.assertLessEqual(row["initial_training_work_ratio"],
                                     settings["max_training_work_ratio"])
                self.assertEqual(row["layers_per_slice"] * row["slices"],
                                 search.DEPTH[settings["model"]])

    def test_copied_manifests_select_disjoint_scoring_and_confirmation(self):
        folder = search.STUDY / "configs/manifests"
        for dataset in ("cifar10", "cifar100"):
            parent_path = folder / f"{dataset}-val5000-seed42.json"
            development_path = folder / f"{dataset}-dev5000-fit60-score20-confirm20-seed42.json"
            parent = load_manifest(parent_path)
            model = "resnet50" if dataset == "cifar10" else "vit_b_16"
            scores = []
            for pool in ("scoring", "confirmation"):
                config = Config(dataset=dataset, model=model, pretrained=False,
                                split_manifest=str(parent_path),
                                development_manifest=str(development_path), development_pool=pool)
                effective = development_split(config, parent)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "split.json"
                    path.write_text(json.dumps(effective))
                    self.assertEqual(load_saved_split(path, config), effective)
                self.assertEqual(len(effective["train_indices"]), 3000)
                self.assertEqual(len(effective["val_indices"]), 1000)
                scores.append(set(effective["val_indices"]))
            self.assertFalse(scores[0] & scores[1])

    def test_resnet_adapters_replay_and_keep_batch_norm_buffers_fixed(self):
        torch.set_num_threads(2)
        config = Config(dataset="cifar10", model="resnet50", pretrained=False,
                        shards=2, slices=4, budget=1, layers_per_slice=1,
                        rank=2, alpha=4, epochs_per_slice=1, num_workers=0)
        backbone, spec = build_backbone(config, 10)
        model = S3TModel(backbone, spec["kind"], config)
        self.assertEqual(model.layer_groups, [[3], [2], [1], [0]])
        self.assertGreater(model.adapter_parameter_count(), 0)
        batch_norm = next(module for module in model.modules()
                          if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
        before = batch_norm.running_mean.clone()
        model.begin_stage(0)
        model.train()
        self.assertFalse(batch_norm.training)
        with torch.no_grad():
            self.assertEqual(tuple(model(torch.randn(1, 3, 64, 64)).shape), (1, 10))
        self.assertTrue(torch.equal(before, batch_norm.running_mean))
        model.finish_stage()
        state = model.export_state()
        restored_backbone, _ = build_backbone(config, 10)
        restored = S3TModel(restored_backbone, spec["kind"], config)
        restored.load_exported_state(state)
        restored.activate_prefix(1)
        self.assertEqual(restored.active_prefix, 1)

    def test_confirmation_report_selects_one_per_shard_band(self):
        settings = json.loads((search.STUDY / "configs/vit_cifar100.json").read_text())
        candidates = search.candidate_plan(settings)
        plan = {"settings": settings, "candidates": candidates}
        scoring = [dict(candidate=row, accuracy=0.6 + index / 1000,
                        plan_sha256="test") for index, row in enumerate(candidates)]
        frozen = search.freeze_finalists(plan, scoring)
        selected_ids = {identity for band in frozen["bands"] for identity in band["candidate_ids"]}

        def fake_result(_run, candidate, pool, seed):
            if pool == "scoring":
                return next(row for row in scoring if row["candidate"]["id"] == candidate["id"])
            if candidate["id"] in selected_ids:
                return dict(candidate=candidate, accuracy=0.7 + seed / 1000,
                            pool=pool, seed=seed)
            return None

        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            search.signed_write(run / "finalists.json", frozen)
            with patch.object(search, "result", side_effect=fake_result):
                search.Runner(run, plan, ["cpu"]).reports()
            with (run / "selected_by_band.csv").open(newline="") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(selected), 3)
            self.assertEqual({int(row["shard_band"]) for row in selected}, {0, 1, 2})

    def test_interrupted_preparation_is_preserved_before_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "trial"
            (run / "inputs").mkdir(parents=True)
            (run / "inputs/parent.json").write_text("partial")
            archive = search.archive_incomplete_preparation(run)
            self.assertFalse(run.exists())
            self.assertEqual((archive / "inputs/parent.json").read_text(), "partial")
            run.mkdir()
            (run / "search").mkdir()
            with self.assertRaisesRegex(ValueError, "contains other files"):
                search.archive_incomplete_preparation(run)
            self.assertTrue((run / "search").exists())


if __name__ == "__main__":
    unittest.main()
