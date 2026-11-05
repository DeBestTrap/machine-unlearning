"""Small-model checks of development isolation, selection and model-free resume."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from reporting.core import digest, write_json
from studies.baseline_search import run as search


class TinyLoader:
    classes = 2
    fit = np.arange(3)
    selected = np.arange(2)

    def load(self, indices, category="train"):
        if category not in ("train", "train_eval", "val"):
            raise AssertionError("Unexpected data access")
        indices = np.asarray(indices)
        return np.stack((indices + 1, indices % 2), axis=1).astype(np.float32), indices % 2


def tiny_model(config, device):
    return torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.Dropout(0.2),
                               torch.nn.Linear(4, 2)).to(device)


class BaselineSearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        self.run = self.root / "studies/fixture/runs/test"
        (self.run / "inputs").mkdir(parents=True)
        self.parent = dict(dataset_id="cifar10", source_size=50, source_sha256="fixture", split_seed=42,
                           stratified=True, train_indices=list(range(45)), val_indices=list(range(45, 50)))
        self.parent["manifest_sha256"] = digest(self.parent)
        self.development = dict(dataset_id="cifar10", source_size=50, source_sha256="fixture", split_seed=42,
            stratified=True, parent_manifest_sha256=self.parent["manifest_sha256"],
            train_indices=[45, 46, 47], val_indices=[48], confirmation_indices=[49], confirmation_excluded_indices=[])
        self.development["manifest_sha256"] = digest(self.development)
        write_json(self.run / "inputs/parent.json", self.parent)
        write_json(self.run / "inputs/development.json", self.development)
        self.settings = json.loads((search.STUDY / "configs/resnet50_cifar10.json").read_text())
        self.candidate = dict(id="candidate-000", shards=2, slices=1, epochs=1, epochs_per_slice=1,
                              learning_rate=0.01, normalized_expected_cost=0.5, cost_stratum=0.5)
        self.plan = dict(settings={**self.settings, "cost_budgets": [0.5]}, candidates=[self.candidate],
            parent_manifest_sha256=self.parent["manifest_sha256"],
            development_manifest_sha256=self.development["manifest_sha256"])
        search.signed_write(self.run / "plan.json", self.plan)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)

    def test_search_is_reproducible_and_all_candidates_have_valid_schedules_and_costs(self):
        candidates = search.candidate_plan(self.settings, 3000)
        self.assertEqual(candidates, search.candidate_plan(self.settings, 3000))
        self.assertEqual(len(candidates), 100)
        self.assertEqual([sum(row["cost_stratum"] == ceiling for row in candidates)
                          for ceiling in self.settings["cost_budgets"]], [34, 33, 33])
        self.assertEqual(sum(row["shards"] for row in candidates), 1880)
        self.assertEqual(len({(row["shards"], row["slices"], row["epochs"], row["learning_rate"])
                              for row in candidates}), 100)
        for candidate in candidates:
            self.assertEqual(candidate["epochs_per_slice"] * (candidate["slices"] + 1), 2 * candidate["epochs"])
            self.assertLessEqual(candidate["normalized_expected_cost"], candidate["cost_stratum"])
            self.assertTrue(1e-5 <= candidate["learning_rate"] <= 1e-3)

    def test_memberships_reject_overlap_and_confirmation_contamination_even_when_resigned(self):
        self.assertEqual(search.validate_memberships(self.parent, self.development)["fit"], [45, 46, 47])
        for changed in (dict(self.development, val_indices=[47]),
                        dict(self.development, confirmation_excluded_indices=[49]),
                        dict(self.development, parent_manifest_sha256="different")):
            changed.pop("manifest_sha256")
            changed["manifest_sha256"] = digest(changed)
            with self.assertRaises(ValueError):
                search.validate_memberships(self.parent, changed)

    def test_finalist_selection_obeys_budgets_and_is_method_specific(self):
        rows = [dict(candidate=dict(id="cheap", normalized_expected_cost=0.1, learning_rate=0.001),
                     sisa_accuracy=0.7, protosisa_accuracy=0.9),
                dict(candidate=dict(id="costly", normalized_expected_cost=0.5, learning_rate=0.001),
                     sisa_accuracy=0.8, protosisa_accuracy=0.8)]
        selection = search.select_finalists(rows, [0.1, 0.5], 1)
        self.assertEqual([item["candidate_ids"] for item in selection["selections"]],
                         [["cheap"], ["costly"], ["cheap"], ["cheap"]])

    def test_features_only_use_fitting_examples_for_prototypes(self):
        loader = TinyLoader()
        with patch.object(search, "DevelopmentLoader", return_value=loader), \
             patch("protosisa.build_encoder", return_value=torch.nn.Identity()):
            search.features(self.run, "score", "cpu")
        marker = search.feature_marker(self.run, "score")
        self.assertEqual(marker["fitting_examples"], 3)
        self.assertEqual(marker["prototype_counts"], [2, 1])
        self.assertEqual(marker["evaluation_examples"], 2)

    def test_shared_training_reports_both_methods_and_resumes_after_model_cleanup(self):
        loader = TinyLoader()
        with patch.object(search, "DevelopmentLoader", return_value=loader), \
             patch("protosisa.build_encoder", return_value=torch.nn.Identity()):
            search.features(self.run, "score", "cpu")
        search.run_trial(self.run, self.candidate, "score", 1, "cpu", loader=loader, model_factory=tiny_model)
        directory = search.trial_directory(self.run, self.candidate, "score", 1)
        first = search.result(self.run, self.candidate, "score", 1)
        self.assertEqual(first["initial_example_presentations"], 3)
        self.assertFalse(list((directory / "cache").glob("*.pt")))
        self.assertTrue((directory / "sisa/run.json").exists())
        self.assertTrue((directory / "protosisa/run.json").exists())
        arrays = np.load(directory / "predictions.npz")
        weight = first["prototype_weight"]
        prototype = np.load(search.feature_paths(self.run, "score")[0])["prototype"]
        np.testing.assert_allclose(arrays["protosisa"], (1 - weight) * arrays["sisa"] + weight * prototype)
        # Simulate an interruption after all members were scored but before the
        # final result was published: their durable arrays replace checkpoints.
        (directory / "result.json").unlink()
        with patch("protosisa.train_member", side_effect=AssertionError("Must reuse completed members")):
            search.run_trial(self.run, self.candidate, "score", 1, "cpu", loader=loader, model_factory=tiny_model)
        frozen = search.freeze_finalists(self.run)
        self.assertEqual(frozen["candidate_ids"], ["candidate-000"])
        self.assertEqual(search.summarize(self.run)["search_completed"], 1)
        with patch.object(search, "DevelopmentLoader", return_value=loader), \
             patch("protosisa.build_encoder", return_value=torch.nn.Identity()):
            search.features(self.run, "confirmation", "cpu")
        for seed in (1, 2, 3):
            search.run_trial(self.run, self.candidate, "confirmation", seed, "cpu",
                             loader=loader, model_factory=tiny_model)
        summary = search.summarize(self.run)
        self.assertEqual(summary["confirmation_completed"], 3)
        self.assertEqual(len(summary["selected"]), 2)
        self.assertTrue(all(row["seeds"] == 3 for row in summary["selected"]))
        self.assertEqual(search.signed_read(self.run / "finalists.json"), frozen)

    def test_confirmation_rejects_unfrozen_candidates(self):
        with self.assertRaisesRegex(ValueError, "frozen finalist"):
            search.run_trial(self.run, self.candidate, "confirmation", 1, "cpu", loader=TinyLoader(), model_factory=tiny_model)

    def test_resnet_task_layers_and_feature_encoder(self):
        from architectures import resnet50 as architecture
        from torchvision.models import resnet50
        with patch.object(architecture, "resnet50", side_effect=lambda **kwargs: resnet50(weights=None)):
            model = architecture.Model([32, 32], 10)
        self.assertFalse(model.model.conv1.weight.requires_grad)
        self.assertFalse(model.model.bn1.weight.requires_grad)
        self.assertTrue(all(parameter.requires_grad for parameter in model.model.layer1.parameters()))
        self.assertTrue(model.model.fc.weight.requires_grad)
        self.assertEqual(model.model.fc.out_features, 10)
        with patch("torchvision.models.resnet50", side_effect=lambda **kwargs: resnet50(weights=None)):
            import protosisa
            encoder = protosisa.build_encoder("resnet50", torch.device("cpu"))
        self.assertIsInstance(encoder.fc, torch.nn.Identity)
        self.assertFalse(any(parameter.requires_grad for parameter in encoder.parameters()))
        self.assertFalse(encoder.training)

    def test_vit_scope_matches_the_native_last_four_layers_and_xavier_head(self):
        class Wrapped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.stem = torch.nn.Linear(4,4)
                self.model.encoder = torch.nn.Module()
                self.model.encoder.ln = torch.nn.LayerNorm(4)
                self.model.encoder.layers = torch.nn.ModuleDict({f"encoder_layer_{i}":torch.nn.Linear(4,4) for i in range(12)})
                self.model.heads = torch.nn.Module()
                self.model.heads.head = torch.nn.Linear(4,100)
        settings=json.loads((search.STUDY / "configs/vit_b16_cifar100.json").read_text())
        model=Wrapped()
        original=model.model.encoder.layers.encoder_layer_0.weight.detach().clone()
        with patch("protosisa.build_model",return_value=model):
            configured=search.build_task_model(dict(model="vit",model_args=settings["model_args"]),torch.device("cpu"))
        self.assertIs(configured,model)
        scope=settings["model_args"]["layers_to_finetune"]
        for name,parameter in model.model.named_parameters():
            self.assertEqual(parameter.requires_grad,name.startswith(tuple(scope)))
        torch.testing.assert_close(model.model.encoder.layers.encoder_layer_0.weight,original)
        torch.testing.assert_close(model.model.heads.head.bias,torch.zeros(100))
        self.assertFalse(model.model.stem.weight.requires_grad)
        self.assertTrue(model.model.encoder.layers.encoder_layer_8.weight.requires_grad)

    def test_legacy_model_factory_keeps_its_existing_scope(self):
        model=torch.nn.Module();model.model=torch.nn.Linear(4,2)
        model.model.bias.requires_grad_(False)
        with patch("protosisa.build_model",return_value=model):
            returned=search.build_task_model(dict(model="vit"),torch.device("cpu"))
        self.assertTrue(returned.model.weight.requires_grad)
        self.assertFalse(returned.model.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
