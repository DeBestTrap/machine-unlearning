"""CPU checks of ProtoSISA arithmetic, deletion replay, and prefix recovery."""
import argparse
from hashlib import sha256
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from container_storage import clear_models, initialize_container
from dataset_splits import _json
from deletion_requests import generate_workload, local_batches
import protosisa as proto
from reporting import load_bundle
from reporting.reports import refresh


class Loader:
    def __init__(self):
        self.calls = []

    def load(self, indices, category="train"):
        indices = np.asarray(indices, dtype=np.int64)
        self.calls.append((category, indices.tolist()))
        inputs = np.stack((1 + indices / 10, (indices % 3) - 1), axis=1).astype(np.float32)
        return inputs, indices % 3


def model_factory(config, device):
    return nn.Sequential(nn.Linear(2, 5), nn.Dropout(0.3), nn.Tanh(), nn.Linear(5, 3)).to(device)


def encoder_factory(name, device):
    # The test encoder has a parameter to verify freezing as well as eval mode.
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Dropout(0.9)).to(device)
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2, device=device))
    return model


class ProtoSISATests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, original)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.device = torch.device("cpu")
        self.config = dict(model="vit", dataset_metadata=dict(nb_classes=3, input_shape=[2]),
            seed=5, slices=3, epochs=4, optimizer="adamw", learning_rate=0.03,
            batch_size=2, dropout_rate=0.3, training_sha256="fixture")

    def train(self, directory, deleted, step=0, factory=model_factory):
        return proto.train_member(directory, self.config, 0, np.arange(7),
                                  deleted, step, Loader(), self.device, factory)

    def test_half_mixture_is_of_logits_and_has_correct_endpoints(self):
        ensemble = torch.tensor([[4., -2., 0.]])
        prototype = torch.tensor([[-1., 1., 0.]])
        torch.testing.assert_close(proto.mix_logits(ensemble, prototype), torch.tensor([[1.5, -0.5, 0.]]))
        torch.testing.assert_close(proto.mix_logits(ensemble, prototype, 0), ensemble)
        torch.testing.assert_close(proto.mix_logits(ensemble, prototype, 1), prototype)
        for weight in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                proto.mix_logits(ensemble, prototype, weight)

    def test_safe_weighting_uses_mean_retained_size_without_calibration(self):
        config = dict(prototype_weighting="safe", prototype_weight=None)
        initial = proto.mixture_parameters(config, 4750, 10)
        self.assertAlmostEqual(initial["prototype_weight"], math.exp(-4.75))
        self.assertNotIn("prototype_temperature", initial)
        self.assertEqual(initial["mean_shard_size"], 475)
        retained = proto.mixture_parameters(config, 4749, 10)
        self.assertAlmostEqual(retained["prototype_weight"], math.exp(-4.749))
        self.assertGreater(retained["prototype_weight"], initial["prototype_weight"])
        self.assertAlmostEqual(proto.mixture_parameters(config, 4751, 10)["prototype_weight"], math.exp(-4.751))
        for invalid in (dict(config, prototype_weight=0.5), dict(config, prototype_temperature=0.05),
                        dict(config, prototype_weighting="unknown")):
            with self.assertRaises(ValueError):
                proto.mixture_parameters(invalid, 4750, 10)
        for count, shards in ((0, 10), (10, 0), (-1, 1), (4.5, 2)):
            with self.assertRaises(ValueError):
                proto.mixture_parameters(config, count, shards)

    def test_safe_run_records_size_dependent_weights_and_exact_scores(self):
        args = self.fixture_args()
        args.prototype_weighting = "safe"
        args.prototype_weight = None
        directory = proto.run(args, model_factory=model_factory, encoder_factory=encoder_factory)
        run, steps, evaluations = load_bundle(directory)
        self.assertEqual(run["config"]["prototype_weighting"], "safe")
        self.assertIsNone(run["config"]["prototype_weight"])
        for step, evaluation in zip(steps, evaluations):
            expected_weight = math.exp(-(11 - step["step"]) / 2 / 100)
            self.assertAlmostEqual(step["method_details"]["prototype_weight"], expected_weight)
            self.assertNotIn("prototype_temperature", evaluation["method_details"])
            self.assertEqual(evaluation["method_details"]["prototype_weighting"], "safe")
            components = np.load(directory / evaluation["method_details"]["component_scores"])
            scores = np.load(directory / evaluation["prediction_artifact"])["scores"]
            np.testing.assert_allclose(scores,
                (1 - expected_weight) * components["ensemble"] + expected_weight * components["prototype"])

    def test_safe_cli_default_and_invalid_calibration_are_checked_before_training(self):
        with patch("protosisa.run") as run:
            proto.main(["run", "--model", "vit", "--dataset", "unused", "--container", "unused"])
            args = run.call_args.args[0]
        self.assertEqual(args.prototype_weighting, "safe")
        self.assertIsNone(args.prototype_weight)
        self.assertFalse(hasattr(args, "prototype_temperature"))
        args = self.fixture_args()
        args.prototype_weighting = "safe"
        with self.assertRaisesRegex(ValueError, "SAFE weighting computes"):
            proto.prepare(args, self.device)
        self.assertFalse(Path("containers/end_to_end").exists())

    def test_temperature_option_is_removed_and_historical_scaling_is_not_silently_reused(self):
        with patch("protosisa.run") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                proto.main(["run", "--model", "vit", "--dataset", "unused", "--container", "unused",
                            "--prototype-temperature", "0.05"])
            run.assert_not_called()
        with self.assertRaisesRegex(ValueError, "historical"):
            proto.mixture_parameters(dict(prototype_weighting="fixed", prototype_weight=0.5,
                                          prototype_temperature=0.05), 4750, 10)

    def test_prototypes_are_global_means_of_unit_features(self):
        loader = Loader()
        encoder = encoder_factory("vit", self.device)
        left = proto.prototype_statistics(encoder, loader, np.arange(2), 4, 2, self.device)
        right = proto.prototype_statistics(encoder, loader, np.arange(2, 7), 4, 2, self.device)
        combined = proto.merge_statistics([left, right])
        direct = proto.merge_statistics([proto.prototype_statistics(encoder, loader, np.arange(7), 4, 3, self.device)])
        torch.testing.assert_close(combined["vectors"], direct["vectors"])
        self.assertEqual(combined["counts"].tolist(), [3, 2, 2, 0])
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in encoder.parameters()))
        self.assertTrue(all(category == "train_eval" for category, _ in loader.calls))
        x, y = loader.load(np.arange(7))
        normalized = torch.nn.functional.normalize(torch.from_numpy(x), dim=1).double()
        for cls in range(3):
            torch.testing.assert_close(left["sums"][cls] + right["sums"][cls], normalized[y == cls].sum(0))
        logits = proto.prototype_logits(torch.from_numpy(x), combined)
        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(logits[:, 3], torch.full((7,), -1.))

    def test_production_feature_encoders_use_pretrained_backbones_without_task_heads(self):
        from torchvision.models import ViT_B_16_Weights
        vit = nn.Sequential(nn.Linear(2, 3))
        vit.heads = nn.Linear(3, 9)
        with patch("torchvision.models.vit_b_16", return_value=vit) as constructor:
            encoder = proto.build_encoder("vit", self.device)
        constructor.assert_called_once_with(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.assertIsInstance(encoder.heads, nn.Identity)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

        from transformers import RobertaConfig, RobertaModel
        backbone = RobertaModel(RobertaConfig(vocab_size=12, hidden_size=8,
            num_hidden_layers=1, num_attention_heads=2, intermediate_size=16), add_pooling_layer=False)
        with patch("transformers.RobertaModel.from_pretrained", return_value=backbone) as constructor:
            encoder = proto.build_encoder("roberta", self.device)
        constructor.assert_called_once_with("roberta-base", add_pooling_layer=False)
        inputs = torch.tensor([[0, 5, 2, 1], [0, 6, 2, 1]])
        with torch.inference_mode():
            expected = backbone(input_ids=inputs, attention_mask=(inputs != 1).long()).last_hidden_state[:, 0]
            torch.testing.assert_close(encoder(inputs), expected)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder.parameters()))

    def test_cifar_train_eval_is_deterministic_and_uses_training_indices(self):
        from PIL import Image
        class FakeCIFAR:
            base_folder = "unused-fixture"
            def __init__(self, root, train, transform, download):
                self.data = np.stack([np.full((32, 32, 3), i * 20, dtype=np.uint8) for i in range(8)])
                self.targets = list(range(8))
                self.transform = transform

            def __len__(self):
                return 8

            def __getitem__(self, index):
                return self.transform(Image.fromarray(self.data[index])), self.targets[index]

        split = dict(train_indices=[4, 1, 6, 2], val_indices=[0, 3, 5, 7])
        for name in ("cifar10", "cifar100"):
            spec = importlib.util.spec_from_file_location(f"_proto_test_{name}",
                proto.ROOT / "datasets" / name / "dataloader.py")
            loader = importlib.util.module_from_spec(spec)
            with patch("torchvision.datasets.CIFAR10", FakeCIFAR), \
                 patch("torchvision.datasets.CIFAR100", FakeCIFAR), \
                 patch("split_support.configured_manifest", return_value=split):
                spec.loader.exec_module(loader)
            actual, labels = loader.load([0, 1], category="train_eval")
            repeated, repeated_labels = loader.load([0, 1], category="train_eval")
            np.testing.assert_array_equal(actual, repeated)
            np.testing.assert_array_equal(labels, [4, 1])
            np.testing.assert_array_equal(labels, repeated_labels)
            self.assertEqual(actual.shape, (2, 3, 224, 224))
            np.testing.assert_array_equal(loader.load([0, 1], category="val")[1], [0, 3])

    def test_deletion_reuses_prefix_and_matches_full_retained_retraining(self):
        directory = initialize_container("updated")
        original = self.train(directory, [])
        original_paths = proto.checkpoint_paths(directory, self.config, 0, np.arange(7), [])
        updated = self.train(directory, [6], step=1)
        self.assertNotEqual(original, updated)  # The remainder is part of the final checkpoint identity.
        record = json.loads((directory / "times/shard-0:1.json").read_text())
        self.assertEqual([item["reused"] for item in record["slices"]], [True, True, False])
        fresh = self.train(initialize_container("fresh"), [6])
        saved = torch.load(updated, weights_only=True)
        reference = torch.load(fresh, weights_only=True)
        for name in saved["model"]:
            torch.testing.assert_close(saved["model"][name], reference["model"][name], rtol=0, atol=0)
        for state, expected in zip(saved["optimizer"]["state"].values(), reference["optimizer"]["state"].values()):
            for name in state:
                torch.testing.assert_close(state[name], expected[name], rtol=0, atol=0)
        self.assertTrue(all(path.exists() for path, _, _ in original_paths))
        before = (directory / "times/shard-0:1.json").read_bytes()
        self.train(directory, [6], step=1)
        self.assertEqual(before, (directory / "times/shard-0:1.json").read_bytes())

    def test_interrupted_slice_restarts_with_same_model_and_rng(self):
        directory = initialize_container("interrupted")
        calls = 0
        def factory(config, device):
            model = model_factory(config, device)
            forward = model.forward
            def fail_once(inputs):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("simulated failure")
                return forward(inputs)
            model.forward = fail_once
            return model
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            self.train(directory, [], factory=factory)
        self.assertEqual(json.loads((directory / "times/shard-0:0.json").read_text())["status"], "interrupted")
        resumed = self.train(directory, [])
        fresh = self.train(initialize_container("fresh"), [])
        actual = torch.load(resumed, weights_only=True)["model"]
        expected = torch.load(fresh, weights_only=True)["model"]
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_prototype_update_only_encodes_changed_membership(self):
        directory = initialize_container("prototypes")
        shards = [np.arange(4), np.arange(4, 7)]
        loader = Loader()
        encoder = encoder_factory("vit", self.device)
        proto.update_prototypes(directory, self.config, shards, [[], []], 0, loader, encoder, self.device)
        loader.calls.clear()
        updated = proto.update_prototypes(directory, self.config, shards, [[0], []], 1, loader, encoder, self.device)
        self.assertEqual([index for _, indices in loader.calls for index in indices], [1, 2, 3])
        direct = proto.merge_statistics([proto.prototype_statistics(encoder, Loader(), np.arange(1, 7), 3, 2, self.device)])
        torch.testing.assert_close(updated["vectors"], direct["vectors"])
        torch.testing.assert_close(updated["counts"], direct["counts"])
        empty = proto.prototype_statistics(encoder, Loader(), [], 3, 2, self.device)
        torch.testing.assert_close(proto.merge_statistics([empty, direct_to_statistics(direct)])["vectors"], direct["vectors"])
        with self.assertRaisesRegex(ValueError, "retained"):
            proto.merge_statistics([empty])

    def fixture_args(self):
        source_size = 14
        val = [2, 7, 13]
        train = [i for i in range(source_size) if i not in val]
        split = dict(schema_version=1, dataset_id="fixture", source_split="train", source_size=source_size,
                     source_sha256="fixture-source", split_seed=42, stratified=False,
                     train_indices=train, val_indices=val)
        split["manifest_sha256"] = sha256(_json(split).encode()).hexdigest()
        (self.root / "split.json").write_text(json.dumps(split))
        metadata = dict(nb_train=len(train), nb_val=len(val), nb_test=5, nb_classes=3,
                        input_shape=[2], dataloader="dataloader", split_manifest="split.json")
        dataset = self.root / "datasetfile"
        dataset.write_text(json.dumps(metadata))
        (self.root / "dataloader.py").write_text(
            "import numpy as np\n"
            f"train = np.array({train})\nval = np.array({val})\n"
            "def load(indices, category='train'):\n"
            "    indices = np.asarray(indices, dtype=np.int64)\n"
            "    source = train[indices] if category in ('train', 'train_eval') else val[indices] if category == 'val' else indices\n"
            "    inputs = np.stack((1 + source / 10, source % 3 - 1), axis=1).astype(np.float32)\n"
            "    return inputs, source % 3\n")
        workload = generate_workload(split, steps=2, batch_size=1)
        requests = self.root / "requests.json"
        requests.write_text(json.dumps(workload))
        self.split, self.workload = split, workload
        return argparse.Namespace(model="vit", dataset=str(dataset), container="end_to_end", requests=str(requests),
            shards=2, slices=2, epochs=3, batch_size=2, learning_rate=0.03,
            optimizer="adamw", dropout_rate=0.3, seed=5, prototype_weight=0.5,
            eval_split="test", device="cpu")

    def test_complete_run_shared_ids_reports_resume_and_cleanup(self):
        args = self.fixture_args()
        directory = proto.run(args, model_factory=model_factory, encoder_factory=encoder_factory)
        run, steps, evaluations = load_bundle(directory)
        self.assertEqual(run["method"], "protosisa")
        self.assertEqual(run["split_sha256"], self.split["manifest_sha256"])
        self.assertEqual(run["workload_sha256"], self.workload["workload_sha256"])
        self.assertEqual([step["deleted_total"] for step in steps], [0, 1, 2])
        for step in steps:
            self.assertEqual(sum(step["method_details"]["prototype_counts"]), 11 - step["step"])
            self.assertIsNotNone(step["update_time_sec"])
        for index in (1, 2):
            self.assertEqual(steps[index]["request_sha256"], proto.digest(self.workload["batches"][index - 1]))
        self.assertEqual([value["examples"] for value in evaluations], [5, 5, 5])
        self.assertTrue(all(value["method_details"]["prototype_weight"] == 0.5 for value in evaluations))
        self.assertNotIn("prototype_temperature", run["config"])
        self.assertTrue(all("prototype_temperature" not in value["method_details"] for value in evaluations))
        components = np.load(directory / "outputs/components-step-0-test.npz")
        np.testing.assert_allclose(0.5 * components["ensemble"] + 0.5 * components["prototype"],
            np.load(directory / evaluations[0]["prediction_artifact"])["scores"])
        expected = set(sum(local_batches(self.workload, self.split), []))
        deleted = np.load(directory / "requestfile:2.npy", allow_pickle=True)
        self.assertEqual({int(i) for shard in deleted for i in shard}, expected)
        timers = {path.name: path.read_bytes() for path in (directory / "times").glob("*.json")}
        proto.run(args, model_factory=model_factory, encoder_factory=encoder_factory)
        self.assertEqual(timers, {path.name: path.read_bytes() for path in (directory / "times").glob("*.json")})
        args.prototype_weight = 0.2
        with self.assertRaisesRegex(ValueError, "settings differ"):
            proto.run(args, model_factory=model_factory, encoder_factory=encoder_factory)
        args.prototype_weight = 0.5
        clear_models([args.container])
        refresh([directory], self.root / "report", plots=False)
        self.assertTrue((self.root / "report/summary.csv").exists())
        self.assertEqual(load_bundle(directory)[1], steps)
        with self.assertRaisesRegex(ValueError, "models were removed"):
            proto.run(args, model_factory=model_factory, encoder_factory=encoder_factory)

    def test_membership_tampering_and_invalid_options_are_rejected(self):
        args = self.fixture_args()
        directory, _, _, _ = proto.prepare(args, self.device)
        shards = np.load(directory / "splitfile.npy", allow_pickle=True)
        shards[0][0], shards[1][0] = shards[1][0], shards[0][0]
        np.save(directory / "splitfile.npy", shards, allow_pickle=True)
        with self.assertRaisesRegex(ValueError, "membership or deletion requests changed"):
            proto.prepare(args, self.device)
        args.container = "invalid"
        args.prototype_weight = float("nan")
        with self.assertRaisesRegex(ValueError, "prototype weight"):
            proto.prepare(args, self.device)
        self.assertFalse(Path("containers/invalid").exists())


def direct_to_statistics(prototypes):
    # Only used for checking that merging an empty contribution is neutral.
    return {"sums": prototypes["vectors"].double() * prototypes["counts"][:, None],
            "counts": prototypes["counts"]}


if __name__ == "__main__":
    unittest.main()
