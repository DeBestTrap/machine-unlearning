from copy import deepcopy
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from s3t.adapters import LoRASelfAttention, S3TModel, cpu_state
from s3t.config import Config
from s3t.models import build_backbone


torch.set_num_threads(1)


def tiny_config(model="tiny_vit", **kwargs):
    values = dict(dataset="synthetic", model=model, pretrained=False, shards=1, slices=4, budget=1,
                  rank=4, alpha=8, lora_dropout=0.0, epochs_per_slice=1, num_workers=0,
                  batch_size=8, eval_batch_size=16)
    values.update(kwargs)
    return Config.from_dict(values)


class AdapterTests(unittest.TestCase):
    def test_torchvision_attention_matches_frozen_base_and_has_working_gradients(self):
        torch.manual_seed(5)
        base = nn.MultiheadAttention(16, 4, batch_first=True, dropout=0.0).eval()
        adapted = LoRASelfAttention(deepcopy(base), rank=3, alpha=6, dropout=0.0, scaling="lora", stage=0).eval()
        inputs = torch.randn(2, 7, 16)
        with torch.no_grad():
            expected = base(inputs, inputs, inputs, need_weights=False)[0]
            actual = adapted(inputs, inputs, inputs)[0]
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
        adapted.query_delta.enabled = adapted.value_delta.enabled = True
        torch.testing.assert_close(adapted(inputs, inputs, inputs)[0], expected, atol=2e-7, rtol=2e-6)
        adapted(inputs, inputs, inputs)[0].square().mean().backward()
        self.assertGreater(float(adapted.query_delta.b.grad.abs().sum()), 0)
        self.assertGreater(float(adapted.value_delta.b.grad.abs().sum()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in adapted.base.parameters()))

    def _prefix_restoration(self, model_name, head_policy):
        torch.manual_seed(12)
        config = tiny_config(model_name, head_policy=head_policy)
        raw, spec = build_backbone(config, 3)
        model = S3TModel(raw, spec["kind"], config)
        if model_name == "tiny_vit":
            inputs = torch.randn(8, 3, 16, 16)
        else:
            inputs = torch.randint(4, 64, (8, 12))
            inputs[:, 0] = 0
        labels = torch.arange(8) % 3
        prefixes = []
        frozen = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                  if ".delta." not in name and "_delta." not in name and not name.startswith("backbone." + model.head_path)}
        for stage in range(4):
            earlier = {name: cpu_state(delta) for name, delta in model.deltas() if delta.stage < stage}
            optimizer = torch.optim.AdamW(model.begin_stage(stage), lr=0.02, weight_decay=0.3)
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                F.cross_entropy(model(inputs), labels).backward()
                optimizer.step()
            model.finish_stage()
            with torch.no_grad():
                prefixes.append(model(inputs).clone())
            for name, value in frozen.items():
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], value), name)
            for name, delta in model.deltas():
                if name in earlier:
                    for key, value in earlier[name].items():
                        self.assertTrue(torch.equal(delta.state_dict()[key], value), name)
        final_head = cpu_state(model.head)
        for prefix in range(1, 5):
            model.activate_prefix(prefix)
            with torch.no_grad():
                restored = model(inputs)
            torch.testing.assert_close(restored, prefixes[prefix - 1], rtol=0, atol=0)
        if head_policy == "snapshot":
            self.assertTrue(any(not torch.equal(final_head[key], value) for key, value in model.head_history[0].items()))
        else:
            for key, value in model.head_history[0].items():
                self.assertTrue(torch.equal(final_head[key], value))
        # Reload into a new model and verify the disabled prefix stays identical.
        # deepcopy preserves exactly the frozen backbone; replace the active state.
        restored = deepcopy(model)
        restored.load_exported_state(model.export_state())
        restored.activate_prefix(2)
        torch.testing.assert_close(restored(inputs), prefixes[1], rtol=0, atol=0)

    def test_every_trained_vit_prefix_is_restored_exactly(self):
        self._prefix_restoration("tiny_vit", "snapshot")

    def test_every_trained_roberta_prefix_is_restored_exactly(self):
        self._prefix_restoration("tiny_roberta", "snapshot")

    def test_first_slice_head_remains_frozen(self):
        for model in ("tiny_vit", "tiny_roberta"):
            with self.subTest(model=model):
                self._prefix_restoration(model, "first_slice")

    def test_huggingface_vit_adapter_injection(self):
        from transformers import ViTConfig, ViTForImageClassification
        raw = ViTForImageClassification(ViTConfig(image_size=16, patch_size=4, hidden_size=32,
                                                 intermediate_size=64, num_hidden_layers=4,
                                                 num_attention_heads=2, num_labels=3))
        model = S3TModel(raw, "hf_vit", tiny_config())
        params = model.begin_stage(0)
        loss = F.cross_entropy(model(torch.randn(2, 3, 16, 16)), torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(len(params), 0)
        for _, delta in model.deltas():
            self.assertEqual(delta.b.grad is not None, delta.stage == 0)

    def test_geometry_errors_do_not_silently_drop_stages(self):
        raw, spec = build_backbone(tiny_config(), 3)
        with self.assertRaises(ValueError):
            S3TModel(raw, spec["kind"], tiny_config(slices=5))

    def test_paper_roberta_grouping_leaves_only_bottom_three_blocks_unadapted(self):
        from transformers import RobertaConfig, RobertaForSequenceClassification
        raw = RobertaForSequenceClassification(RobertaConfig(vocab_size=64, hidden_size=32,
                                                             num_hidden_layers=24, num_attention_heads=2,
                                                             intermediate_size=64, num_labels=3))
        model = S3TModel(raw, "roberta", tiny_config("tiny_roberta", slices=7, layers_per_slice=3))
        self.assertEqual(model.layer_groups[0], [23, 22, 21])
        self.assertEqual(model.layer_groups[-1], [5, 4, 3])
        adapted = set(sum(model.layer_groups, []))
        self.assertEqual(adapted, set(range(3, 24)))
        self.assertEqual(len(model.deltas()), 21 * 2)


if __name__ == "__main__":
    unittest.main()
