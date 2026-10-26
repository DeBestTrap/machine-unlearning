"""Unmerged, independently switchable query/value LoRA adapters."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRADelta(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, dropout, scaling, stage):
        super().__init__()
        self.a = nn.Parameter(torch.empty(rank, in_features))
        self.b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / (math.sqrt(rank) if scaling == "rslora" else rank)
        self.dropout = nn.Dropout(dropout)
        self.stage = stage
        self.enabled = False

    def forward(self, inputs):
        return F.linear(F.linear(self.dropout(inputs), self.a), self.b) * self.scale


class LoRALinear(nn.Module):
    def __init__(self, base, **kwargs):
        super().__init__()
        self.base = base
        self.delta = LoRADelta(base.in_features, base.out_features, **kwargs)
        self.base.requires_grad_(False)

    def forward(self, inputs):
        result = self.base(inputs)
        if self.delta.enabled:
            result = result + self.delta(inputs)
        return result


class LoRASelfAttention(nn.Module):
    """torchvision ViT's batch-first self attention, with Q/V LoRA branches.

    Keep the frozen fused QKV matrix unchanged. Explicit SDPA prevents the
    nn.MultiheadAttention inference fast path from bypassing adapter modules.
    This wrapper deliberately supports only the call used by torchvision ViT.
    """
    def __init__(self, base, **kwargs):
        super().__init__()
        if (not base.batch_first or not base._qkv_same_embed_dim or base.bias_k is not None
                or base.bias_v is not None or base.add_zero_attn):
            raise ValueError("Unsupported attention layout for torchvision ViT")
        self.base = base
        self.query_delta = LoRADelta(base.embed_dim, base.embed_dim, **kwargs)
        self.value_delta = LoRADelta(base.embed_dim, base.embed_dim, **kwargs)
        self.base.requires_grad_(False)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if (query is not key or query is not value or key_padding_mask is not None
                or attn_mask is not None or need_weights or is_causal):
            raise ValueError("LoRASelfAttention supports unmasked ViT self attention only")
        batch, length, width = query.shape
        q, k, v = F.linear(query, self.base.in_proj_weight, self.base.in_proj_bias).chunk(3, dim=-1)
        if self.query_delta.enabled:
            q = q + self.query_delta(query)
        if self.value_delta.enabled:
            v = v + self.value_delta(query)
        heads = self.base.num_heads
        q, k, v = [tensor.reshape(batch, length, heads, width // heads).transpose(1, 2)
                   for tensor in (q, k, v)]
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.base.dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, length, width)
        return self.base.out_proj(output), None


def cpu_state(module):
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


class S3TModel(nn.Module):
    """Train one new adapter group at a time and reconstruct clean prefixes.

    The backbone is never merged with adapters. Head snapshots are small,
    separate states, because a head updated on later data is not a clean prefix.
    """
    def __init__(self, backbone, kind, config):
        super().__init__()
        self.backbone, self.kind, self.config = backbone, kind, config
        backbone.requires_grad_(False)
        if kind == "torchvision_vit":
            layers = list(backbone.encoder.layers)
            self.head_path = "heads"
        elif kind == "hf_vit":
            layers = list(backbone.vit.encoder.layer)
            self.head_path = "classifier"
        elif kind == "roberta":
            layers = list(backbone.roberta.encoder.layer)
            self.head_path = "classifier"
        else:
            raise ValueError(f"Unsupported model kind: {kind}")
        width = config.layers_per_slice or len(layers) // config.slices
        if width < 1 or width * config.slices > len(layers):
            raise ValueError("slices * layers_per_slice must fit in the transformer depth")
        self.layer_groups = []
        for stage in range(config.slices):
            group = list(range(len(layers) - 1 - stage * width, len(layers) - 1 - (stage + 1) * width, -1))
            self.layer_groups.append(group)
            kwargs = dict(rank=config.rank, alpha=config.alpha, dropout=config.lora_dropout,
                          scaling=config.scaling, stage=stage)
            for index in group:
                layer = layers[index]
                if kind == "torchvision_vit":
                    layer.self_attention = LoRASelfAttention(layer.self_attention, **kwargs)
                else:
                    attention = layer.attention.attention if kind == "hf_vit" else layer.attention.self
                    attention.query = LoRALinear(attention.query, **kwargs)
                    attention.value = LoRALinear(attention.value, **kwargs)
        self.initial_head = cpu_state(self.head)
        self.head_history = []
        self.current_stage = None
        self.active_prefix = 0
        self.requires_grad_(False)

    @property
    def head(self):
        return self.backbone.get_submodule(self.head_path)

    def deltas(self):
        return [(name, module) for name, module in self.named_modules() if isinstance(module, LoRADelta)]

    def forward(self, inputs):
        if self.kind == "torchvision_vit":
            return self.backbone(inputs)
        if self.kind == "hf_vit":
            return self.backbone(pixel_values=inputs).logits
        return self.backbone(input_ids=inputs,
                             attention_mask=inputs.ne(self.backbone.config.pad_token_id)).logits

    def begin_stage(self, stage):
        if not 0 <= stage < self.config.slices or stage != len(self.head_history):
            raise ValueError("Stages must be trained in order")
        self.requires_grad_(False)
        for parameter in self.parameters():
            parameter.grad = None
        for _, delta in self.deltas():
            delta.enabled = delta.stage <= stage
            delta.requires_grad_(delta.stage == stage)
        self.head.load_state_dict(self.head_history[-1] if stage else self.initial_head)
        self.head.requires_grad_(self.config.head_policy == "snapshot" or stage == 0)
        self.current_stage = stage
        self.active_prefix = stage + 1
        self.train()
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def finish_stage(self):
        if self.current_stage != len(self.head_history):
            raise ValueError("No matching stage is being trained")
        head = self.head_history[0] if self.config.head_policy == "first_slice" and self.head_history else cpu_state(self.head)
        self.head_history.append(head)
        self.current_stage = None
        self.requires_grad_(False)
        self.eval()

    def activate_prefix(self, prefix):
        if not 0 <= prefix <= len(self.head_history):
            raise ValueError("Requested prefix has not been trained")
        self.requires_grad_(False)
        for _, delta in self.deltas():
            delta.enabled = delta.stage < prefix
        self.head.load_state_dict(self.head_history[prefix - 1] if prefix else self.initial_head)
        self.active_prefix = prefix
        self.current_stage = None
        self.eval()

    def export_state(self):
        head = (self.head_history[-1] if self.current_stage is None and self.head_history
                and self.active_prefix == len(self.head_history) else cpu_state(self.head))
        return {"adapters": {name: cpu_state(delta) for name, delta in self.deltas()},
                "head_current": head, "head_history": self.head_history,
                "layer_groups": self.layer_groups}

    def load_exported_state(self, state):
        if state["layer_groups"] != self.layer_groups:
            raise ValueError("Checkpoint layer grouping differs from the model")
        deltas = dict(self.deltas())
        if set(state["adapters"]) != set(deltas):
            raise ValueError("Checkpoint adapter layout differs from the model")
        for name, value in state["adapters"].items():
            deltas[name].load_state_dict(value, strict=True)
        self.head_history = [{name: value.detach().cpu().clone() for name, value in head.items()}
                             for head in state["head_history"]]
        if len(self.head_history) > self.config.slices:
            raise ValueError("Too many head snapshots in checkpoint")
        self.head.load_state_dict(state["head_current"], strict=True)

    def adapter_parameter_count(self):
        return sum(parameter.numel() for _, delta in self.deltas() for parameter in delta.parameters())
