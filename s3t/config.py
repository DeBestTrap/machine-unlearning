"""Validated, serializable experiment configuration (paths relative to the repo)."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Config:
    dataset: str = "cifar10"
    model: str = "vit_b_16"
    pretrained: bool = True
    model_name: str | None = None
    revision: str | None = None
    data_root: str = "."
    split_manifest: str | None = None
    shards: int = 5
    slices: int = 6
    budget: int = 6
    sequence_strategy: str = "cyclic"
    deletion_probabilities: list[float] | None = None
    score_horizon: int = 100
    seed: int = 42
    partition_seed: int = 42
    sequence_seed: int = 42
    layers_per_slice: int | None = None
    rank: int = 16
    alpha: float = 32.0
    scaling: str = "lora"
    lora_dropout: float = 0.1
    head_policy: str = "snapshot"
    epochs_per_slice: int = 15
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 0.002
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "constant"
    warmup_ratio: float = 0.0
    max_grad_norm: float | None = 1.0
    selection: str = "longest_prefix"
    aggregation: str = "logits"
    num_workers: int = 4
    precision: str = "fp32"
    deterministic: bool = True
    max_train_examples: int | None = None
    max_eval_examples: int | None = None

    def validate(self):
        choices = {
            "dataset": ("cifar10", "cifar100", "sst2", "synthetic"),
            "model": ("vit_b_16", "hf_vit", "roberta", "tiny_vit", "tiny_roberta"),
            "sequence_strategy": ("cyclic", "bms", "random"),
            "scaling": ("lora", "rslora"),
            "head_policy": ("snapshot", "first_slice"),
            "optimizer": ("adamw", "adam", "sgd"),
            "scheduler": ("constant", "linear", "cosine"),
            "selection": ("longest_prefix", "validation"),
            "aggregation": ("logits", "probabilities", "vote"),
            "precision": ("fp32", "bf16"),
        }
        for key, options in choices.items():
            if getattr(self, key) not in options:
                raise ValueError(f"{key} must be one of {options}")
        for key in ("shards", "slices", "budget", "score_horizon", "rank", "epochs_per_slice",
                    "batch_size", "eval_batch_size"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("seed", "partition_seed", "sequence_seed", "num_workers"):
            if type(getattr(self, key)) is not int or not 0 <= getattr(self, key) < 2**32:
                raise ValueError(f"{key} must be an integer in [0, 2**32)")
        for key in ("layers_per_slice", "max_train_examples", "max_eval_examples"):
            value = getattr(self, key)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{key} must be a positive integer or null")
        for key in ("alpha", "learning_rate", "weight_decay", "lora_dropout", "warmup_ratio"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.alpha <= 0 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("alpha/lr must be positive and weight_decay nonnegative")
        if not 0 <= self.lora_dropout < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Dropout and warmup_ratio must be in [0, 1)")
        if self.max_grad_norm is not None and (not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0):
            raise ValueError("max_grad_norm must be positive or null")
        if self.budget > math.factorial(self.slices):
            raise ValueError("budget exceeds the number of distinct slice permutations")
        if self.sequence_strategy == "bms" and self.deletion_probabilities is None:
            raise ValueError("bms requires deletion_probabilities, one per slice")
        if self.deletion_probabilities is not None:
            from .sequences import validate_probabilities
            validate_probabilities(self.deletion_probabilities, self.slices)
        if type(self.pretrained) is not bool or type(self.deterministic) is not bool:
            raise ValueError("pretrained and deterministic must be booleans")
        if self.dataset == "sst2" and self.model not in ("roberta", "tiny_roberta"):
            raise ValueError("sst2 requires a RoBERTa model")
        if self.dataset.startswith("cifar") and self.model not in ("vit_b_16", "hf_vit"):
            raise ValueError("CIFAR experiments require vit_b_16 or hf_vit")
        if self.dataset == "synthetic" and self.model not in ("tiny_vit", "tiny_roberta"):
            raise ValueError("synthetic is an offline fixture for tiny models only")
        if self.model.startswith("tiny_") and self.pretrained:
            raise ValueError("Tiny smoke-test models must set pretrained=false")
        if self.model == "roberta" and self.model_name not in (None, "roberta-base", "roberta-large",
                                                             "FacebookAI/roberta-base", "FacebookAI/roberta-large"):
            raise ValueError("Prepared SST-2 tokens require the standard RoBERTa vocabulary")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown S3T config keys: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            import yaml
            values = yaml.safe_load(path.read_text())
        else:
            values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Config must be an object")
        return cls.from_dict(values)


def resolve_path(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()
