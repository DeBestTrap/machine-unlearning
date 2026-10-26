"""Data adapters using the repo's verified source IDs and held-out pools."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from dataset_splits import fingerprint_cifar, load_manifest, split_indices
from .config import ROOT, resolve_path


class ImageSubset(Dataset):
    def __init__(self, source, indices, transform):
        self.source, self.indices, self.transform = source, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[self.indices[index]]
        return self.transform(image), label


@dataclass
class DataBundle:
    train: Dataset
    val: Dataset
    test: Dataset
    source_ids: list[int]
    split: dict
    num_classes: int
    preprocessing: dict

    def __post_init__(self):
        if len(self.source_ids) != len(self.train) or len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("Training data/source mapping mismatch")
        self.source_to_local = {source_id: index for index, source_id in enumerate(self.source_ids)}

    def subset(self, source_ids):
        try:
            return Subset(self.train, [self.source_to_local[index] for index in source_ids])
        except KeyError as exc:
            raise ValueError(f"Source ID is outside the training data: {exc.args[0]}") from exc


def _manifest_path(config):
    if config.split_manifest:
        return resolve_path(config.split_manifest)
    name = {"cifar10": "cifar10-val2500-seed42.json", "cifar100": "cifar100-val2500-seed42.json",
            "sst2": "glue-sst2-val3367-seed42.json"}[config.dataset]
    return ROOT / "splits" / name


def _cap(bundle, config):
    if config.max_train_examples is not None and config.max_train_examples < len(bundle.train):
        chosen = np.random.default_rng(config.partition_seed).permutation(len(bundle.train))[:config.max_train_examples]
        bundle.train = Subset(bundle.train, chosen.tolist())
        bundle.source_ids = [bundle.source_ids[i] for i in chosen]
    if config.max_eval_examples is not None:
        bundle.val = Subset(bundle.val, list(range(min(len(bundle.val), config.max_eval_examples))))
        bundle.test = Subset(bundle.test, list(range(min(len(bundle.test), config.max_eval_examples))))
    bundle.__post_init__()
    if not len(bundle.val) or not len(bundle.test):
        raise ValueError("Validation and test pools must be nonempty")
    return bundle


def _synthetic(config):
    generator = torch.Generator().manual_seed(1729)
    if config.model == "tiny_roberta":
        inputs = torch.randint(4, 64, (120, 12), generator=generator)
        inputs[:, 0], inputs[:, -1] = 0, 2
        labels = (inputs[:, 1] % 3).long()
        preprocessing = {"kind": "synthetic_tokens", "length": 12}
    else:
        inputs = torch.randn(120, 3, 16, 16, generator=generator)
        labels = inputs.mean(dim=(2, 3)).argmax(dim=1)
        preprocessing = {"kind": "synthetic_images", "size": 16}
    train_ids, val_ids = split_indices(labels[:96], 18, seed=42)
    source_hash = fingerprint_cifar(inputs[:96].numpy(), labels[:96])
    split = {"schema_version": 1, "dataset_id": "synthetic/" + config.model, "source_split": "train",
             "source_size": 96, "source_sha256": source_hash, "split_seed": 42,
             "stratified": True, "train_indices": train_ids, "val_indices": val_ids}
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    split["manifest_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    source = TensorDataset(inputs[:96], labels[:96])
    return DataBundle(Subset(source, train_ids), Subset(source, val_ids),
                      TensorDataset(inputs[96:], labels[96:]), train_ids, split, 3, preprocessing)


def load_data(config, preprocessing=None):
    if config.dataset == "synthetic":
        return _cap(_synthetic(config), config)
    root = resolve_path(config.data_root)
    if config.dataset.startswith("cifar"):
        from torchvision import datasets, transforms
        cls = datasets.CIFAR10 if config.dataset == "cifar10" else datasets.CIFAR100
        source = cls(str(root), train=True, download=False)
        test = cls(str(root), train=False, download=False)
        split = load_manifest(_manifest_path(config), dataset_id=config.dataset, source_size=len(source),
                              source_sha256=fingerprint_cifar(source.data, source.targets))
        if preprocessing is None:
            if config.model == "hf_vit":
                from transformers import AutoImageProcessor
                processor = AutoImageProcessor.from_pretrained(
                    config.model_name or "google/vit-base-patch16-224-in21k", revision=config.revision,
                )
                size = processor.size.get("height", processor.size.get("shortest_edge", 224))
                mean, std = processor.image_mean, processor.image_std
            else:
                size, mean, std = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            preprocessing = {"size": size, "mean": mean, "std": std,
                             "train": "random_resized_crop_horizontal_flip", "eval": "resize_center_crop"}
        size = preprocessing["size"]
        normalize = transforms.Normalize(preprocessing["mean"], preprocessing["std"])
        train_transform = transforms.Compose([transforms.RandomResizedCrop(size),
                                               transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
        eval_transform = transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size),
                                              transforms.ToTensor(), normalize])
        bundle = DataBundle(ImageSubset(source, split["train_indices"], train_transform),
                            ImageSubset(source, split["val_indices"], eval_transform),
                            ImageSubset(test, range(len(test)), eval_transform), split["train_indices"], split,
                            10 if config.dataset == "cifar10" else 100, preprocessing)
    else:
        path = root / "datasets" / "sst2"
        train_path, test_path = path / "sst2_train.npz", path / "sst2_test.npz"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError("Prepare SST-2 first: python datasets/sst2/prepare_data.py")
        with np.load(train_path, allow_pickle=False) as arrays:
            if "source_sha256" not in arrays:
                raise ValueError("SST-2 arrays predate verified source provenance; prepare them again")
            inputs = torch.from_numpy(arrays["X"].astype(np.int64))
            labels = torch.from_numpy(arrays["y"].astype(np.int64))
            source_hash = str(arrays["source_sha256"].item())
        with np.load(test_path, allow_pickle=False) as arrays:
            test = TensorDataset(torch.from_numpy(arrays["X"].astype(np.int64)),
                                 torch.from_numpy(arrays["y"].astype(np.int64)))
        split = load_manifest(_manifest_path(config), dataset_id="glue/sst2", source_size=len(labels),
                              source_sha256=source_hash)
        source = TensorDataset(inputs, labels)
        # Bind prepared tokens as well as the original raw-source fingerprint.
        token_hash = hashlib.sha256(inputs.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        test_hash = hashlib.sha256(test.tensors[0].numpy().tobytes() + test.tensors[1].numpy().tobytes()).hexdigest()
        actual = {"tokenizer": "roberta-base", "max_length": inputs.shape[1],
                  "training_arrays_sha256": token_hash, "test_arrays_sha256": test_hash}
        if preprocessing is not None and actual != preprocessing:
            raise ValueError("SST-2 prepared arrays changed since this run was initialized")
        bundle = DataBundle(Subset(source, split["train_indices"]), Subset(source, split["val_indices"]),
                            test, split["train_indices"], split, 2, actual)
    return _cap(bundle, config)
