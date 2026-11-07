"""Data adapters using the repo's verified source IDs and held-out pools."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from dataset_splits import _json, fingerprint_cifar, load_manifest, split_indices
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
    test: Dataset | None
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


def development_split(config, parent):
    """Validate the fixed 10% source-index pools and select one evaluation pool."""
    path = resolve_path(config.development_manifest)
    value = json.loads(path.read_text())
    checksum = value.pop("manifest_sha256", None)
    if checksum != hashlib.sha256(_json(value).encode()).hexdigest():
        raise ValueError("Development manifest checksum mismatch")
    value["manifest_sha256"] = checksum
    fit, score, confirm = (value[key] for key in
                           ("train_indices", "val_indices", "confirmation_indices"))
    source_size = parent["source_size"]
    if (value.get("schema_version") != 2 or value.get("purpose") != "allocation_search"
            or value.get("source_split") != "train" or value.get("pool_source") != "fixed_validation"
            or value.get("dataset_id") != parent["dataset_id"]
            or value.get("source_size") != source_size
            or value.get("source_sha256") != parent["source_sha256"]
            or value.get("parent_manifest_sha256") != parent["manifest_sha256"]
            or value.get("split_seed") != 42 or parent["split_seed"] != 42
            or not value.get("stratified") or not parent["stratified"]
            or len(parent["val_indices"]) != source_size // 10
            or [len(fit), len(score), len(confirm)] != [len(parent["val_indices"]) - 2 * (len(parent["val_indices"]) // 5),
                                                       len(parent["val_indices"]) // 5,
                                                       len(parent["val_indices"]) // 5]
            or sorted(fit + score + confirm) != sorted(parent["val_indices"])
            or len(set(fit + score + confirm)) != len(parent["val_indices"])
            or set(confirm) & set(value.get("confirmation_excluded_indices", []))):
        raise ValueError("Development memberships disagree with the fixed 10% protocol")
    selected = score if config.development_pool == "scoring" else confirm
    effective = {**value, "val_indices": selected, "evaluation_pool": config.development_pool,
                 "purpose": "s3t_development_" + config.development_pool,
                 "development_manifest_sha256": checksum}
    effective.pop("manifest_sha256")
    effective["manifest_sha256"] = hashlib.sha256(_json(effective).encode()).hexdigest()
    return effective


def load_saved_split(path, config):
    if config.development_manifest is None:
        return load_manifest(path)
    parent = load_manifest(_manifest_path(config))
    expected = development_split(config, parent)
    saved = json.loads(path.read_text())
    if saved != expected:
        raise ValueError("Saved S3T development split differs from its manifests")
    return saved


def _cap(bundle, config):
    if config.max_train_examples is not None and config.max_train_examples < len(bundle.train):
        chosen = np.random.default_rng(config.partition_seed).permutation(len(bundle.train))[:config.max_train_examples]
        bundle.train = Subset(bundle.train, chosen.tolist())
        bundle.source_ids = [bundle.source_ids[i] for i in chosen]
    if config.max_eval_examples is not None:
        bundle.val = Subset(bundle.val, list(range(min(len(bundle.val), config.max_eval_examples))))
        if bundle.test is not None:
            bundle.test = Subset(bundle.test, list(range(min(len(bundle.test), config.max_eval_examples))))
    bundle.__post_init__()
    if not len(bundle.val) or (bundle.test is not None and not len(bundle.test)):
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
        source = cls(str(root), train=True, download=bool(config.development_manifest))
        test = None if config.development_manifest else cls(str(root), train=False, download=False)
        parent = load_manifest(_manifest_path(config), dataset_id=config.dataset, source_size=len(source),
                               source_sha256=fingerprint_cifar(source.data, source.targets))
        split = development_split(config, parent) if config.development_manifest else parent
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
                            ImageSubset(test, range(len(test)), eval_transform) if test is not None else None,
                            split["train_indices"], split,
                            10 if config.dataset == "cifar10" else 100, preprocessing)
    else:
        path = root / "datasets" / "sst2"
        train_path, test_path = path / "sst2_train.npz", path / "sst2_test.npz"
        if not train_path.exists() or (not config.development_manifest and not test_path.exists()):
            raise FileNotFoundError("Prepare SST-2 first: python datasets/sst2/prepare_data.py")
        with np.load(train_path, allow_pickle=False) as arrays:
            if "source_sha256" not in arrays:
                raise ValueError("SST-2 arrays predate verified source provenance; prepare them again")
            inputs = torch.from_numpy(arrays["X"].astype(np.int64))
            labels = torch.from_numpy(arrays["y"].astype(np.int64))
            source_hash = str(arrays["source_sha256"].item())
        test = None
        if not config.development_manifest:
            with np.load(test_path, allow_pickle=False) as arrays:
                test = TensorDataset(torch.from_numpy(arrays["X"].astype(np.int64)),
                                     torch.from_numpy(arrays["y"].astype(np.int64)))
        parent = load_manifest(_manifest_path(config), dataset_id="glue/sst2", source_size=len(labels),
                               source_sha256=source_hash)
        split = development_split(config, parent) if config.development_manifest else parent
        source = TensorDataset(inputs, labels)
        # Bind prepared tokens as well as the original raw-source fingerprint.
        token_hash = hashlib.sha256(inputs.numpy().tobytes() + labels.numpy().tobytes()).hexdigest()
        actual = {"tokenizer": "roberta-base", "max_length": inputs.shape[1],
                  "training_arrays_sha256": token_hash}
        if test is not None:
            actual["test_arrays_sha256"] = hashlib.sha256(
                test.tensors[0].numpy().tobytes() + test.tensors[1].numpy().tobytes()).hexdigest()
        if preprocessing is not None and actual != preprocessing:
            raise ValueError("SST-2 prepared arrays changed since this run was initialized")
        bundle = DataBundle(Subset(source, split["train_indices"]), Subset(source, split["val_indices"]),
                            test, split["train_indices"], split, 2, actual)
    return _cap(bundle, config)
