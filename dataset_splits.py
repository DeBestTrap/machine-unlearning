"""Portable, verified training holdouts, shared with machine-unlearning.

Indices refer to the original, ordered training source, before tokenization,
augmentation, caps, sharding or deletions. Loading a manifest never resamples it.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile


DEFAULT_SPLIT_SEED = 42
SCHEMA_VERSION = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint_records(records):
    """Hash canonical raw records, preserving their order and content."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_cifar(images, labels):
    digest = hashlib.sha256()
    digest.update(_json(list(images.shape)).encode("ascii"))
    digest.update(str(images.dtype).encode("ascii"))
    digest.update(images.tobytes(order="C"))
    digest.update(_json([int(label) for label in labels]).encode("ascii"))
    return digest.hexdigest()


def split_seed(config):
    val_seed = config.get("val_args", {}).get("split_seed")
    data_seed = config.get("dataset_args", {}).get("split_seed")
    if val_seed is not None and data_seed is not None and val_seed != data_seed:
        raise ValueError("val_args.split_seed and dataset_args.split_seed disagree")
    seed = val_seed if val_seed is not None else data_seed
    if seed is None:
        seed = DEFAULT_SPLIT_SEED
    if type(seed) is not int:
        raise ValueError("split_seed must be an integer")
    return seed


def split_indices(labels, val_size, seed=DEFAULT_SPLIT_SEED, *, stratify=True):
    import torch

    total = len(labels)
    if type(val_size) is not int or not 0 <= val_size < total:
        raise ValueError("Validation size must be an integer in [0, source_size)")
    if not val_size:
        return list(range(total)), []
    generator = torch.Generator().manual_seed(seed)
    if not stratify:
        indices = torch.randperm(total, generator=generator).tolist()
        return sorted(indices[val_size:]), sorted(indices[:val_size])
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    classes = sorted(groups)
    exact = {label: val_size * len(groups[label]) / total for label in classes}
    sizes = {label: int(exact[label]) for label in classes}
    remaining = val_size - sum(sizes.values())
    for label in sorted(classes, key=lambda label: (-(exact[label] - sizes[label]), label))[:remaining]:
        sizes[label] += 1
    train, val = [], []
    for label in classes:
        indices = groups[label]
        shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        val.extend(shuffled[:sizes[label]])
        train.extend(shuffled[sizes[label]:])
    return sorted(train), sorted(val)


def load_manifest(path, *, dataset_id=None, source_size=None, source_sha256=None):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    checksum = manifest.pop("manifest_sha256", None)
    if checksum != hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest():
        raise ValueError(f"Split manifest checksum mismatch: {path}")
    manifest["manifest_sha256"] = checksum
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("source_split") != "train":
        raise ValueError(f"Unsupported split manifest: {path}")
    for name, expected in (("dataset_id", dataset_id), ("source_size", source_size), ("source_sha256", source_sha256)):
        if expected is not None and manifest.get(name) != expected:
            raise ValueError(f"Split manifest {name} mismatch: {path}")
    train, val = manifest["train_indices"], manifest["val_indices"]
    size = manifest["source_size"]
    indices = train + val
    if (not train or type(size) is not int or any(type(i) is not int for i in indices)
            or len(indices) != size or sorted(indices) != list(range(size))):
        raise ValueError(f"Split indices must be disjoint and cover the training source: {path}")
    return manifest


def get_manifest(dataset_id, labels, val_size, source_sha256, *, seed=DEFAULT_SPLIT_SEED,
                 stratify=True, path=None):
    """Read an explicit manifest, or atomically create/load a default one.

    Explicit paths must already exist: a typo must not create a different pool.
    Concurrent jobs publish complete files using an exclusive hard link.
    """
    explicit = path is not None
    if path is None:
        directory = Path(os.environ.get("DATASET_SPLIT_DIR", Path(__file__).resolve().parent / "splits"))
        path = directory / f"{dataset_id.replace('/', '-')}-val{val_size}-seed{seed}.json"
    path = Path(path).expanduser().resolve()
    if not path.exists() and not explicit:
        train, val = split_indices(labels, val_size, seed, stratify=stratify)
        manifest = dict(schema_version=SCHEMA_VERSION, dataset_id=dataset_id, source_split="train",
                        source_size=len(labels), source_sha256=source_sha256, split_seed=seed,
                        stratified=stratify, train_indices=train, val_indices=val)
        manifest["manifest_sha256"] = hashlib.sha256(_json(manifest).encode("utf-8")).hexdigest()
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(_json(manifest) + "\n")
        try:
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
    manifest = load_manifest(path, dataset_id=dataset_id, source_size=len(labels), source_sha256=source_sha256)
    if (len(manifest["val_indices"]) != val_size or manifest["split_seed"] != seed
            or manifest["stratified"] != stratify):
        raise ValueError(f"Split manifest disagrees with requested validation size/seed/stratification: {path}")
    return manifest, path


def remember_split(config, manifest, path):
    """Keep the split identity in the resolved config and per-run metadata."""
    config.setdefault("dataset_args", {})["resolved_split"] = {
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_path": str(path),
        "dataset_id": manifest["dataset_id"],
        "split_seed": manifest["split_seed"],
        "source_sha256": manifest["source_sha256"],
        "train_size": len(manifest["train_indices"]),
        "val_size": len(manifest["val_indices"]),
    }


def record_experiment_split(config, run_dir, existing_models_dir=None):
    resolved = config.get("dataset_args", {}).get("resolved_split")
    if resolved is None:
        return
    data_args = config["dataset_args"]
    record = dict(resolved)
    record["caps"] = {name: data_args.get(f"max_{name}_examples") for name in ("train", "val", "test")}
    # Training caps may vary with the experiment seed; evaluation caps stay fixed.
    record["train_cap_seed"] = config.get("experiment_args", {}).get("seed", 42) if record["caps"]["train"] is not None else None
    if existing_models_dir is not None:
        previous_path = Path(existing_models_dir) / "dataset_split.json"
        if not previous_path.exists():
            raise ValueError("Existing training has no verified dataset split. Retrain with the fixed split before unlearning.")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        keys = ("manifest_sha256", "caps", "train_cap_seed")
        if any(previous.get(key) != record.get(key) for key in keys):
            raise ValueError("Existing training used a different dataset split or cap. Retrain with the requested split.")
    Path(run_dir, "dataset_split.json").write_text(_json(record) + "\n", encoding="utf-8")
