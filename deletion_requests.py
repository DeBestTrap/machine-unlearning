"""Portable deletion workloads. Keep this file identical in both repositories.

IDs refer to the ordered original training source, before holdouts or sharding.
Only generation uses an RNG; replay always consumes the saved batches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random

from dataset_splits import _json, load_manifest


def checksum(manifest):
    body = {key: value for key, value in manifest.items() if key != "workload_sha256"}
    return hashlib.sha256(_json(body).encode("utf-8")).hexdigest()


def validate_workload(workload, split):
    if workload.get("workload_sha256") != checksum(workload):
        raise ValueError("Deletion workload checksum mismatch")
    if workload.get("schema_version") != 1:
        raise ValueError("Unsupported deletion workload schema")
    for key in ("dataset_id", "source_sha256"):
        if workload.get(key) != split[key]:
            raise ValueError(f"Deletion workload {key} mismatch")
    if workload.get("split_manifest_sha256") != split["manifest_sha256"]:
        raise ValueError("Deletion workload split manifest mismatch")
    batches = workload.get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("Deletion workload requires non-empty batches")
    pool = set(split["train_indices"])
    seen = set()
    for batch in batches:
        if not isinstance(batch, list) or not batch:
            raise ValueError("Deletion batches must be non-empty lists")
        for source_id in batch:
            if type(source_id) is not int or source_id not in pool:
                raise ValueError("Deletion source index is outside the training pool")
            if source_id in seen:
                raise ValueError("Duplicate deletion source index across workload")
            seen.add(source_id)
    return workload


def load_workload(path, split):
    return validate_workload(json.loads(Path(path).read_text(encoding="utf-8")), split)


def generate_workload(split, *, seed=123, steps=100, batch_size=1):
    if type(seed) is not int:
        raise ValueError("Deletion seed must be an integer")
    if any(type(value) is not int or value <= 0 for value in (steps, batch_size)):
        raise ValueError("Steps and batch size must be positive integers")
    count = steps * batch_size
    if count > len(split["train_indices"]):
        raise ValueError("Deletion workload exceeds the training pool")
    selected = random.Random(seed).sample(split["train_indices"], count)
    workload = dict(
        schema_version=1, dataset_id=split["dataset_id"],
        source_sha256=split["source_sha256"],
        split_manifest_sha256=split["manifest_sha256"], deletion_seed=seed,
        batches=[selected[start:start + batch_size] for start in range(0, count, batch_size)],
    )
    workload["workload_sha256"] = checksum(workload)
    return validate_workload(workload, split)


def local_batches(workload, split, source_indices=None):
    """Translate IDs and require the exact full training pool in either method."""
    validate_workload(workload, split)
    source_indices = list(split["train_indices"] if source_indices is None else source_indices)
    if (len(source_indices) != len(split["train_indices"])
            or set(source_indices) != set(split["train_indices"])):
        raise ValueError("Shared deletions require the full training pool; disable training caps")
    inverse = {source_id: local_id for local_id, source_id in enumerate(source_indices)}
    return [[inverse[source_id] for source_id in batch] for batch in workload["batches"]]


def bind_training_source(dataset, split):
    """Attach provenance where a verified one-example-per-source dataset is built."""
    if len(dataset) != len(split["train_indices"]):
        raise ValueError("Training dataset length disagrees with source mapping")
    dataset.deletion_source_indices = list(split["train_indices"])
    dataset.deletion_split_manifest = split
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    split = load_manifest(args.split_manifest)
    workload = generate_workload(split, seed=args.seed, steps=args.steps, batch_size=args.batch_size)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never replace a saved workload accidentally. Identical reruns are harmless.
    if path.exists():
        if load_workload(path, split) != workload:
            raise ValueError(f"Existing deletion workload differs: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_json(workload) + "\n")
    print(json.dumps(dict(path=str(path.resolve()), workload_sha256=workload["workload_sha256"],
                          steps=len(workload["batches"]), unique_deletions=sum(map(len, workload["batches"])))))


if __name__ == "__main__":
    main()
