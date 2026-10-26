"""Convert a portable deletion workload into cumulative SISA request files."""
import json
from pathlib import Path

import numpy as np

from deletion_requests import load_workload, local_batches
from split_support import configured_manifest, validate_container_split


def export_deletion_requests(container, dataset, workload_path):
    validate_container_split(container, dataset)
    split = configured_manifest(dataset)
    if split is None:
        raise ValueError("Shared deletions require a verified dataset split manifest")
    workload = load_workload(workload_path, split)
    batches = local_batches(workload, split)
    directory = Path("containers") / container
    shards = np.load(directory / "splitfile.npy", allow_pickle=True)
    flattened = [int(idx) for shard in shards for idx in shard]
    if sorted(flattened) != list(range(len(split["train_indices"]))):
        raise ValueError("SISA shards must cover the full training pool exactly once")
    marker = directory / "deletion_requests.json"
    if marker.exists() and json.loads(marker.read_text()) != workload:
        raise ValueError("Container already contains a different deletion workload")

    # Preflight every label before writing any files. Existing matching labels
    # can be reused, but different requests must use a new container.
    planned = []
    cumulative = set()
    for label in range(len(batches) + 1):
        if label:
            cumulative.update(batches[label - 1])
        arrays = [np.asarray(sorted(cumulative.intersection(map(int, shard))), dtype=np.int64)
                  for shard in shards]
        path = directory / f"requestfile:{label}.npy"
        if path.exists():
            previous = np.load(path, allow_pickle=True)
            if len(previous) != len(arrays) or any(
                not np.array_equal(old, new) for old, new in zip(previous, arrays)
            ):
                raise ValueError(f"Existing requests differ at label {label}; use a new container")
        planned.append((path, arrays))
    for path, arrays in planned:
        requests = np.empty(len(arrays), dtype=object)
        requests[:] = arrays
        np.save(path, requests, allow_pickle=True)
    marker.write_text(json.dumps(workload, indent=2, sort_keys=True) + "\n")
    return workload
