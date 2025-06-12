"""Distribution helper for SISA / machine‑unlearning
---------------------------------------------------
This script creates the per‑shard **splitfile.npy** and, optionally, one or
more **requestfile:*.npy** files inside a `containers/<experiment name>` folder.

Modern NumPy (>=1.24) no longer allows saving ragged lists directly with
``np.save``.  Every time we write the shard index lists we therefore convert
them to an *object‑dtype* array and pass ``allow_pickle=True``.  Readers in the
original repo already load with that flag, so no other file needs to change.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--shards",
    type=int,
    default=None,
    help="Split the dataset into N shards and create splitfile.npy",
)
parser.add_argument(
    "--requests",
    type=int,
    default=None,
    help="Generate N unlearning requests and update requestfile",
)
parser.add_argument(
    "--distribution",
    default="uniform",
    help=(
        "Distribution to assume / sample from.  With --shards it means how the "
        "points are distributed in the dataset ('uniform', 'exponential:λ', "
        "'pareto:α').  With --requests it chooses the sampling law for the "
        "requests.  Use 'reset' together with --requests to zero the request "
        "file."),
)
parser.add_argument(
    "--container",
    default="default",
    help="Name of the experiment container (folder under ./containers/)",
)
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Path to the dataset metadata file produced by prepare_data.py",
)
parser.add_argument(
    "--label",
    default="latest",
    help="Label that distinguishes successive request files",
)
# Parameter used only by the PLS‑GAP branch (non‑uniform partitioning).
parser.add_argument(
    "--algo",
    default="gap:0.01",
    help="PLS‑GAP parameter as algo:limit_fraction (default gap:0.01)",
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _save_object(path: str | Path, data: List[np.ndarray] | np.ndarray) -> None:
    """Save *ragged* data safely with NumPy >=1.24."""
    arr = np.asarray(data, dtype=object)
    np.save(str(path), arr, allow_pickle=True)


def _ensure_container() -> Path:
    base = Path("containers") / args.container
    base.mkdir(parents=True, exist_ok=True)
    return base


def _load_dataset() -> dict:
    with open(args.dataset) as f:
        return json.load(f)


datasetfile = _load_dataset()
container_dir = _ensure_container()

# -----------------------------------------------------------------------------
# --shards : build splitfile + empty request file
# -----------------------------------------------------------------------------
if args.shards is not None:
    nb_train = datasetfile["nb_train"]

    # Uniform split: just chunk the indices.
    if args.distribution == "uniform":
        partition = np.array_split(np.arange(nb_train), args.shards)

    # Non‑uniform split using PLS‑GAP (legacy code, rarely used in the paper).
    else:
        def mass(index: np.ndarray) -> np.ndarray:
            if args.distribution.startswith("exponential"):
                lbd = float(args.distribution.split(":")[1]) if ":" in args.distribution else -np.log(0.05) / nb_train
                return np.exp(-lbd * index) - np.exp(-lbd * (index + 1))
            if args.distribution.startswith("pareto"):
                a = float(args.distribution.split(":")[1]) if ":" in args.distribution else 1.16
                return a / ((index + 1) ** (a + 1))
            raise ValueError("Unsupported distribution for PLS‑GAP")

        weights = mass(np.arange(nb_train))
        indices = np.argsort(weights)
        queue = np.vstack((weights[indices], np.ones_like(weights))).T
        partition = [np.array([idx]) for idx in indices]

        bottom_queue = queue.shape[0]
        lim_fraction = float(args.algo.split(":")[1]) if ":" in args.algo else 0.01
        lim = int(lim_fraction * nb_train)

        for _ in range(nb_train - args.shards):
            w1, w2 = queue[0], queue[1]
            l1, l2 = partition[0], partition[1]

            queue = queue[2:]
            partition = partition[2:]
            bottom_queue -= 2

            merged_weight = w1 + w2

            if merged_weight[1] < lim:
                offset = np.argmax(queue[:bottom_queue, 1] < merged_weight[1]) if bottom_queue else 0
                limit = np.argmax(queue[:bottom_queue, 1] > merged_weight[1]) if bottom_queue else 0
                position = offset + np.argmax(queue[offset:limit, 0] < merged_weight[0]) if limit > offset else bottom_queue
                bottom_queue += 1
            else:
                position = bottom_queue + np.argmax(queue[bottom_queue:, 0] < merged_weight[0])

            queue = np.insert(queue, position, merged_weight, axis=0)
            partition = partition[:position] + [np.concatenate((l1, l2))] + partition[position:]

    # --- Write files --------------------------------------------------------
    _save_object(container_dir / "splitfile.npy", partition)
    empty_requests = np.array([np.array([], dtype=int) for _ in range(len(partition))], dtype=object)
    _save_object(container_dir / f"requestfile:{args.label}.npy", empty_requests)

# -----------------------------------------------------------------------------
# --requests : generate or reset unlearning requests
# -----------------------------------------------------------------------------
if args.requests is not None:
    splitfile = np.load(container_dir / "splitfile.npy", allow_pickle=True)
    nb_shards = len(splitfile)

    if args.distribution == "reset":
        new_requests = np.array([np.array([], dtype=int) for _ in range(nb_shards)], dtype=object)
    else:
        # Draw indices according to the chosen law.
        if args.distribution.startswith("exponential"):
            lbd = float(args.distribution.split(":")[1]) if ":" in args.distribution else -np.log(0.05) / datasetfile["nb_train"]
            all_requests = np.random.exponential(1 / lbd, args.requests).astype(int)
        elif args.distribution.startswith("pareto"):
            a = float(args.distribution.split(":")[1]) if ":" in args.distribution else 1.16
            all_requests = np.random.pareto(a, args.requests).astype(int)
        else:  # uniform
            all_requests = np.random.randint(0, datasetfile["nb_train"], args.requests)

        new_requests = []
        for shard in range(nb_shards):
            new_requests.append(np.intersect1d(splitfile[shard], all_requests))
        new_requests = np.array(new_requests, dtype=object)

    _save_object(container_dir / f"requestfile:{args.label}.npy", new_requests)
