"""Expected SISA/ProtoSISA shard replay work under the paper's nominal model.

Source: BlockUnlearn/.paper/sections/appendix_ablations.tex, "Comparison of
expected retraining cost". Assumes equal shards/slices, uniform single-example
deletions, and fractional epochs 2 * epochs / (slices + 1) per cumulative stage.
Costs count examples processed, including repeated passes, rather than seconds.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
import math
from numbers import Real
import sys


def _integer(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def normalized_expected_cost(shards, slices):
    """Expected replay work per deletion divided by full retraining (epochs * n).

    Returns (2/3 + 1/(3 * slices)) / shards for both SISA and ProtoSISA's
    gradient training. Prototype maintenance and other update overhead are excluded.
    """
    _integer("shards", shards)
    _integer("slices", slices)
    return (2 / 3 + 1 / (3 * slices)) / shards


def expected_cost(shards, slices, training_examples, *, epochs=1.0, requests=1):
    """Expected examples processed across independent single-example requests.

    Uses a fixed nominal training pool, ignoring removed examples, integer epoch
    rounding, and uneven shard/slice sizes, as in the paper. This is not the cost
    of batching several deletions into one replay.
    """
    normalized = normalized_expected_cost(shards, slices)
    _integer("training_examples", training_examples)
    _integer("requests", requests, minimum=0)
    if isinstance(epochs, bool) or not isinstance(epochs, Real) or not math.isfinite(epochs) or epochs <= 0:
        raise ValueError("epochs must be finite and positive")
    if training_examples < shards * slices:
        raise ValueError("training_examples must allow at least one example per slice in every shard")
    return normalized * epochs * training_examples * requests


def cost_grid(shards, slices, *, training_examples=None, epochs=1.0, requests=1):
    """Return one row per shard/slice pair, optionally scaled to a training pool."""
    shards, slices = list(shards), list(slices)
    if not shards or not slices:
        raise ValueError("Supply at least one shard count and one slice count")
    rows = []
    for shard_count, slice_count in product(shards, slices):
        # Validate epochs/requests even when only normalized costs are requested.
        per_request = expected_cost(shard_count, slice_count,
                                    training_examples if training_examples is not None else shard_count * slice_count,
                                    epochs=epochs)
        _integer("requests", requests, minimum=0)
        rows.append({
            "shards": shard_count, "slices": slice_count,
            "normalized_expected_cost": normalized_expected_cost(shard_count, slice_count),
            "training_examples": training_examples, "epochs": epochs, "requests": requests,
            "expected_examples_per_request": per_request if training_examples is not None else None,
            "expected_total_examples": per_request * requests if training_examples is not None else None,
        })
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=int, required=True, help="Shard counts to compare")
    parser.add_argument("--slices", nargs="+", type=int, required=True, help="Slice counts per shard to compare")
    parser.add_argument("--training-examples", type=int, help="Training/deletion pool size after holdout")
    parser.add_argument("--epochs", type=float, default=1.0, help="Nominal full-data epoch budget (default: 1)")
    parser.add_argument("--requests", type=int, default=1, help="Independent single-example requests (default: 1)")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args(argv)
    try:
        rows = cost_grid(args.shards, args.slices, training_examples=args.training_examples,
                         epochs=args.epochs, requests=args.requests)
    except ValueError as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(rows, indent=2, allow_nan=False))
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
