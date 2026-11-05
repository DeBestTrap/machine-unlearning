"""Expected SISA/ProtoSISA shard replay work under the paper's nominal model.

Source: BlockUnlearn/.paper/sections/appendix_ablations.tex, "Comparison of
expected retraining cost". Assumes equal shards/slices, uniform single-example
deletions, and fractional epochs 2 * epochs / (slices + 1) per cumulative stage.
Costs count examples processed, including repeated passes, rather than seconds.
Normalization uses one common reference epoch budget, not each candidate's own.
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


def _positive_epochs(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def normalized_expected_cost(shards, slices, *, epochs=1.0, reference_epochs=1.0):
    """Expected replay work divided by a common reference_epochs * n.

    Returns (epochs/reference_epochs) * (2/3 + 1/(3*slices)) / shards for
    SISA and ProtoSISA gradient training. Prototype maintenance is excluded.
    This nominal analytical model also supports historical fractional schedules;
    training eligibility is checked separately by sisa_schedule.epochs_per_slice.
    """
    _integer("shards", shards)
    _integer("slices", slices)
    _positive_epochs("epochs", epochs)
    _positive_epochs("reference_epochs", reference_epochs)
    return (epochs / reference_epochs) * (2 / 3 + 1 / (3 * slices)) / shards


def expected_cost(shards, slices, training_examples, *, epochs=1.0, requests=1):
    """Expected examples processed across independent single-example requests.

    Uses a fixed nominal training pool, ignoring removed examples, integer epoch
    rounding, and uneven shard/slice sizes, as in the paper. This is not the cost
    of batching several deletions into one replay.
    """
    normalized = normalized_expected_cost(shards, slices, epochs=epochs)
    _integer("training_examples", training_examples)
    _integer("requests", requests, minimum=0)
    if training_examples < shards * slices:
        raise ValueError("training_examples must allow at least one example per slice in every shard")
    return normalized * training_examples * requests


def cost_grid(shards, slices, *, training_examples=None, epochs=1.0, requests=1, reference_epochs=1.0):
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
            "normalized_expected_cost": normalized_expected_cost(
                shard_count, slice_count, epochs=epochs, reference_epochs=reference_epochs),
            "training_examples": training_examples, "epochs": epochs,
            "reference_epochs": reference_epochs, "requests": requests,
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
    parser.add_argument("--reference-epochs", type=float, default=1.0,
                        help="Common full-retraining epoch budget used for normalization (default: 1)")
    parser.add_argument("--requests", type=int, default=1, help="Independent single-example requests (default: 1)")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    args = parser.parse_args(argv)
    try:
        rows = cost_grid(args.shards, args.slices, training_examples=args.training_examples,
                         epochs=args.epochs, requests=args.requests, reference_epochs=args.reference_epochs)
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
