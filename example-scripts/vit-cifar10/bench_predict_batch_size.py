#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time


def _parse_batch_sizes(raw: list[str]) -> list[int]:
    if len(raw) == 1 and "," in raw[0]:
        raw = [item.strip() for item in raw[0].split(",") if item.strip()]
    sizes = [int(item) for item in raw]
    if not sizes:
        raise ValueError("batch sizes list is empty")
    return sizes


def _run_predict(
    *,
    model: str,
    dataset_path: str,
    container: str,
    shard: int,
    label: str,
    batch_size: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "sisa.py",
            "--model",
            model,
            "--test",
            "--dataset",
            dataset_path,
            "--label",
            label,
            "--batch_size",
            str(batch_size),
            "--container",
            container,
            "--shard",
            str(shard),
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark SISA prediction time vs batch size for ViT/CIFAR-10."
    )
    parser.add_argument("--model", default="vit")
    parser.add_argument("--dataset", default="datasets/cifar10/datasetfile")
    parser.add_argument("--container", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--label", default="0")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        default=["32", "64", "96", "128", "180", "256"],
        help="Space- or comma-separated list of batch sizes.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    results: list[tuple[int, float, float]] = []

    print("Benchmarking predict batch sizes...")
    for batch_size in batch_sizes:
        for _ in range(args.warmup):
            _run_predict(
                model=args.model,
                dataset_path=args.dataset,
                container=args.container,
                shard=args.shard,
                label=args.label,
                batch_size=batch_size,
            )

        timings: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            _run_predict(
                model=args.model,
                dataset_path=args.dataset,
                container=args.container,
                shard=args.shard,
                label=args.label,
                batch_size=batch_size,
            )
            timings.append(time.perf_counter() - start)

        mean = statistics.mean(timings)
        stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
        results.append((batch_size, mean, stdev))
        print(f"batch_size={batch_size:>4} mean={mean:.3f}s stdev={stdev:.3f}s")

    results.sort(key=lambda item: item[1])
    best = results[0]
    print(f"\nBest batch size: {best[0]} (mean {best[1]:.3f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
