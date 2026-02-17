#!/usr/bin/env python3

from __future__ import annotations

import time
import argparse
import subprocess
import sys
import os
from tqdm import tqdm


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SISA predictions.")
    parser.add_argument("model")
    parser.add_argument("dataset_path")
    parser.add_argument("shards", type=int)
    parser.add_argument("ul_reqs", type=int)
    parser.add_argument("container_name")
    parser.add_argument("batch_size", type=int)
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Iterate shards/ul_reqs in reverse order.",
    )
    args = parser.parse_args()

    if args.ul_reqs < 0:
        print("Error can't be below 1", file=sys.stderr)
        return 1

    print("predict.py")

    total = args.shards * (args.ul_reqs + 1)
    progress = tqdm(
            total=total,
            desc="predict",
            bar_format="{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            dynamic_ncols=True,
            leave=False,
    )
    shard_range = (
        range(args.shards - 1, -1, -1) if args.reverse_order else range(args.shards)
    )
    ul_req_range = (
        range(args.ul_reqs, -1, -1)
        if args.reverse_order
        else range(args.ul_reqs + 1)
    )
    start = time.time()
    for shard in shard_range:
        for r in ul_req_range:
            output_path = "containers/{}/outputs/shard-{}:{}.npy".format(
                args.container_name, shard, r
            )
            if os.path.exists(output_path):
                continue
            subprocess.run(
                [
                    sys.executable,
                    "sisa.py",
                    "--model",
                    args.model,
                    "--test",
                    "--dataset",
                    args.dataset_path,
                    "--label",
                    str(r),
                    "--batch_size",
                    str(args.batch_size),
                    "--container",
                    args.container_name,
                    "--shard",
                    str(shard),
                ],
                check=True,
            )
            progress.update(1)
    time_taken = time.time()-start
    progress.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
