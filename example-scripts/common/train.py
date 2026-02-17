#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Train SISA shards.")
    parser.add_argument("model")
    parser.add_argument("dataset_path")
    parser.add_argument("shards", type=int)
    parser.add_argument("ul_reqs", type=int)
    parser.add_argument("container_name")
    parser.add_argument("slices", type=int)
    parser.add_argument("epochs", type=int)
    parser.add_argument("batch_size", type=int)
    parser.add_argument("learning_rate")
    parser.add_argument("optimizer")
    parser.add_argument("chkpt_interval", type=int)
    parser.add_argument("dropout_rate", nargs="?")
    args = parser.parse_args()

    if args.ul_reqs < 0:
        print("Error can't be below 1", file=sys.stderr)
        return 1

    print("train.py")

    for shard in range(args.shards):
        for r in range(args.ul_reqs + 1):
            print(
                f"shard: {shard + 1}/{args.shards}, "
                f"requests: {r + 1}/{args.ul_reqs + 1}"
            )
            cmd = [
                sys.executable,
                "sisa.py",
                "--model",
                args.model,
                "--train",
                "--slices",
                str(args.slices),
                "--dataset",
                args.dataset_path,
                "--label",
                str(r),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--learning_rate",
                str(args.learning_rate),
                "--optimizer",
                args.optimizer,
                "--chkpt_interval",
                str(args.chkpt_interval),
                "--container",
                args.container_name,
                "--shard",
                str(shard),
            ]
            if args.dropout_rate:
                cmd.extend(["--dropout_rate", str(args.dropout_rate)])
            subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
