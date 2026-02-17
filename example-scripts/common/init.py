#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Initialize containers and request distributions."
    )
    parser.add_argument("dataset_path")
    parser.add_argument("shards", type=int)
    parser.add_argument("ul_reqs", type=int)
    parser.add_argument("container_name")
    args = parser.parse_args()

    if args.ul_reqs < 0:
        print("Error can't be below 1", file=sys.stderr)
        return 1

    container_dir = os.path.join("containers", args.container_name)
    if not os.path.isdir(container_dir):
        os.makedirs(os.path.join(container_dir, "cache"), exist_ok=True)
        os.makedirs(os.path.join(container_dir, "times"), exist_ok=True)
        os.makedirs(os.path.join(container_dir, "outputs"), exist_ok=True)
        with open(os.path.join(container_dir, "times", "null.time"), "w") as handle:
            handle.write("0\n")

    print("init.py")

    subprocess.run(
        [
            sys.executable,
            "distribution.py",
            "--shards",
            str(args.shards),
            "--distribution",
            "uniform",
            "--container",
            args.container_name,
            "--dataset",
            args.dataset_path,
            "--label",
            "0",
        ],
        check=True,
    )

    if args.ul_reqs >= 1:
        for r in range(1, args.ul_reqs + 1):
            print(r)
            subprocess.run(
                [
                    sys.executable,
                    "distribution.py",
                    "--append_requests",
                    "1",
                    "--distribution",
                    "uniform",
                    "--container",
                    args.container_name,
                    "--dataset",
                    args.dataset_path,
                    "--label",
                    str(r),
                    "--prev_label",
                    str(r - 1),
                ],
                check=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
