#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate accuracy and timing data.")
    parser.add_argument("dataset_path")
    parser.add_argument("shards", type=int)
    parser.add_argument("ul_reqs", type=int)
    parser.add_argument("container_name")
    parser.add_argument("report_name")
    parser.add_argument("hyperparams_json", nargs="?", default="-")
    args = parser.parse_args()

    if args.ul_reqs < 0:
        print("Error can't be below 1", file=sys.stderr)
        return 1

    report_path = f"{args.report_name}.csv"
    if not os.path.isfile(report_path):
        with open(report_path, "w") as handle:
            handle.write("nb_shards,nb_requests,accuracy,retraining_time,hyperparams_json\n")

    print("data.py")

    hyperparams_csv = args.hyperparams_json.replace('"', '""')
    times_dir = os.path.join("containers", args.container_name, "times")
    times_path = os.path.join(times_dir, "times")

    for r in range(args.ul_reqs + 1):
        acc_result = subprocess.run(
            [
                sys.executable,
                "aggregation.py",
                "--strategy",
                "uniform",
                "--container",
                args.container_name,
                "--shards",
                str(args.shards),
                "--dataset",
                args.dataset_path,
                "--label",
                str(r),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        acc = acc_result.stdout.strip()

        pattern = os.path.join(times_dir, f"shard-*:{r}.time")
        time_files = sorted(glob.glob(pattern))
        if not time_files:
            raise FileNotFoundError(f"No time files matched {pattern}")
        with open(times_path, "w") as out_handle:
            for time_file in time_files:
                with open(time_file, "r") as in_handle:
                    out_handle.write(in_handle.read())

        time_result = subprocess.run(
            [sys.executable, "time.py", "--container", args.container_name],
            check=True,
            capture_output=True,
            text=True,
        )
        time_value = time_result.stdout.strip().split(",")[0]

        with open(report_path, "a") as handle:
            handle.write(
                f"{args.shards},{r},{acc},{time_value},\"{hyperparams_csv}\"\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
