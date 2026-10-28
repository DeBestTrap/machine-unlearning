#!/usr/bin/env python3
"""Reproduce the full-pool S3T validation protocol in a fresh study run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset_splits import load_manifest
from deletion_requests import load_workload
from s3t.config import Config
from studies.artifacts import execute_plan


PRESETS = {"cifar10": ("vit-cifar10.json", "cifar10-val2500-seed42.json", "cifar10-seed123-100x1.json"),
           "sst2": ("roberta-sst2.json", "glue-sst2-val3367-seed42.json", "glue-sst2-seed123-100x1.json")}


def build_plan(datasets, output, device):
    commands = []
    for dataset in datasets:
        preset, split_name, requests_name = PRESETS[dataset]
        config = ROOT / "configs/s3t" / preset
        split = load_manifest(ROOT / "splits" / split_name)
        requests = ROOT / "deletions" / requests_name
        workload = load_workload(requests, split)
        run = Path(output) / dataset
        overrides = {"shards": 2, "slices": 2, "budget": 2, "layers_per_slice": 6,
                     "epochs_per_slice": 1, "split_manifest": str(ROOT / "splits" / split_name),
                     "max_train_examples": None, "max_eval_examples": None}
        resolved = Config.from_dict({**Config.load(config).to_dict(), **overrides}).to_dict()
        train = [sys.executable, "-m", "s3t", "train", "--config", str(config),
                 "--output", str(run), "--device", device]
        for key, value in overrides.items():
            train.extend(["--set", f"{key}={json.dumps(value)}"])
        commands.append({"dataset": dataset, "argv": train, "resolved_config": resolved,
                         "split_sha256": split["manifest_sha256"], "workload_sha256": workload["workload_sha256"]})
        commands.append({"dataset": dataset, "argv": [sys.executable, "-m", "s3t", "replay", "--run", str(run),
                         "--requests", str(requests), "--device", device, "--on-exhaustion", "continue"]})
        commands.append({"dataset": dataset, "argv": [sys.executable, "-m", "s3t", "audit", "--run", str(run)]})
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(PRESETS), default=list(PRESETS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Tag must contain only letters, numbers, underscores, or hyphens")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("Repeated datasets would duplicate runs")
    output = STUDY / "runs" / args.tag
    if output.exists():
        parser.error("Validation run exists; choose a new tag")
    commands = build_plan(args.datasets, output, args.device)
    for item in commands:
        print(shlex.join(item["argv"]), flush=True)
    if args.dry_run:
        return 0
    return execute_plan(output, STUDY, commands, {"commands": commands, "cwd": str(ROOT)}, ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
