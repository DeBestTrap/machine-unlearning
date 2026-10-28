#!/usr/bin/env python3
"""Run one configuration per method/dataset using the shared comparison manifest.

Use the shared protocol for BlockUnlearn and the machine-unlearning baselines.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
STUDY_RELATIVE = STUDY.relative_to(ROOT)
sys.path.insert(0, str(ROOT))

from dataset_splits import load_manifest
from deletion_requests import load_workload
from studies.artifacts import create_sisa_records, execute_plan


def load_protocol(path, root):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported comparison manifest schema")
    if type(protocol.get("epochs")) is not int or protocol["epochs"] < 1:
        raise ValueError("Comparison epochs must be a positive integer")
    for name, dataset in protocol["datasets"].items():
        split = load_manifest(root / "splits" / dataset["split"])
        workload = load_workload(root / "deletions" / dataset["workload"], split)
        if split["manifest_sha256"] != dataset["split_sha256"]:
            raise ValueError(f"{name}: comparison split checksum differs")
        if workload["workload_sha256"] != dataset["workload_sha256"]:
            raise ValueError(f"{name}: comparison workload checksum differs")
        if (len(workload["batches"]) != protocol["steps"]
                or any(len(batch) != protocol["points_per_step"] for batch in workload["batches"])):
            raise ValueError(f"{name}: unexpected deletion count")
    return protocol


def build_plan(protocol, root, methods, datasets, tag, device, on_exhaustion="continue"):
    """Build every command before running anything; no filesystem writes."""
    block = (root / "run_sweep.py").exists()
    allowed = {"fcm", "fc"} if block else {"sisa", "s3t"}
    if set(methods) - allowed:
        raise ValueError(f"Methods for this repository: {', '.join(sorted(allowed))}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Tag must contain only letters, numbers, underscores, or hyphens")
    if not re.fullmatch(r"(?:cuda:)?[0-9]+", device):
        raise ValueError("Device must be a GPU index or cuda:<index>")
    gpu = device.removeprefix("cuda:")
    commands = []

    def add(method, dataset, argv, env=None):
        commands.append(dict(method=method, dataset=dataset, argv=argv, env=env or {}))

    for name in datasets:
        entry = protocol["datasets"][name]
        split = root / "splits" / entry["split"]
        requests = root / "deletions" / entry["workload"]
        for method in methods:
            label = f"{tag}_{method}_{name}"
            if block:
                settings = entry[method]
                config = root / settings["config"]
                if not config.is_file():
                    raise ValueError(f"Missing config: {config}")
                argv = [sys.executable, "run_sweep.py", str(config), "--device", f"cuda:{gpu}",
                        "--workdir", str(root / STUDY_RELATIVE / "runs" / tag / "sweeps" / method / name)]
                overrides = {
                    "experiment_name": f"Shared25_{label}",
                    "experiment_args.mode": "train_then_unlearn",
                    "experiment_args.seed": protocol["training_seed"],
                    "experiment_args.total_runs": 1,
                    "train_args.max_epochs": protocol["epochs"],
                    "val_args.dataset_size": 0.05,
                    "val_args.split_seed": 42,
                    "val_args.split_manifest": str(split),
                    "dataset_args.max_train_examples": None,
                    "dataset_args.max_val_examples": None,
                    "dataset_args.max_test_examples": None,
                    "unlearning_args.type": "random_datapoint",
                    "unlearning_args.request_manifest": str(requests),
                    "unlearning_args.num_sequential": protocol["steps"],
                    "unlearning_args.num_unlearned_points": protocol["points_per_step"],
                    "unlearning_args.use_completed": False,
                }
                for index, count in enumerate(settings.get("num_blocks", [])):
                    overrides[f"train_args.training_groups[{index}].num_blocks"] = count
                for key, value in overrides.items():
                    argv.extend(["--set", f"{key}={json.dumps(value)}"])
                add(method, name, argv)
            elif method == "sisa":
                settings = protocol["sisa"]
                metadata = root / entry["sisa_dataset"]
                info = json.loads(metadata.read_text())
                actual = load_manifest(metadata.parent / info["split_manifest"])
                if actual["manifest_sha256"] != entry["split_sha256"]:
                    raise ValueError(f"{name}: SISA datasetfile uses a different split")
                if (info["nb_train"] != len(actual["train_indices"])
                        or info["nb_val"] != len(actual["val_indices"])):
                    raise ValueError(f"{name}: SISA datasetfile pool sizes differ")
                records = root / STUDY_RELATIVE / "runs" / tag / f"sisa-{name}"
                if records.exists():
                    raise ValueError(f"SISA records exist; choose a new tag: {records}")
                argv = ["bash", "example-scripts/common/run_all.sh", entry["sisa_model"],
                        str(metadata), str(settings["shards"]), str(protocol["steps"]), str(records),
                        str(settings["slices"]), str(protocol["epochs"]), str(entry["batch_size"]),
                        str(entry["learning_rate"]), "adamw", "1", str(entry["batch_size"]),
                        str(root / STUDY_RELATIVE / "runs" / tag / "reports" / f"sisa-{name}")]
                add(method, name, argv, {"CUDA_VISIBLE_DEVICES": gpu,
                                        "DELETION_REQUEST_MANIFEST": str(requests)})
            else:
                output = root / STUDY_RELATIVE / "runs" / tag / f"s3t-{name}"
                if output.exists():
                    raise ValueError(f"S3T run exists; choose a new tag: {output}")
                config = root / entry["s3t_config"]
                if not config.is_file():
                    raise ValueError(f"Missing config: {config}")
                argv = [sys.executable, "-m", "s3t", "train", "--config", str(config),
                        "--output", str(output), "--device", f"cuda:{gpu}"]
                overrides = {"split_manifest": str(split), "max_train_examples": None,
                             "max_eval_examples": None, "seed": protocol["training_seed"],
                             "epochs_per_slice": protocol["epochs"]}
                for key, value in overrides.items():
                    argv.extend(["--set", f"{key}={json.dumps(value)}"])
                add(method, name, argv)
                add(method, name, [sys.executable, "-m", "s3t", "replay", "--run", str(output),
                                   "--requests", str(requests), "--device", f"cuda:{gpu}",
                                   "--on-exhaustion", on_exhaustion])
                add(method, name, [sys.executable, "-m", "s3t", "audit", "--run", str(output)])
    return commands


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=STUDY / "configs/quick25.json")
    parser.add_argument("--datasets", nargs="+", choices=("cifar10", "cifar100", "sst2"),
                        default=["cifar10", "cifar100", "sst2"])
    parser.add_argument("--methods", nargs="+", choices=("fcm", "fc", "sisa", "s3t"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("quick25_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--s3t-on-exhaustion", choices=("stop", "continue", "retrain"), default="continue")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands without writing or training")
    args = parser.parse_args(argv)
    try:
        protocol = load_protocol(args.manifest.resolve(), ROOT)
        methods = args.methods or (["fcm", "fc"] if (ROOT / "run_sweep.py").exists() else ["sisa", "s3t"])
        if len(set(methods)) != len(methods) or len(set(args.datasets)) != len(args.datasets):
            raise ValueError("Repeated methods or datasets would duplicate runs")
        commands = build_plan(protocol, ROOT, methods, args.datasets, args.tag, args.device,
                              args.s3t_on_exhaustion)
        output = STUDY / "runs" / args.tag
        if (output / "plan.json").exists():
            raise ValueError(f"Comparison already launched; choose a new --tag: {args.tag}")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Tag: {args.tag}; {protocol['steps']} shared deletions per dataset", flush=True)
    for item in commands:
        env = " ".join(f"{key}={shlex.quote(value)}" for key, value in item["env"].items())
        print(f"[{item['method']}/{item['dataset']}] {env} {shlex.join(item['argv'])}".strip(), flush=True)
    if args.dry_run:
        return 0
    def prepare():
        (output / "reports").mkdir(exist_ok=True)
        for name in args.datasets:
            if "sisa" in methods:
                create_sisa_records(output / f"sisa-{name}", ROOT)
    settings = dict(protocol=protocol, cwd=str(ROOT), commands=commands,
                    manifest_path=str(args.manifest.resolve()))
    return execute_plan(output, STUDY, commands, settings, ROOT, prepare=prepare)


if __name__ == "__main__":
    raise SystemExit(main())
