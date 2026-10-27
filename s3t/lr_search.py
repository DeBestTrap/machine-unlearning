"""Durable, sequential learning-rate screening on one representative shard.

Training preserves the full five-shard partition. Only shard 0 / sequence 0 is
trained; rankings use its final-prefix validation accuracy, never test accuracy.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter, time

from .artifacts import _atomic, atomic_json, lock, read_json
from .config import Config, ROOT
from .partition import Partition, digest


GRIDS = {"cifar10": ("vit-cifar10.json", [0.0002, 0.001, 0.002]),
         "cifar100": ("vit-cifar100.json", [0.0002, 0.001, 0.002]),
         "sst2": ("roberta-sst2.json", [0.000005, 0.00001, 0.00002])}


def make_plan(datasets, device):
    trials = []
    for dataset in datasets:
        preset, rates = GRIDS[dataset]
        base = Config.load(ROOT / "configs/s3t" / preset).to_dict()
        base.update(budget=1, sequence_strategy="bms", head_policy="first_slice",
                    deletion_probabilities=[1.0 / base["slices"]] * base["slices"])
        for rate in rates:
            config = Config.from_dict(dict(base, learning_rate=rate))
            trials.append({"name": f"{dataset}-lr-{rate:.0e}", "dataset": dataset,
                           "learning_rate": rate, "config": config.to_dict()})
    return {"schema_version": 1, "device": device, "shard": 0, "sequence": 0,
            "selection_metric": "final_prefix_validation_accuracy", "trials": trials}


def run_trial(config_path, directory, device):
    from .engine import Run
    started = perf_counter()
    config = Config.load(config_path)
    run = Run.create(directory, config)
    initialized = perf_counter()
    run.train(device, shard=0, sequence=0)
    checkpoint = run.checkpoint(0, 0, 0)
    if not checkpoint["complete"]:
        raise RuntimeError("Pilot member did not finish")
    partition = run.partition(0)
    result = {"status": "completed", "dataset": config.dataset, "learning_rate": config.learning_rate,
              "head_policy": config.head_policy, "sequence_strategy": config.sequence_strategy,
              "deletion_probabilities": config.deletion_probabilities, "score_horizon": config.score_horizon,
              "shards": config.shards, "slices": config.slices, "budget": config.budget,
              "trained_shard": 0, "trained_sequence": 0, "slice_order": list(partition.sequences[0]),
              "shard_examples": len(partition.training_ids(0, 0, config.slices)),
              "validation_examples": len(run.data.val), "epochs_per_slice": config.epochs_per_slice,
              "validation_accuracy": checkpoint["validation_scores"][-1],
              "prefix_validation_accuracies": checkpoint["validation_scores"],
              "training_seconds": checkpoint["train_seconds"],
              "validation_seconds": checkpoint["validation_seconds"],
              "initialization_seconds": initialized - started, "worker_wall_seconds": perf_counter() - started,
              "completed_epochs": len(checkpoint["history"]), "device": str(device),
              "config_sha256": digest(config.to_dict()), "run": str(run.directory),
              "note": "One-shard/one-sequence pilot; full ensemble and test pool were not evaluated."}
    atomic_json(Path(directory) / "pilot_result.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _reference_unit_seconds(dataset):
    # An initial estimate from earlier runs of the same local backbones/batches.
    # CIFAR-100 initially uses the same ViT/input-size timing reference as CIFAR-10.
    name = {"cifar10": "vit-cifar10-full-pool", "cifar100": "vit-cifar10-full-pool",
            "sst2": "roberta-sst2-full-pool"}.get(dataset)
    if name is None:
        return None
    root = ROOT / "runs/s3t-validation" / name
    if not (root / "config.json").exists():
        return None
    config = read_json(root / "config.json")
    units = []
    for path in (root / "generations/0000/members").glob("*.json"):
        for epoch in read_json(path)["history"]:
            work = epoch["examples"] * (1.0 + (epoch["stage"] + 1) / config["slices"])
            units.append(epoch["seconds"] / work)
    return statistics.median(units) if units else None


def summarize(directory, plan, state):
    directory = Path(directory)
    rows = []
    completed_times = {}
    for trial in plan["trials"]:
        output = directory / trial["name"]
        result_path = output / "pilot_result.json"
        if result_path.exists():
            result = read_json(result_path)
            if result["config_sha256"] != digest(trial["config"]):
                raise ValueError("Pilot result/configuration mismatch")
            completed_times.setdefault(trial["dataset"], []).append(result["worker_wall_seconds"])
    for trial in plan["trials"]:
        name, config = trial["name"], trial["config"]
        output = directory / name
        progress = output / "generations/0000/members/shard-0000-sequence-0000.json"
        checkpoint = read_json(progress) if progress.exists() else None
        result_path = output / "pilot_result.json"
        result = read_json(result_path) if result_path.exists() else {}
        trial_state = state.get("trials", {}).get(name, {})
        history = checkpoint["history"] if checkpoint else []
        row = {"name": name, "dataset": trial["dataset"], "learning_rate": trial["learning_rate"],
               "status": "completed" if result else trial_state.get("status", "queued"),
               "completed_epochs": len(history), "total_epochs": config["slices"] * config["epochs_per_slice"],
               "validation_accuracy": result.get("validation_accuracy"),
               "latest_prefix_validation_accuracy": checkpoint["validation_scores"][-1]
                   if checkpoint and checkpoint["validation_scores"] else None,
               "wall_seconds": result.get("worker_wall_seconds", trial_state.get("wall_seconds")),
               "training_seconds": checkpoint["train_seconds"] if checkpoint else 0.0,
               "validation_seconds": checkpoint["validation_seconds"] if checkpoint else 0.0,
               "estimated_remaining_seconds": 0.0 if result else None}
        if row["status"] == "running":
            row["wall_seconds"] = time() - trial_state["started_at"]
        if not result and row["status"] != "failed":
            partition_path = output / "generations/0000/partition.json"
            if partition_path.exists():
                partition = Partition.from_dict(read_json(partition_path))
                sizes = [len(partition.training_ids(0, 0, prefix)) for prefix in range(1, config["slices"] + 1)]
            else:
                full_size = {"cifar10": 47500, "cifar100": 47500, "sst2": 63982, "synthetic": 78}[trial["dataset"]]
                shard_size = (full_size + config["shards"] - 1) // config["shards"]
                sizes = [shard_size * prefix / config["slices"] for prefix in range(1, config["slices"] + 1)]
            units = [epoch["seconds"] / (epoch["examples"] * (1 + (epoch["stage"] + 1) / config["slices"]))
                     for epoch in history]
            unit = statistics.median(units) if units else _reference_unit_seconds(trial["dataset"])
            if unit is not None:
                done = [sum(epoch["stage"] == stage for epoch in history) for stage in range(config["slices"])]
                remaining = sum(size * (1 + (stage + 1) / config["slices"])
                                * (config["epochs_per_slice"] - done[stage]) for stage, size in enumerate(sizes))
                # Include an approximate validation/IO allowance; initialization is small.
                row["estimated_remaining_seconds"] = remaining * unit * 1.1
            elif trial["dataset"] in completed_times:
                row["estimated_remaining_seconds"] = statistics.median(completed_times[trial["dataset"]])
        rows.append(row)
    remaining = [row["estimated_remaining_seconds"] for row in rows if row["status"] not in ("completed", "failed")]
    finished = all(row["status"] in ("completed", "failed") for row in rows)
    status = {"schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(),
              "status": ("failed" if any(row["status"] == "failed" for row in rows) else "completed") if finished else "running",
              "pid": state.get("pid"), "device": plan["device"], "selection_metric": plan["selection_metric"],
              "estimated_remaining_seconds": sum(remaining) if all(value is not None for value in remaining) else None,
              "timing_note": "ETA is approximate; extrapolates epoch time by prefix size and transformer depth.",
              "trials": rows}
    atomic_json(directory / "status.json", status)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic(directory / "results.csv", lambda stream: stream.write(buffer.getvalue().encode()))
    lines = ["# S3T learning-rate screening", "", f"Sequential runs on {plan['device']}. BMS uses uniform deletion priors; "
             "the classifier trains on the first slice and then stays frozen.", "",
             "Each trial trains shard 0 / sequence 0 of a five-shard partition. Rankings use final-prefix "
             "validation accuracy. These are single-member pilots, not full-ensemble results.", "",
             "| Dataset | LR | Status | Epochs | Final validation accuracy | Wall minutes | Estimated minutes left |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        score = "—" if row["validation_accuracy"] is None else f"{100 * row['validation_accuracy']:.2f}%"
        elapsed = "—" if row["wall_seconds"] is None else f"{row['wall_seconds'] / 60:.1f}"
        eta = "—" if row["estimated_remaining_seconds"] is None else f"{row['estimated_remaining_seconds'] / 60:.1f}"
        lines.append(f"| {row['dataset']} | {row['learning_rate']:g} | {row['status']} | "
                     f"{row['completed_epochs']}/{row['total_epochs']} | {score} | {elapsed} | {eta} |")
    lines.extend(["", status["timing_note"], "", "Completed trials contain `pilot_result.json`; "
                  "`logs/TRIAL_NAME.log` records per-epoch timing and progress.", ""])
    _atomic(directory / "report.md", lambda stream: stream.write("\n".join(lines).encode()))
    return status


def run_search(directory, plan):
    import os
    directory = Path(directory).expanduser().resolve()
    with lock(directory / ".sweep.lock"):
        plan_path = directory / "plan.json"
        if plan_path.exists() and read_json(plan_path) != plan:
            raise ValueError("Existing search has a different plan")
        atomic_json(plan_path, plan)
        state_path = directory / "queue.json"
        state = read_json(state_path) if state_path.exists() else {"trials": {}}
        state["pid"] = os.getpid()
        for trial in plan["trials"]:
            name = trial["name"]
            output = directory / name
            config_path = directory / "configs" / f"{name}.json"
            atomic_json(config_path, trial["config"])
            if (output / "pilot_result.json").exists():
                continue
            output.mkdir(parents=True, exist_ok=True)
            trial_state = {"status": "running", "started_at": time()}
            state["trials"][name] = trial_state
            atomic_json(state_path, state)
            summarize(directory, plan, state)
            command = [sys.executable, "-u", "-m", "s3t.lr_search", "worker", "--config", str(config_path),
                       "--output", str(output), "--device", plan["device"]]
            print(f"Starting {name}: first_slice, BMS, device={plan['device']}", flush=True)
            log_path = directory / "logs" / f"{name}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", buffering=1) as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                trial_state["worker_pid"] = process.pid
                atomic_json(state_path, state)
                last_progress = -1
                while True:
                    try:
                        code = process.wait(timeout=10)
                        break
                    except subprocess.TimeoutExpired:
                        progress = summarize(directory, plan, state)
                        active = next(row for row in progress["trials"] if row["name"] == name)
                        if active["completed_epochs"] != last_progress:
                            eta = progress["estimated_remaining_seconds"]
                            estimate = "unknown" if eta is None else f"{eta / 60:.1f} min"
                            print(f"{name}: {active['completed_epochs']}/{active['total_epochs']} epochs, "
                                  f"wall={active['wall_seconds'] / 60:.1f} min, "
                                  f"estimated search remaining={estimate}", flush=True)
                            last_progress = active["completed_epochs"]
            trial_state.update(status="completed" if code == 0 else "failed", returncode=code,
                               wall_seconds=time() - trial_state["started_at"])
            atomic_json(state_path, state)
            summarize(directory, plan, state)
            print(f"Finished {name}: {trial_state['status']}, wall={trial_state['wall_seconds'] / 60:.1f} min", flush=True)
        return summarize(directory, plan, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--datasets", nargs="+", choices=tuple(GRIDS), default=["cifar10", "sst2"])
    worker = sub.add_parser("worker")
    worker.add_argument("--config", required=True)
    worker.add_argument("--output", required=True)
    worker.add_argument("--device", required=True)
    status = sub.add_parser("status")
    status.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_search(args.output, make_plan(args.datasets, args.device))
    elif args.command == "worker":
        result = run_trial(args.config, args.output, args.device)
    else:
        directory = Path(args.output)
        result = summarize(directory, read_json(directory / "plan.json"), read_json(directory / "queue.json"))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
