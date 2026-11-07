#!/usr/bin/env python3
"""Fixed-20-shard development search over complete S3T ensembles."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dataset_splits import load_manifest
from reporting.core import atomic_text, digest, write_json
from s3t.config import Config
from s3t.data import development_split, load_data
from studies.artifacts import capture_run, verify_run_source


DEPTH = {"vit_b_16": 12, "resnet50": 4, "densenet121": 4, "roberta": 12}


def now():
    return datetime.now(timezone.utc).isoformat()


def signed_write(path, value):
    write_json(path, {**value, "sha256": digest(value)})


def signed_read(path):
    value = json.loads(Path(path).read_text())
    checksum = value.pop("sha256")
    if digest(value) != checksum:
        raise ValueError(f"Signed study record changed: {path}")
    return value


def validate_settings(settings):
    if settings.get("shard_bands") != [[20]]:
        raise ValueError("Fixed-20 S3T study requires exactly 20 shards")
    supported = {"vit_b_16": "cifar100", "resnet50": "cifar10",
                 "densenet121": "cifar100", "roberta": "sst2"}
    if settings.get("dataset") != supported.get(settings.get("model")):
        raise ValueError("Unsupported fixed-20 S3T model/dataset pair")
    bands = settings.get("shard_bands")
    if (not bands or any(not band for band in bands)
            or len({s for band in bands for s in band}) != sum(map(len, bands))
            or any(type(s) is not int or s < 2 for band in bands for s in band)):
        raise ValueError("Shard bands must be nonempty and disjoint")
    for name in ("slices", "sequence_budgets", "epochs_per_slice", "learning_rates"):
        values = settings.get(name)
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")
    depth = DEPTH[settings["model"]]
    if any(type(r) is not int or r < 2 or depth % r for r in settings["slices"]):
        raise ValueError("Slices must divide the selected model's depth")
    if any(type(b) is not int or b < 1 for b in settings["sequence_budgets"]):
        raise ValueError("Sequence budgets must be positive integers")
    if any(type(e) is not int or e < 1 for e in settings["epochs_per_slice"]):
        raise ValueError("Epoch counts must be positive integers")
    if any(not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0
           for lr in settings["learning_rates"]):
        raise ValueError("Learning rates must be finite and positive")
    for name in ("screening_trials", "finalists_per_band", "screen_seed", "search_seed",
                 "batch_size", "rank", "num_workers", "minimum_free_gb", "max_members"):
        if type(settings.get(name)) is not int or settings[name] < (0 if name == "num_workers" else 1):
            raise ValueError(f"{name} must be a positive integer")
    if (settings["screening_trials"] % len(bands)
            or settings["screening_trials"] // len(bands) < settings["finalists_per_band"]
            or not settings.get("confirmation_seeds")
            or any(type(seed) is not int or seed < 1 for seed in settings["confirmation_seeds"])
            or len(set(settings["confirmation_seeds"])) != len(settings["confirmation_seeds"])):
        raise ValueError("Trial count or confirmation seeds are invalid")
    if not math.isfinite(settings["max_training_work_ratio"]) or settings["max_training_work_ratio"] <= 0:
        raise ValueError("Training-work limit must be positive")
    cap = settings.get("maximum_gpu_power_watts")
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("Invalid GPU power limit")
    if (settings.get("head_policy") not in ("first_slice", "snapshot")
            or settings.get("sequence_strategy") not in ("cyclic", "random")
            or settings.get("lora_dropout") is None):
        raise ValueError("Unsupported S3T policy")


def candidate_plan(settings):
    validate_settings(settings)
    rng = random.Random(settings["search_seed"])
    depth = DEPTH[settings["model"]]
    per_band = settings["screening_trials"] // len(settings["shard_bands"])
    results = []
    for band_index, shards in enumerate(settings["shard_bands"]):
        pool = []
        for s, r, b, e, lr in product(shards, settings["slices"], settings["sequence_budgets"],
                                      settings["epochs_per_slice"], settings["learning_rates"]):
            ratio = b * e * (r + 1) / 2
            if (b <= r and s * b <= settings["max_members"]
                    and ratio <= settings["max_training_work_ratio"]):
                pool.append(dict(shards=s, slices=r, budget=b, layers_per_slice=depth // r,
                                 epochs_per_slice=e, learning_rate=lr, member_count=s * b,
                                 initial_training_work_ratio=ratio, shard_band=band_index))
        rng.shuffle(pool)
        if len(pool) < per_band:
            raise ValueError(f"Shard band {band_index} has fewer than {per_band} feasible configurations")
        counts = defaultdict(Counter)
        for _ in range(per_band):
            # Greedy level coverage keeps this finite search from spending all
            # its trials on the same LR, epoch count, or sequence budget.
            best = min(range(len(pool)), key=lambda i: (
                sum(counts[key][pool[i][key]] for key in
                    ("shards", "slices", "budget", "epochs_per_slice", "learning_rate")), i))
            chosen = pool.pop(best)
            for key in ("shards", "slices", "budget", "epochs_per_slice", "learning_rate"):
                counts[key][chosen[key]] += 1
            results.append(chosen)
    rng.shuffle(results)
    for index, row in enumerate(results):
        row["id"] = f"candidate-{index:03d}"
    return results


def config_for(run, settings, candidate, pool, seed):
    values = dict(dataset=settings["dataset"], model=settings["model"], pretrained=True,
                  data_root=str(Path(settings["data_root"]).resolve()),
                  split_manifest=str(run / "inputs/parent.json"),
                  development_manifest=str(run / "inputs/development.json"),
                  development_pool=pool, shards=candidate["shards"], slices=candidate["slices"],
                  budget=candidate["budget"], layers_per_slice=candidate["layers_per_slice"],
                  epochs_per_slice=candidate["epochs_per_slice"], learning_rate=candidate["learning_rate"],
                  rank=settings["rank"], alpha=settings["alpha"],
                  lora_dropout=settings["lora_dropout"], head_policy=settings["head_policy"],
                  sequence_strategy=settings["sequence_strategy"], batch_size=settings["batch_size"],
                  eval_batch_size=settings["batch_size"], num_workers=settings["num_workers"],
                  seed=seed, partition_seed=42, sequence_seed=42, selection="longest_prefix",
                  aggregation="logits", precision="fp32", max_train_examples=None, max_eval_examples=None)
    return Config.from_dict(values)


def run_path(run, candidate, pool, seed):
    return run / ("search" if pool == "scoring" else "confirmation") / f"{candidate['id']}-seed{seed}"


def read_plan(run):
    if not (run / "plan.json").is_file():
        raise ValueError(f"S3T run has no completed preparation plan: {run}; rerun with 'run --config ... --name ...'")
    verify_run_source(run)
    plan = signed_read(run / "plan.json")
    settings = plan["settings"]
    validate_settings(settings)
    parent = load_manifest(run / "inputs/parent.json")
    dataset_id = "glue/sst2" if settings["dataset"] == "sst2" else settings["dataset"]
    if (parent["manifest_sha256"] != plan["parent_manifest_sha256"]
            or parent["dataset_id"] != dataset_id
            or len(parent["train_indices"]) + len(parent["val_indices"]) != parent["source_size"]):
        raise ValueError("Parent source-index manifest changed")
    probe = config_for(run, settings, plan["candidates"][0], "scoring", settings["screen_seed"])
    development = development_split(probe, parent)
    if development["development_manifest_sha256"] != plan["development_manifest_sha256"]:
        raise ValueError("Development source-index manifest changed")
    if candidate_plan(settings) != plan["candidates"]:
        raise ValueError("Candidate plan changed")
    return plan


def prepare(config_path, name, data_root):
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = STUDY / config_path
    settings = json.loads(config_path.read_text())
    settings["data_root"] = str(Path(data_root).expanduser().resolve())
    candidates = candidate_plan(settings)
    run = STUDY / "runs" / name
    if run.exists():
        raise ValueError(f"Run already exists: {run}")
    parent_path = STUDY / settings["parent_manifest"]
    development_path = STUDY / settings["development_manifest"]
    dataset_id = "glue/sst2" if settings["dataset"] == "sst2" else settings["dataset"]
    parent = load_manifest(parent_path, dataset_id=dataset_id)
    provisional = config_for(STUDY / "runs" / name, settings, candidates[0], "scoring", settings["screen_seed"])
    # Validate the original manifests before creating the run directory.
    dev_values = json.loads(development_path.read_text())
    if dev_values.get("parent_manifest_sha256") != parent["manifest_sha256"]:
        raise ValueError("Development manifest refers to another parent")
    capture_run(run, STUDY, settings, ROOT)
    (run / "inputs").mkdir()
    (run / "inputs/parent.json").write_bytes(parent_path.read_bytes())
    (run / "inputs/development.json").write_bytes(development_path.read_bytes())
    effective = development_split(provisional, parent)
    # Preparation opens the training source once to verify its fingerprint.
    data = load_data(provisional)
    if (data.split != effective or len(data.train) != len(effective["train_indices"])
            or len(data.val) != len(effective["val_indices"]) or data.test is not None):
        raise ValueError("Prepared development data differs from the fixed manifest")
    signed_write(run / "plan.json", dict(schema_version=1, purpose="s3t_dev10_allocation_search",
        config_source=str(config_path.resolve()), settings=settings, candidates=candidates,
        parent_manifest_sha256=parent["manifest_sha256"],
        development_manifest_sha256=dev_values["manifest_sha256"], prepared_at=now()))
    write_json(run / "progress.json", dict(status="prepared", phase="screening",
        search_completed=0, search_planned=len(candidates), confirmation_completed=0,
        confirmation_planned=None, active={}, errors=[]))
    print(f"Prepared {len(candidates)} complete-ensemble S3T candidates: {run}; no training started", flush=True)
    return run


def archive_incomplete_preparation(run):
    """Preserve an interrupted preparation before retrying it from scratch."""
    if not run.exists():
        return None
    if run.is_symlink() or not run.is_dir() or (run / "plan.json").exists():
        raise ValueError(f"Cannot archive a prepared or non-directory S3T run: {run}")
    allowed = {"launch_source", "study_run.json", "inputs", "progress.json"}
    unexpected = {path.name for path in run.iterdir()} - allowed
    if unexpected:
        raise ValueError(f"S3T run has no plan but contains other files: {run}: {sorted(unexpected)}")
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run.with_name(f"{run.name}.incomplete-{suffix}")
    if archive.exists():
        raise ValueError(f"Incomplete-preparation archive already exists: {archive}")
    run.rename(archive)
    print(f"Preserved interrupted preparation at {archive}", flush=True)
    return archive


def result(run, candidate, pool, seed):
    path = run_path(run, candidate, pool, seed) / "result.json"
    if not path.exists():
        return None
    value = signed_read(path)
    if (value["candidate"] != candidate or value["pool"] != pool or value["seed"] != seed
            or value["plan_sha256"] != digest(signed_read(run / "plan.json"))):
        raise ValueError("Completed S3T trial identity changed")
    return value


def discard_models(directory):
    native = directory / "native"
    paths = [native / "backbone.pt", *sorted((native / "generations/0000/members").glob("*.pt"))]
    receipt = directory / "checkpoint_retention.json"
    if receipt.exists():
        prior = signed_read(receipt)
        if prior["status"] == "discarded":
            if any(path.exists() for path in paths):
                raise ValueError("Discarded models reappeared")
            return
        if prior["status"] != "discarding":
            raise ValueError("Unknown model retention state")
    else:
        if not (native / "backbone.pt").exists():
            raise ValueError("Missing backbone without a retention receipt")
        signed_write(receipt, dict(status="discarding", policy="scores_only", native=str(native)))
    for path in paths:
        if not path.is_relative_to(native) or path.is_symlink():
            raise ValueError("Refusing to remove a model outside the native S3T trial")
        path.unlink(missing_ok=True)
    signed_write(receipt, dict(status="discarded", policy="scores_only", native=str(native),
                               completed_at=now()))


def run_trial(run, candidate, pool, seed, device):
    directory = run_path(run, candidate, pool, seed)
    existing = result(run, candidate, pool, seed)
    if existing:
        discard_models(directory)
        return existing
    plan = read_plan(run)
    if pool == "confirmation":
        frozen = signed_read(run / "finalists.json")
        if candidate["id"] not in [identity for group in frozen["bands"] for identity in group["candidate_ids"]]:
            raise ValueError("Candidate was not frozen for confirmation")
    config = config_for(run, plan["settings"], candidate, pool, seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists() and json.loads(path.read_text()) != config.to_dict():
        raise ValueError("Trial configuration changed on resume")
    write_json(path, config.to_dict())
    from s3t.engine import Run
    native = Run.create(directory / "native", config)
    native.train(device)
    scored = native.evaluate(device, split="val")
    if (scored["status"] != "ready" or scored["split"] != "val"
            or scored["examples"] != len(native.data.val)
            or scored["split_manifest_sha256"] != native.split["manifest_sha256"]
            or scored["capped_training"] or scored["capped_evaluation"]
            or scored["accuracy"] is None or not math.isfinite(scored["accuracy"])):
        raise ValueError("S3T trial did not produce a finite development score")
    value = dict(candidate=candidate, pool=pool, seed=seed, status="completed",
                 accuracy=scored["accuracy"], train_examples=len(native.data.train),
                 evaluation_examples=scored["examples"],
                 initial_example_presentations=scored["generation_training_examples"],
                 normalized_initial_work=scored["generation_training_examples"] / len(native.data.train),
                 generation_training_seconds=scored["generation_training_seconds"],
                 generation_validation_seconds=scored["generation_validation_seconds"],
                 evaluation_seconds=scored["evaluation_seconds"],
                 backbone_bytes=scored["backbone_bytes"],
                 checkpoint_bytes=scored["generation_checkpoint_bytes"],
                 split_manifest_sha256=native.split["manifest_sha256"],
                 plan_sha256=digest(plan), completed_at=now())
    signed_write(directory / "result.json", value)
    del native
    discard_models(directory)
    return value


def screening_digest(rows):
    return digest(sorted((row["candidate"]["id"], row["accuracy"], row["plan_sha256"])
                         for row in rows))


def freeze_finalists(plan, rows):
    if len(rows) != len(plan["candidates"]):
        raise ValueError("Cannot freeze finalists before every candidate is scored")
    bands = []
    for index in range(len(plan["settings"]["shard_bands"])):
        choices = sorted((row for row in rows if row["candidate"]["shard_band"] == index),
                         key=lambda row: (-row["accuracy"], row["candidate"]["initial_training_work_ratio"],
                                          row["candidate"]["id"]))
        bands.append(dict(shard_band=index,
                          candidate_ids=[row["candidate"]["id"] for row in
                                         choices[:plan["settings"]["finalists_per_band"]]]))
    return dict(source_screening_sha256=screening_digest(rows), bands=bands, frozen_at=now())


def write_csv(path, rows):
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def flat(row):
    return {**row["candidate"], **{key: value for key, value in row.items() if key != "candidate"}}


class Runner:
    def __init__(self, run, plan, devices):
        self.run, self.plan, self.devices = run, plan, devices
        self.active, self.errors = {}, []
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.phase = "screening"

    def progress(self, status="running", confirmation_planned=None):
        screen = [result(self.run, c, "scoring", self.plan["settings"]["screen_seed"])
                  for c in self.plan["candidates"]]
        confirmation = []
        if (self.run / "finalists.json").exists():
            frozen = signed_read(self.run / "finalists.json")
            by_id = {c["id"]: c for c in self.plan["candidates"]}
            for band in frozen["bands"]:
                for identity in band["candidate_ids"]:
                    for seed in self.plan["settings"]["confirmation_seeds"]:
                        confirmation.append(result(self.run, by_id[identity], "confirmation", seed))
        write_json(self.run / "progress.json", dict(status=status, phase=self.phase,
            coordinator_pid=os.getpid(), devices=self.devices, active=dict(self.active), errors=list(self.errors),
            search_completed=sum(x is not None for x in screen), search_planned=len(screen),
            confirmation_completed=sum(x is not None for x in confirmation),
            confirmation_planned=confirmation_planned, updated_at=now()))

    def reports(self):
        scoring = [result(self.run, c, "scoring", self.plan["settings"]["screen_seed"])
                   for c in self.plan["candidates"]]
        write_csv(self.run / "search_results.csv", [flat(x) for x in scoring if x])
        if not (self.run / "finalists.json").exists():
            return
        frozen = signed_read(self.run / "finalists.json")
        by_id = {c["id"]: c for c in self.plan["candidates"]}
        rows = []
        for band in frozen["bands"]:
            for identity in band["candidate_ids"]:
                for seed in self.plan["settings"]["confirmation_seeds"]:
                    value = result(self.run, by_id[identity], "confirmation", seed)
                    if value:
                        rows.append(flat(value))
        write_csv(self.run / "confirmation_results.csv", rows)
        groups = defaultdict(list)
        for row in rows:
            groups[row["id"]].append(row)
        selected = []
        for band in frozen["bands"]:
            options = []
            for identity in band["candidate_ids"]:
                seeds = groups[identity]
                if len(seeds) == len(self.plan["settings"]["confirmation_seeds"]):
                    scores = [row["accuracy"] for row in seeds]
                    options.append(dict(**by_id[identity],
                        mean_accuracy=statistics.mean(scores),
                        std_accuracy=statistics.stdev(scores) if len(scores) > 1 else 0.0,
                        seeds=len(scores)))
            if options:
                selected.append(min(options, key=lambda row: (-row["mean_accuracy"],
                                                          row["initial_training_work_ratio"], row["id"])))
        write_csv(self.run / "selected_by_band.csv", selected)

    def phase_run(self, phase, jobs):
        self.phase = phase
        iterator = iter(jobs)
        planned = len(jobs) if phase == "confirmation" else None
        self.progress(confirmation_planned=planned)

        def worker(device):
            while not self.stop.is_set():
                with self.lock:
                    if (self.run / "STOP").exists():
                        self.stop.set()
                        return
                    item = next(iterator, None)
                    if item is None:
                        return
                    candidate, pool, seed = item
                    self.active[device] = candidate["id"] + ":" + pool + ":" + str(seed)
                    self.progress(confirmation_planned=planned)
                try:
                    if result(self.run, candidate, pool, seed) is None:
                        if shutil.disk_usage(self.run).free < self.plan["settings"]["minimum_free_gb"] * 1024**3:
                            raise RuntimeError("Free-space guard stopped before next S3T trial")
                    command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker",
                               "--run", str(self.run), "--candidate", candidate["id"],
                               "--pool", pool, "--seed", str(seed), "--device", device]
                    directory = run_path(self.run, candidate, pool, seed)
                    directory.mkdir(parents=True, exist_ok=True)
                    environment = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                    with (directory / "worker.log").open("a") as log:
                        process = subprocess.run(command, cwd=ROOT, env=environment,
                                                 stdout=log, stderr=subprocess.STDOUT)
                    if process.returncode or result(self.run, candidate, pool, seed) is None:
                        raise RuntimeError(f"S3T trial failed: {candidate['id']} {pool} seed{seed}; see worker.log")
                    with self.lock:
                        self.active.pop(device, None)
                        self.reports()
                        self.progress(confirmation_planned=planned)
                        print(f"Completed {candidate['id']} {pool} seed={seed}", flush=True)
                except Exception as error:
                    with self.lock:
                        self.errors.append(f"{device}: {type(error).__name__}: {error}")
                        self.active.pop(device, None)
                        self.stop.set()
                        self.progress("failed", confirmation_planned=planned)
                    return

        with ThreadPoolExecutor(max_workers=len(self.devices)) as pool:
            for future in [pool.submit(worker, device) for device in self.devices]:
                future.result()
        self.reports()
        if self.errors:
            raise RuntimeError(self.errors[0])
        if self.stop.is_set() or (self.run / "STOP").exists():
            self.progress("paused", confirmation_planned=planned)
            raise InterruptedError("Paused after active S3T trials")


def verify_resources(settings):
    cap = settings["maximum_gpu_power_watts"]
    if cap is None:
        return
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=power.limit",
                                      "--format=csv,noheader,nounits"], text=True)
    readings = [float(line) for line in output.splitlines() if line.strip()]
    if not readings or any(not math.isfinite(value) or value > cap for value in readings):
        raise RuntimeError(f"Configured GPU power limits must be at most {cap} W")


def execute(run, devices):
    run = Path(run).resolve()
    plan = read_plan(run)
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("Select distinct execution devices")
    verify_resources(plan["settings"])
    runner = Runner(run, plan, devices)
    with (run / ".search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            jobs = [(c, "scoring", plan["settings"]["screen_seed"]) for c in plan["candidates"]]
            runner.phase_run("screening", jobs)
            scoring = [result(run, c, "scoring", plan["settings"]["screen_seed"])
                       for c in plan["candidates"]]
            frozen_path = run / "finalists.json"
            if not frozen_path.exists():
                signed_write(frozen_path, freeze_finalists(plan, scoring))
            frozen = signed_read(frozen_path)
            if frozen["source_screening_sha256"] != screening_digest(scoring):
                raise ValueError("Frozen finalists disagree with screening scores")
            by_id = {c["id"]: c for c in plan["candidates"]}
            confirmation = [(by_id[identity], "confirmation", seed)
                            for band in frozen["bands"] for identity in band["candidate_ids"]
                            for seed in plan["settings"]["confirmation_seeds"]]
            runner.phase_run("confirmation", confirmation)
            runner.progress("completed", confirmation_planned=len(confirmation))
            print(f"S3T search complete: {len(scoring)} screened, {len(confirmation)} confirmed", flush=True)
        except InterruptedError as error:
            print(str(error), flush=True)
        except Exception:
            runner.progress("failed")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "run"):
        item = sub.add_parser(action)
        item.add_argument("--config", type=Path, required=True,
                          help="Study config path, relative to this study or absolute")
        item.add_argument("--name", required=True, help="Unique run name")
        item.add_argument("--data-root", type=Path, default=ROOT,
                          help="Directory containing the original CIFAR archive folders")
        if action == "run":
            item.add_argument("--devices", nargs="+", default=["cuda:0"])
    item = sub.add_parser("resume")
    item.add_argument("--run", type=Path, required=True)
    item.add_argument("--devices", nargs="+", default=["cuda:0"])
    item = sub.add_parser("status")
    item.add_argument("--run", type=Path, required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--candidate", required=True)
    worker.add_argument("--pool", choices=("scoring", "confirmation"), required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.action in ("prepare", "run"):
        run = STUDY / "runs" / args.name
        if run.exists() and not (run / "plan.json").is_file():
            archive_incomplete_preparation(run)
        if not run.exists():
            prepare(args.config, args.name, args.data_root)
        elif args.action == "prepare":
            raise ValueError("Run is already prepared")
        if args.action == "run":
            execute(run, args.devices)
    elif args.action == "resume":
        execute(args.run, args.devices)
    elif args.action == "status":
        print(json.dumps(json.loads((args.run / "progress.json").read_text()), indent=2))
    else:
        run = args.run.resolve()
        plan = read_plan(run)
        candidate = next(c for c in plan["candidates"] if c["id"] == args.candidate)
        print(json.dumps(run_trial(run, candidate, args.pool, args.seed, args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
