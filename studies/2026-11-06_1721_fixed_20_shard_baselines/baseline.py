#!/usr/bin/env python3
"""Fixed-20-shard development search for SISA and ProtoSISA."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import fcntl
from hashlib import sha256
import io
from itertools import product
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import random
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from reporting.core import atomic_text, digest, write_json
from sisa_cost import normalized_expected_cost
from sisa_schedule import epochs_per_slice
from studies.artifacts import capture_run, verify_run_source


def now():
    return datetime.now(timezone.utc).isoformat()


def signed_write(path, value):
    write_json(path, {**value, "sha256": digest(value)})
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_arrays(path, **arrays):
    import numpy as np
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def signed_read(path):
    value = json.loads(Path(path).read_text())
    checksum = value.pop("sha256")
    if digest(value) != checksum:
        raise ValueError(f"Checksum mismatch: {path}")
    return value


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def validate_memberships(parent, development):
    for value in (parent, development):
        body = {key: item for key, item in value.items() if key != "manifest_sha256"}
        if digest(body) != value["manifest_sha256"]:
            raise ValueError("Source manifest checksum mismatch")
    if (development["parent_manifest_sha256"] != parent["manifest_sha256"]
            or development["source_sha256"] != parent["source_sha256"]
            or development["dataset_id"] != parent["dataset_id"]
            or parent["split_seed"] != 42 or development["split_seed"] != 42
            or not parent["stratified"] or not development["stratified"]):
        raise ValueError("Development manifest does not match the fixed parent split")
    size = parent["source_size"]
    final, pool = parent["train_indices"], parent["val_indices"]
    fit, score, confirm = (development[key] for key in
                           ("train_indices", "val_indices", "confirmation_indices"))
    arrays = (final, pool, fit, score, confirm)
    if any(type(index) is not int or not 0 <= index < size for values in arrays for index in values):
        raise ValueError("Invalid source index")
    if (sorted(final + pool) != list(range(size)) or len(pool) != size // 10
            or sorted(fit + score + confirm) != sorted(pool)
            or [len(fit), len(score), len(confirm)] != [len(pool) - 2 * (len(pool) // 5),
                                                       len(pool) // 5, len(pool) // 5]
            or set(confirm) & set(development.get("confirmation_excluded_indices", []))):
        raise ValueError("Expected disjoint 90% final / 10% development, split 60/20/20")
    return {"fit": fit, "score": score, "confirmation": confirm}


def candidate_plan(settings, fit_size):
    budgets = settings["cost_budgets"]
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] <= 0:
        raise ValueError("Cost budgets must be positive, increasing and unique")
    if settings["trials"] < len(budgets):
        raise ValueError("Need at least one candidate per budget stratum")
    low, high = settings["lr_bounds"]
    if not 0 < low < high:
        raise ValueError("Invalid LR bounds")
    shapes = [[] for _ in budgets]
    for shards, slices, epochs in product(settings["shards"], settings["slices"], settings["epochs"]):
        try:
            passes = epochs_per_slice(epochs, slices)
        except ValueError:
            continue
        if shards * slices > fit_size:
            continue
        cost = normalized_expected_cost(shards, slices, epochs=epochs,
                                        reference_epochs=settings["reference_epochs"])
        for index, ceiling in enumerate(budgets):
            if cost <= ceiling:
                shapes[index].append(dict(shards=shards, slices=slices, epochs=epochs,
                                          epochs_per_slice=passes, normalized_expected_cost=cost))
                break
    if any(not group for group in shapes):
        raise ValueError("A cost stratum has no valid configurations")
    rng = random.Random(settings["search_seed"])
    for group in shapes:
        rng.shuffle(group)
    trials = []
    for index in range(settings["trials"]):
        stratum = index % len(budgets)
        shape = shapes[stratum][(index // len(budgets)) % len(shapes[stratum])]
        learning_rate = float(f"{math.exp(rng.uniform(math.log(low), math.log(high))):.10g}")
        trials.append(dict(id=f"candidate-{index:03d}", **shape, learning_rate=learning_rate,
                           cost_stratum=budgets[stratum]))
    return trials


def context(run):
    plan = signed_read(run / "plan.json")
    if plan["settings"].get("shards") != [20]:
        raise ValueError("Fixed-20 baseline study requires exactly 20 shards")
    parent = json.loads((run / "inputs/parent.json").read_text())
    development = json.loads((run / "inputs/development.json").read_text())
    dataset_id = "glue/sst2" if plan["settings"]["dataset"] == "sst2" else plan["settings"]["dataset"]
    if parent["dataset_id"] != dataset_id:
        raise ValueError("Saved baseline manifest belongs to another dataset")
    memberships = validate_memberships(parent, development)
    if (parent["manifest_sha256"] != plan["parent_manifest_sha256"]
            or development["manifest_sha256"] != plan["development_manifest_sha256"]):
        raise ValueError("Saved plan refers to different data")
    return plan, memberships


def prepare(args):
    settings = json.loads(Path(args.config).read_text())
    if settings.get("shards") != [20]:
        raise ValueError("Fixed-20 baseline study requires exactly 20 shards")
    if settings["dataset"] == "sst2":
        import numpy as np
        path = Path(settings["source_root"]) / "datasets/sst2/sst2_train.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing tokenized SST-2 file: {path.resolve()}. source_root must point to "
                "the directory containing datasets/. From machine-unlearning, run: "
                "python datasets/sst2/prepare_data.py")
        with np.load(path, allow_pickle=False) as arrays:
            settings["training_tokens_sha256"] = sha256(
                arrays["X"].astype(np.int64).tobytes() + arrays["y"].astype(np.int64).tobytes()).hexdigest()
    parent = json.loads(Path(settings["parent_manifest"]).read_text())
    development = json.loads(Path(settings["development_manifest"]).read_text())
    memberships = validate_memberships(parent, development)
    trials = candidate_plan(settings, len(memberships["fit"]))
    run = STUDY / "runs" / args.name
    if run.exists():
        raise ValueError("Run already exists; use execute to resume or choose a new name")
    capture_run(run, STUDY, settings, ROOT)
    (run / "inputs").mkdir()
    write_json(run / "inputs/parent.json", parent)
    write_json(run / "inputs/development.json", development)
    signed_write(run / "plan.json", dict(settings=settings, candidates=trials,
        created_at=now(), parent_manifest_sha256=parent["manifest_sha256"],
        development_manifest_sha256=development["manifest_sha256"],
        protocol="dev10_fit60_score20_confirm20_cost_stratified_random_v1"))
    # Read the actual source and verify its fingerprint, without fitting a model.
    DevelopmentLoader(run, "score")
    print(json.dumps(dict(run=str(run), candidates=len(trials), methods=["sisa", "protosisa"],
                          memberships={key: len(value) for key, value in memberships.items()})), flush=True)


class DevelopmentLoader:
    """Only the original training source is opened; official test data is never loaded."""
    def __init__(self, run, pool):
        import numpy as np
        plan, memberships = context(run)
        if pool not in ("score", "confirmation"):
            raise ValueError("Unknown development evaluation pool")
        if pool == "confirmation" and not (run / "finalists.json").exists():
            raise ValueError("Confirmation requires a frozen finalist set")
        settings = plan["settings"]
        parent = json.loads((run / "inputs/parent.json").read_text())
        self.tokens = None
        if settings["dataset"] == "sst2":
            path = Path(settings["source_root"]) / "datasets/sst2/sst2_train.npz"
            with np.load(path, allow_pickle=False) as arrays:
                if "source_sha256" not in arrays or str(arrays["source_sha256"].item()) != parent["source_sha256"]:
                    raise ValueError("SST-2 original training source fingerprint changed")
                self.tokens = arrays["X"].astype(np.int64)
                self.labels = arrays["y"].astype(np.int64)
            if sha256(self.tokens.tobytes() + self.labels.tobytes()).hexdigest() != settings["training_tokens_sha256"]:
                raise ValueError("SST-2 token arrays changed after plan preparation")
            if len(self.labels) != parent["source_size"] or len(self.tokens) != len(self.labels):
                raise ValueError("SST-2 token rows differ from original source indices")
            self.classes = 2
            self.input_shape = [self.tokens.shape[1]]
        else:
            from torchvision import datasets, transforms
            from dataset_splits import fingerprint_cifar
            constructor = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[settings["dataset"]]
            normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            train = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(), normalization])
            evaluation = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalization])
            self.train = constructor(root=settings["source_root"], train=True, download=False, transform=train)
            self.evaluation = constructor(root=settings["source_root"], train=True, download=False, transform=evaluation)
            if fingerprint_cifar(self.train.data, self.train.targets) != parent["source_sha256"]:
                raise ValueError("Training source fingerprint changed")
            self.classes = len(self.train.classes)
            self.input_shape = [32, 32]
        self.fit = np.asarray(memberships["fit"], dtype=np.int64)
        self.selected = np.asarray(memberships[pool], dtype=np.int64)

    def load(self, indices, category="train"):
        import numpy as np
        import torch
        if self.tokens is not None:
            source = self.selected if category == "val" else self.fit
            if category not in ("train", "train_eval", "val"):
                raise ValueError("Search may only access development fitting/scoring data")
            selected = source[np.asarray(indices, dtype=np.int64)]
            return self.tokens[selected], self.labels[selected]
        if category == "train":
            dataset, source = self.train, self.fit
        elif category == "train_eval":
            dataset, source = self.evaluation, self.fit
        elif category == "val":
            dataset, source = self.evaluation, self.selected
        else:
            raise ValueError("Search may only access development fitting/scoring data")
        rows = [dataset[int(source[index])] for index in indices]
        return torch.stack([row[0] for row in rows]).numpy(), np.asarray([row[1] for row in rows])


def feature_paths(run, pool):
    return run / "features" / f"{pool}.npz", run / "features" / f"{pool}.json"


def feature_marker(run, pool):
    path, marker = feature_paths(run, pool)
    if not marker.exists():
        return None
    value = signed_read(marker)
    if value["plan_sha256"] != file_hash(run / "plan.json") or file_hash(path) != value["array_sha256"]:
        raise ValueError("Frozen-feature cache identity changed")
    return value


def features(run, pool, device):
    import numpy as np
    import torch
    import protosisa as proto
    if feature_marker(run, pool):
        return
    plan, _ = context(run)
    torch.set_num_threads(plan["settings"]["threads"])
    if pool == "confirmation" and not (run / "finalists.json").exists():
        raise ValueError("Confirmation requires a frozen finalist set")
    loader = DevelopmentLoader(run, pool)
    proto.seed_all(42)
    if plan["settings"]["model"] == "densenet121":
        from torchvision.models import DenseNet121_Weights, densenet121
        encoder = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        encoder.classifier = torch.nn.Identity()
        encoder = encoder.to(device).requires_grad_(False).eval()
    else:
        encoder = proto.build_encoder(plan["settings"]["model"], torch.device(device))
    batch_size = plan["settings"]["batch_size"]
    statistics = proto.prototype_statistics(encoder, loader, np.arange(len(loader.fit)), loader.classes,
                                            batch_size, torch.device(device))
    statistics = proto.merge_statistics([statistics])
    scores, labels = [], []
    with torch.inference_mode():
        for inputs, target in proto.batches(loader, np.arange(len(loader.selected)), batch_size, "val"):
            values = encoder(torch.as_tensor(inputs, device=device))
            scores.append(proto.prototype_logits(values, statistics).cpu().numpy())
            labels.extend(target.tolist())
    path, marker = feature_paths(run, pool)
    path.parent.mkdir(exist_ok=True)
    save_arrays(path, prototype=np.concatenate(scores), labels=np.asarray(labels))
    signed_write(marker, dict(plan_sha256=file_hash(run / "plan.json"), array_sha256=file_hash(path),
                             fitting_examples=len(loader.fit), evaluation_examples=len(loader.selected),
                             prototype_counts=statistics["counts"].tolist(), pool=pool, created_at=now()))
    print(f"Prepared frozen prototype features for {pool}", flush=True)


def trial_directory(run, candidate, pool, seed):
    return run / ("search" if pool == "score" else "confirmation") / f"{candidate['id']}-seed{seed}"


def result(run, candidate, pool, seed):
    path = trial_directory(run, candidate, pool, seed) / "result.json"
    if not path.exists():
        return None
    value = signed_read(path)
    if (value["plan_sha256"] != file_hash(run / "plan.json") or value["candidate"] != candidate
            or value["pool"] != pool or value["seed"] != seed):
        raise ValueError("Completed trial identity changed")
    if file_hash(path.parent / "predictions.npz") != value["predictions_sha256"]:
        raise ValueError("Completed trial predictions changed")
    return value


def build_task_model(config, device):
    """Apply an optional frozen study scope without changing legacy defaults."""
    import torch
    import protosisa as proto
    model = proto.build_model(config, device)
    settings = config.get("model_args", {})
    scope = settings.get("layers_to_finetune")
    if scope is not None and scope != "all":
        if not isinstance(scope, list) or not scope or any(not isinstance(x, str) or not x for x in scope):
            raise ValueError("layers_to_finetune must be all or a nonempty list of prefixes")
        names = list(model.model.named_parameters())
        if any(not any(name.startswith(prefix) for name, _ in names) for prefix in scope):
            raise ValueError("A selected trainable prefix matches no parameters")
        for name, parameter in names:
            parameter.requires_grad_(name.startswith(tuple(scope)))
    initialization = settings.get("task_head_initialization")
    if initialization is not None:
        if config["model"] != "vit" or initialization != "xavier_uniform":
            raise ValueError("Explicit task-head initialization currently supports ViT Xavier uniform only")
        # The native ViT study replaces its head and then performs this same
        # initialization. Both factories consume the same initial Linear draw.
        head = model.model.heads.head
        torch.nn.init.xavier_uniform_(head.weight, gain=1.0)
        torch.nn.init.zeros_(head.bias)
    return model


def run_trial(run, candidate, pool, seed, device, *, loader=None, model_factory=None):
    import numpy as np
    import torch
    import protosisa as proto
    from container_storage import initialize_container
    from reporting import classification_evaluation, manifest, step_record, write_bundle
    from reporting.core import hardware, runtime_provenance
    if result(run, candidate, pool, seed):
        return
    plan, memberships = context(run)
    if pool == "confirmation":
        if not (run / "finalists.json").exists():
            raise ValueError("Confirmation requires a frozen finalist set")
        finalists = signed_read(run / "finalists.json")
        if candidate["id"] not in finalists["candidate_ids"]:
            raise ValueError("Candidate was not frozen for confirmation")
    settings = plan["settings"]
    torch.set_num_threads(settings["threads"])
    device = torch.device(device)
    loader = loader or DevelopmentLoader(run, pool)
    model_factory = model_factory or build_task_model
    feature_marker(run, pool)
    prototype = np.load(feature_paths(run, pool)[0])
    directory = initialize_container(str(trial_directory(run, candidate, pool, seed)))
    config = dict(model=settings["model"], dataset=settings["dataset"],
                  dataset_metadata=dict(nb_classes=loader.classes, input_shape=loader.input_shape),
                  seed=seed, optimizer=settings["optimizer"], batch_size=settings["batch_size"],
                  dropout_rate=settings["dropout_rate"], learning_rate=candidate["learning_rate"],
                  slices=candidate["slices"], epochs=candidate["epochs"], shards=candidate["shards"],
                  prototype_weighting="safe", prototype_weight=None, eval_split="val",
                  plan_sha256=file_hash(run / "plan.json"), development_pool=pool)
    if "model_args" in settings:
        config["model_args"] = settings["model_args"]
    config["training_sha256"] = digest(config)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing trial settings changed")
    write_json(config_path, config)
    shards = np.array_split(np.random.default_rng(seed).permutation(len(memberships["fit"])), candidate["shards"])
    stored_shards = np.empty(len(shards), dtype=object)
    stored_shards[:] = shards
    np.save(directory / "splitfile.npy", stored_shards, allow_pickle=True)
    ensemble = np.zeros_like(prototype["prototype"], dtype=np.float64)
    started = time.monotonic()
    visits = 0
    for shard_id, shard in enumerate(shards):
        member_path = directory / "outputs" / f"member-{shard_id}.npz"
        marker = directory / "outputs" / f"member-{shard_id}.json"
        paths = proto.checkpoint_paths(directory, config, shard_id, shard, [])
        if marker.exists():
            saved = signed_read(marker)
            if (saved["training_sha256"] != config["training_sha256"]
                    or file_hash(member_path) != saved["array_sha256"]):
                raise ValueError("Completed member predictions changed")
            values = np.load(member_path)
        else:
            checkpoint = proto.train_member(directory, config, shard_id, shard, [], 0, loader, device, model_factory)
            model = model_factory(config, device)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["model"])
            model.eval()
            scores, labels = [], []
            with torch.inference_mode():
                for inputs, target in proto.batches(loader, np.arange(len(loader.selected)), settings["batch_size"], "val"):
                    scores.append(model(torch.as_tensor(inputs, device=device)).cpu().numpy())
                    labels.extend(target.tolist())
            del model
            scores = np.concatenate(scores)
            if not np.isfinite(scores).all():
                raise ValueError("Nonfinite candidate predictions")
            save_arrays(member_path, scores=scores, labels=np.asarray(labels))
            signed_write(marker, dict(training_sha256=config["training_sha256"], array_sha256=file_hash(member_path)))
            values = np.load(member_path)
        if not np.array_equal(values["labels"], prototype["labels"]):
            raise ValueError("Ensemble and prototype evaluation memberships disagree")
        ensemble += values["scores"] / len(shards)
        values.close()
        visits += sum(len(indices) * passes for _, indices, passes in paths)
        # A durable member-prediction marker is the resume boundary. Preserve
        # its config, memberships, predictions and timing; discard only its models.
        for path, _, _ in paths:
            path.unlink(missing_ok=True)
        print(f"{candidate['id']} {pool} seed={seed} member={shard_id + 1}/{len(shards)} scored", flush=True)
    mixture = proto.mixture_parameters(config, len(memberships["fit"]), len(shards))
    weight = mixture["prototype_weight"]
    mixed = (1 - weight) * ensemble + weight * prototype["prototype"]
    labels = prototype["labels"]
    predictions = directory / "predictions.npz"
    save_arrays(predictions, sisa=ensemble, protosisa=mixed, labels=labels)
    value = dict(candidate=candidate, pool=pool, seed=seed, plan_sha256=file_hash(run / "plan.json"),
                 sisa_accuracy=float(np.mean(ensemble.argmax(1) == labels)),
                 protosisa_accuracy=float(np.mean(mixed.argmax(1) == labels)),
                 prototype_accuracy=float(np.mean(prototype["prototype"].argmax(1) == labels)),
                 prototype_weight=weight, fitting_examples=len(memberships["fit"]),
                 evaluation_examples=len(labels), initial_example_presentations=visits,
                 elapsed_sec=time.monotonic() - started, completed_at=now(),
                 predictions_sha256=file_hash(predictions), checkpoint_retention="scores_only")
    for method, scores in (("sisa", ensemble), ("protosisa", mixed)):
        report = directory / method
        run_id = f"baseline-search-{directory.name}-{pool}-{method}"
        evaluation = classification_evaluation(report, run_id=run_id, state_id="step-0", step=0,
            split="val", dataset=settings["dataset"], labels=labels, scores=scores,
            predictions=scores.argmax(1), num_classes=loader.classes,
            split_sha256=plan["development_manifest_sha256"],
            source_sha256=json.loads((run / "inputs/development.json").read_text())["source_sha256"],
            method_details=dict(development_pool=pool, candidate=candidate, **mixture))
        run_manifest = manifest(run_id=run_id, method=method, dataset=settings["dataset"], model=settings["model"],
            config=config, split_sha256=plan["development_manifest_sha256"], seed=seed,
            provenance=runtime_provenance(ROOT), hardware_info=hardware(device), precision="fp32",
            requested_steps=0, execution_mode="concurrent_development_search")
        write_bundle(report, run_manifest, [step_record(run_id, 0)], [evaluation])
    signed_write(directory / "result.json", value)
    print(json.dumps(value), flush=True)


def search_results(run):
    plan, _ = context(run)
    seed = plan["settings"]["training_seed"]
    return [item for candidate in plan["candidates"] if (item := result(run, candidate, "score", seed))]


def select_finalists(results, budgets, count):
    selections, ids = [], set()
    for method in ("sisa", "protosisa"):
        for budget in budgets:
            eligible = [row for row in results if row["candidate"]["normalized_expected_cost"] <= budget]
            eligible.sort(key=lambda row: (-row[f"{method}_accuracy"], row["candidate"]["normalized_expected_cost"],
                                          row["candidate"]["learning_rate"], row["candidate"]["id"]))
            chosen = [row["candidate"]["id"] for row in eligible[:count]]
            if not chosen:
                raise ValueError("No completed candidate meets a cost budget")
            ids.update(chosen)
            selections.append(dict(method=method, cost_budget=budget, candidate_ids=chosen))
    return dict(candidate_ids=sorted(ids), selections=selections)


def freeze_finalists(run):
    plan, _ = context(run)
    results = search_results(run)
    if len(results) != len(plan["candidates"]):
        raise ValueError("Cannot freeze finalists before all search trials complete")
    settings = plan["settings"]
    selected = select_finalists(results, settings["cost_budgets"], settings["finalists_per_method_budget"])
    value = dict(**selected, plan_sha256=file_hash(run / "plan.json"),
                 search_results_sha256=digest(results))
    path = run / "finalists.json"
    if path.exists():
        if signed_read(path) != value:
            raise ValueError("Frozen finalist selection changed")
    else:
        signed_write(path, value)
    return value


def write_csv(path, rows):
    if not rows:
        return
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def summarize(run):
    import statistics
    plan, _ = context(run)
    rows, confirmation = [], []
    for candidate in plan["candidates"]:
        for pool, seeds, output in (("score", [plan["settings"]["training_seed"]], rows),
                                    ("confirmation", plan["settings"]["confirmation_seeds"], confirmation)):
            for seed in seeds:
                value = result(run, candidate, pool, seed)
                if value:
                    for method in ("sisa", "protosisa"):
                        output.append(dict(method=method, **candidate, seed=seed, accuracy=value[f"{method}_accuracy"],
                                           prototype_weight=value["prototype_weight"],
                                           initial_example_presentations=value["initial_example_presentations"]))
    write_csv(run / "search_results.csv", rows)
    write_csv(run / "confirmation_results.csv", confirmation)
    selected = []
    if (run / "finalists.json").exists():
        frozen = signed_read(run / "finalists.json")
        for group in frozen["selections"]:
            options = []
            for identity in group["candidate_ids"]:
                values = [row for row in confirmation if row["method"] == group["method"] and row["id"] == identity]
                if len(values) != len(plan["settings"]["confirmation_seeds"]):
                    continue
                accuracy = [row["accuracy"] for row in values]
                options.append(dict(method=group["method"], cost_budget=group["cost_budget"],
                    **{key: values[0][key] for key in ("id", "shards", "slices", "epochs", "learning_rate", "normalized_expected_cost")},
                    mean_accuracy=statistics.mean(accuracy),
                    std_accuracy=statistics.stdev(accuracy) if len(accuracy) > 1 else 0.0, seeds=len(accuracy)))
            if options:
                options.sort(key=lambda row: (-row["mean_accuracy"], row["normalized_expected_cost"], row["id"]))
                selected.append(options[0])
        write_csv(run / "selected_by_budget.csv", selected)
    return dict(search_completed=len(rows) // 2, search_planned=len(plan["candidates"]),
                confirmation_completed=len(confirmation) // 2, selected=selected)


def check_resources(settings):
    for directory in (ROOT, ROOT / "saved_models"):
        if shutil.disk_usage(directory).free < settings["minimum_free_gb"] * 1024**3:
            raise RuntimeError(f"Less than {settings['minimum_free_gb']} GiB free at {directory}")
    cap = settings.get("maximum_gpu_power_watts")
    if cap is not None:
        values = subprocess.check_output(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"], text=True)
        limits = [float(line) for line in values.splitlines() if line.strip()]
        if not limits or any(not math.isfinite(limit) or limit > cap for limit in limits):
            raise RuntimeError("GPU power limits exceed the configured maximum")


def execute(run, devices):
    verify_run_source(run)
    plan, _ = context(run)
    settings = plan["settings"]
    check_resources(settings)
    lock = (run / ".search.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, errors = {}, []
    state_lock = threading.Lock()
    phase = "search"

    def progress(status="running"):
        counts = summarize(run)
        write_json(run / "progress.json", dict(status=status, phase=phase, coordinator_pid=os.getpid(),
            active=dict(active), errors=list(errors), updated_at=now(), **counts))

    def invoke(mode, pool, device, candidate=None, seed=None):
        command = [sys.executable, "-u", str(Path(__file__).resolve()), mode, "--run", str(run),
                   "--pool", pool, "--device", device]
        if candidate:
            command += ["--candidate", candidate["id"], "--seed", str(seed)]
            log = trial_directory(run, candidate, pool, seed).with_suffix(".log")
        else:
            log = run / f"features-{pool}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, OMP_NUM_THREADS=str(settings["threads"]), MKL_NUM_THREADS=str(settings["threads"]))
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            with state_lock:
                active[device] = dict(pid=process.pid, candidate=candidate["id"] if candidate else "frozen_features",
                                      seed=seed, pool=pool, log=str(log))
                progress()
            code = process.wait()
            with state_lock:
                active.pop(device, None)
                if code:
                    errors.append(dict(command=command, returncode=code, log=str(log)))
                progress()
            if code:
                raise RuntimeError(f"Worker exited {code}: {log}")

    def phase_jobs(pool, jobs):
        pending = Queue()
        for candidate, seed in jobs:
            if not result(run, candidate, pool, seed):
                pending.put((candidate, seed))

        def worker(device):
            while not (run / "STOP").exists() and not errors:
                try:
                    candidate, seed = pending.get_nowait()
                except Empty:
                    return
                check_resources(settings)
                invoke("worker", pool, device, candidate, seed)

        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(worker, device) for device in devices]
            for future in futures:
                future.result()
        return pending.empty()

    try:
        progress()
        if (run / "STOP").exists():
            progress("stopped")
            return
        if not feature_marker(run, "score"):
            invoke("features", "score", devices[0])
        jobs = [(candidate, settings["training_seed"]) for candidate in plan["candidates"]]
        if not phase_jobs("score", jobs):
            progress("stopped")
            return
        frozen = freeze_finalists(run)
        phase = "confirmation"
        if (run / "STOP").exists():
            progress("stopped")
            return
        if not feature_marker(run, "confirmation"):
            invoke("features", "confirmation", devices[0])
        jobs = [(candidate, seed) for candidate in plan["candidates"] if candidate["id"] in frozen["candidate_ids"]
                for seed in settings["confirmation_seeds"]]
        complete = phase_jobs("confirmation", jobs)
        progress("completed" if complete else "stopped")
    except BaseException as error:
        errors.append(dict(error=repr(error)))
        progress("failed")
        raise
    finally:
        lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare", help="Persist a plan and validate data; no training")
    prepare_command.add_argument("--config", required=True)
    prepare_command.add_argument("--name", required=True)
    for mode in ("execute", "worker", "features", "summarize"):
        command = commands.add_parser(mode)
        command.add_argument("--run", type=Path, required=True)
        if mode == "execute":
            command.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
        if mode in ("worker", "features"):
            command.add_argument("--pool", choices=("score", "confirmation"), required=True)
            command.add_argument("--device", required=True)
        if mode == "worker":
            command.add_argument("--candidate", required=True)
            command.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    if args.command == "prepare":
        if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
            parser.error("Run name must be one path component")
        prepare(args)
    else:
        args.run = args.run.resolve()
        if args.run.parent != STUDY / "runs":
            parser.error("Run must belong to this study's runs directory")
        verify_run_source(args.run)
        if args.command == "execute":
            if not args.devices or len(set(args.devices)) != len(args.devices):
                parser.error("Devices must be nonempty and unique")
            execute(args.run, args.devices)
        elif args.command == "features":
            features(args.run, args.pool, args.device)
        elif args.command == "worker":
            plan, _ = context(args.run)
            candidate = next(item for item in plan["candidates"] if item["id"] == args.candidate)
            run_trial(args.run, candidate, args.pool, args.seed, args.device)
        else:
            print(json.dumps(summarize(args.run), indent=2))


if __name__ == "__main__":
    main()
