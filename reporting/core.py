"""Versioned, model-independent records; no training engine imports belong here."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

SCHEMA_VERSION = 1
SCHEMA_NAME = "unlearning_comparison"
STATUSES = {"ready", "requires_retraining", "failed", "incomplete"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def write_jsonl(path, records):
    atomic_text(path, "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in records))


@contextmanager
def record_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".comparison.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def runtime_provenance(root):
    root = Path(root)
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    files = list(root.glob("*.py"))
    for name in ("reporting", "lib", "s3t", "architectures"):
        files.extend((root / name).glob("*.py"))
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": platform.python_version(), "source_files": sources, "source_sha256": digest(sources)}


def hardware(device=None):
    result = {"platform": platform.platform(), "processor": platform.processor(),
              "device": str(device) if device is not None else None}
    # Do not import torch when regenerating reports on a machine without it.
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
        result["gpu"] = torch.cuda.get_device_name(device)
    return result


def manifest(*, run_id, method, dataset, model, config, split_sha256=None, workload_sha256=None,
             requested_steps=None, seed=None, provenance=None, hardware_info=None, precision=None,
             execution_mode="serial", measurement_origin="native"):
    return {"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
            "run_id": run_id, "method": method, "dataset": dataset, "model": model,
            "config": config, "config_sha256": digest(config), "seed": seed,
            "split_sha256": split_sha256, "workload_sha256": workload_sha256,
            "requested_steps": requested_steps, "provenance": provenance or {},
            "hardware": hardware_info or {}, "precision": precision,
            "execution_mode": execution_mode, "measurement_origin": measurement_origin}


def step_record(run_id, step, *, deleted_count=0, deleted_total=0, source_ids=None,
                state_id=None, status="ready", update_time_sec=None, training_time_sec=None,
                timing_scope="unmeasured", method_details=None):
    return {"schema_version": SCHEMA_VERSION, "run_id": run_id, "step": step,
            "operation": "initial_train" if step == 0 else "delete",
            "request_sha256": digest(source_ids) if source_ids is not None and step else None,
            "deleted_count": deleted_count, "deleted_total": deleted_total,
            "state_id": state_id or f"step-{step}", "status": status,
            "update_time_sec": update_time_sec, "training_time_sec": training_time_sec,
            "timing_scope": timing_scope, "method_details": method_details or {}}


def evaluation_identity(dataset, split, labels, *, split_sha256=None, source_sha256=None):
    """Bind the ordered evaluation pool; raw source identity is explicit if available.

    Labels alone cannot prove input identity. Missing raw fingerprints remain
    marked partial, and comparison validation surfaces that limitation.
    """
    labels = [int(value) for value in labels]
    binding = {"dataset": dataset, "split": split, "examples": len(labels),
               "labels_sha256": digest(labels), "split_sha256": split_sha256 if split == "val" else None,
               "source_sha256": source_sha256}
    return {**binding, "evaluation_sha256": digest(binding),
            "identity_status": "verified_source" if source_sha256 else "labels_and_protocol"}


def classification_metrics(confusion):
    classes = len(confusion)
    if not classes or any(len(row) != classes for row in confusion):
        raise ValueError("Confusion matrix must be square and nonempty")
    if any(type(value) is not int or value < 0 for row in confusion for value in row):
        raise ValueError("Confusion matrix counts must be nonnegative integers")
    total = sum(map(sum, confusion))
    if not total:
        raise ValueError("Classification evaluation requires examples")
    recalls, f1s, weighted = [], [], 0.0
    for index in range(classes):
        true = sum(confusion[index])
        predicted = sum(row[index] for row in confusion)
        tp = confusion[index][index]
        if true:
            recalls.append(tp / true)
        # Macro F1 uses classes present in labels OR predictions, zero_division=0.
        if true or predicted:
            f1 = 2 * tp / (true + predicted)
            f1s.append(f1)
            weighted += true * f1
    return {"accuracy": sum(confusion[i][i] for i in range(classes)) / total,
            "macro_f1": sum(f1s) / len(f1s), "weighted_f1": weighted / total,
            "balanced_accuracy": sum(recalls) / len(recalls)}


def classification_evaluation(directory, *, run_id, state_id, step, split, dataset,
                              labels=None, predictions=None, num_classes=None, scores=None,
                              split_sha256=None, source_sha256=None, evaluation_time_sec=None,
                              status="ready", method_details=None):
    """Retain compact aggregate predictions and labels outside checkpoint storage."""
    import numpy as np
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    labels = np.asarray(labels if labels is not None else [], dtype=np.int64)
    record = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "state_id": state_id,
              "step": step, "split": split, "status": status,
              **evaluation_identity(dataset, split, labels, split_sha256=split_sha256,
                                    source_sha256=source_sha256),
              "evaluation_time_sec": evaluation_time_sec, "metrics": {}, "confusion_matrix": None,
              "prediction_artifact": None, "prediction_sha256": None,
              "method_details": method_details or {}}
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=np.int64)
        if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
            raise ValueError("Predictions and labels must be matching nonempty vectors")
        classes = num_classes or int(max(labels.max(), predictions.max())) + 1
        if (labels.min() < 0 or predictions.min() < 0 or labels.max() >= classes or predictions.max() >= classes):
            raise ValueError("Classification labels/predictions outside class range")
        confusion = np.bincount(labels * classes + predictions, minlength=classes * classes).reshape(classes, classes)
        record["confusion_matrix"] = confusion.tolist()
        record["metrics"] = classification_metrics(record["confusion_matrix"])
        # Names depend on state and split, not machine-specific absolute paths.
        name = f"evaluation_data/{digest([state_id, split])[:24]}.npz"
        path = Path(directory) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"labels": labels, "predictions": predictions}
        if scores is not None:
            scores = np.asarray(scores, dtype=np.float32)
            if scores.shape != (len(labels), classes) or not np.isfinite(scores).all():
                raise ValueError("Invalid classification score array")
            arrays["scores"] = scores
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                np.savez_compressed(stream, **arrays)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        temporary.replace(path)
        record["prediction_artifact"] = name
        record["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif status == "ready":
        raise ValueError("Ready classification state requires predictions")
    return record


def record_evaluation(directory, evaluation):
    with record_lock(directory):
        records = read_jsonl(Path(directory) / "evaluations.jsonl")
        key = lambda item: (item["state_id"], item["split"])
        records = [item for item in records if key(item) != key(evaluation)] + [evaluation]
        write_jsonl(Path(directory) / "evaluations.jsonl", sorted(records, key=lambda x: (x["step"], x["state_id"], x["split"])))


def validate_bundle(run, steps, evaluations):
    if run.get("schema_name") != SCHEMA_NAME or run.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported comparison schema")
    if run.get("config_sha256") != digest(run["config"]):
        raise ValueError("Comparison configuration checksum mismatch")
    if not all(isinstance(run.get(key), str) and run[key] for key in ("run_id", "method", "dataset", "model")):
        raise ValueError("Run identity fields must be nonempty strings")
    previous_total = 0
    requested = run.get("requested_steps")
    if requested is not None and (type(requested) is not int or requested < 0 or len(steps) > requested + 1):
        raise ValueError("Invalid requested step count or steps beyond the workload")
    for index, item in enumerate(steps):
        if item.get("schema_version") != SCHEMA_VERSION or item.get("run_id") != run["run_id"] or item.get("step") != index:
            raise ValueError("Steps must be contiguous from zero and belong to this run")
        if item["status"] not in STATUSES or item["operation"] != ("initial_train" if index == 0 else "delete"):
            raise ValueError("Invalid step status or operation")
        if any(type(item[key]) is not int or item[key] < 0 for key in ("deleted_count", "deleted_total")):
            raise ValueError("Deletion counts must be nonnegative integers")
        if item["deleted_total"] != previous_total + item["deleted_count"] or (index == 0 and item["deleted_total"]):
            raise ValueError("Deletion counts disagree")
        previous_total = item["deleted_total"]
        for key in ("update_time_sec", "training_time_sec"):
            value = item.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid {key}")
    keys = set()
    for item in evaluations:
        key = (item["state_id"], item["split"])
        if item.get("schema_version") != SCHEMA_VERSION or item.get("run_id") != run["run_id"] or key in keys:
            raise ValueError("Duplicate evaluation or incompatible run/schema")
        if item["split"] not in ("val", "test") or item["status"] not in STATUSES:
            raise ValueError("Invalid evaluation split or status")
        if item["status"] != "ready" and item["metrics"]:
            raise ValueError("Unavailable predictor cannot have evaluation scores")
        # Intermediate exhausted states can remain as diagnostics after retraining.
        if not any(step["step"] == item["step"] for step in steps):
            raise ValueError("Evaluation refers to an unknown step")
        keys.add(key)
        if item.get("confusion_matrix") is not None:
            computed = classification_metrics(item["confusion_matrix"])
            if sum(map(sum, item["confusion_matrix"])) != item["examples"]:
                raise ValueError("Confusion matrix/example count mismatch")
            if any(name in item["metrics"] and not math.isclose(item["metrics"][name], value, abs_tol=1e-9)
                   for name, value in computed.items()):
                raise ValueError("Stored classification metrics disagree with confusion matrix")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in item["metrics"].values()):
            raise ValueError("Evaluation metrics must be finite numbers")


def write_bundle(directory, run, steps, evaluations=None):
    directory = Path(directory)
    with record_lock(directory):
        if evaluations is None:
            evaluations = read_jsonl(directory / "evaluations.jsonl")
        validate_bundle(run, steps, evaluations)
        # run.json is the commit marker. Readers take the same lock.
        write_jsonl(directory / "steps.jsonl", steps)
        write_jsonl(directory / "evaluations.jsonl", evaluations)
        write_json(directory / "run.json", run)


def load_bundle(directory):
    directory = Path(directory)
    # Reporting can read records from another repo or a read-only archive.
    from contextlib import nullcontext
    lock = directory / ".comparison.lock"
    with lock.open("r") if lock.exists() else nullcontext() as stream:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_SH)
        run = json.loads((directory / "run.json").read_text())
        steps = read_jsonl(directory / "steps.jsonl")
        evaluations = read_jsonl(directory / "evaluations.jsonl")
        validate_bundle(run, steps, evaluations)
    return run, steps, evaluations
