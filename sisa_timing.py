"""Versioned update timings and reports that remain usable after model cleanup."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

from container_storage import record_directory


TIMING_SCOPE = "update_v1"
REPORT_FIELDS = ["nb_shards", "nb_requests", "accuracy", "retraining_time", "hyperparams_json",
                 "update_time_sec", "training_time_sec", "timing_scope"]


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    temporary.replace(path)


class UpdateTimer:
    """Accumulate this request's work across slices and recoverable invocations.

    The supplied clock synchronizes the active device. Publishing the timing
    journal itself is excluded. An unclean termination marks timing incomplete
    rather than silently dropping the interval since the last journal write.
    """
    def __init__(self, directory, shard, label, binding, clock):
        self.path = Path(directory) / "times" / f"shard-{shard}:{label}.json"
        self.clock = clock
        self.started = None
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record["binding"] != binding or self.record["timing_scope"] != TIMING_SCOPE:
                raise ValueError("Timing record belongs to different training settings; use a new container")
            if self.record["status"] == "running":
                self.record["measurement_complete"] = False
        else:
            self.record = {"schema_version": 1, "timing_scope": TIMING_SCOPE,
                           "binding": binding, "shard": shard, "label": str(label),
                           "status": "new", "measurement_complete": True,
                           "update_time_sec": 0.0, "training_time_sec": 0.0, "slices": []}

    def __enter__(self):
        self.record["status"] = "running"
        self._write()
        self.started = self.clock()
        return self

    def slice(self, index, slice_hash, reused):
        for item in self.record["slices"]:
            if item["slice"] == index:
                if item["hash"] != slice_hash:
                    raise ValueError("Slice membership changed during a timed update")
                return item
        item = {"slice": index, "hash": slice_hash, "reused": reused,
                "training_time_sec": 0.0, "epochs_trained": 0}
        self.record["slices"].append(item)
        return item

    def add_training(self, item, seconds, complete=True):
        item["training_time_sec"] += seconds
        item["epochs_trained"] += int(complete)
        item["reused"] = False
        self.record["training_time_sec"] += seconds

    def _write(self):
        atomic_text(self.path, json.dumps(self.record, indent=2, sort_keys=True) + "\n")

    def checkpoint(self, status="running"):
        self.record["update_time_sec"] += self.clock() - self.started
        self.record["status"] = status
        self._write()
        if status == "complete":
            # Replace legacy last-slice timing symlinks with a request total.
            atomic_text(self.path.with_suffix(".time"), f"{self.record['update_time_sec']}\n")
        self.started = self.clock()

    def __exit__(self, exc_type, exc, traceback):
        self.checkpoint("complete" if exc_type is None else "interrupted")


def summarize(container, label, shards=None):
    directory = record_directory(container)
    if shards is None:
        import numpy as np
        shards = len(np.load(directory / "splitfile.npy", allow_pickle=True))
    if shards < 1:
        raise ValueError("At least one shard is required")
    records = []
    for shard in range(shards):
        path = directory / "times" / f"shard-{shard}:{label}.json"
        if not path.exists():
            raise ValueError(f"Missing complete update timing: {path}. Legacy .time files cannot reconstruct update time.")
        record = json.loads(path.read_text())
        if (record.get("timing_scope") != TIMING_SCOPE or record.get("status") != "complete"
                or not record.get("measurement_complete") or record.get("shard") != shard
                or record.get("label") != str(label)):
            raise ValueError(f"Incomplete or incompatible update timing: {path}")
        records.append(record)
    return {"update_time_sec": sum(item["update_time_sec"] for item in records),
            "training_time_sec": sum(item["training_time_sec"] for item in records),
            "timing_scope": TIMING_SCOPE, "shards": shards}


def write_reports(container, dataset, shards, requests, output, hyperparams):
    # Validate timing for every label before invoking evaluation or changing CSVs.
    totals = [summarize(container, label, shards) for label in range(requests + 1)]
    rows = []
    for label, total in enumerate(totals):
        accuracy = float(subprocess.check_output(
            [sys.executable, "aggregation.py", "--strategy", "uniform", "--container", container,
             "--shards", str(shards), "--dataset", dataset, "--label", str(label)], text=True).strip())
        rows.append({"nb_shards": shards, "nb_requests": label, "accuracy": accuracy,
                     "retraining_time": total["update_time_sec"], "hyperparams_json": hyperparams,
                     **{key: total[key] for key in ("update_time_sec", "training_time_sec", "timing_scope")}})
    output = Path(output)
    old_rows = []
    if output.exists():
        with output.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(reader.fieldnames or []).issubset(REPORT_FIELDS):
                raise ValueError(f"Unexpected report columns in {output}; use a new report path")
            for row in reader:
                # Preserve historical measurements, explicitly labelling their scope.
                row["timing_scope"] = row.get("timing_scope") or "legacy_batch_last_slice"
                old_rows.append(row)
    def render(items):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(items)
        return buffer.getvalue()
    local_report = record_directory(container) / "report.csv"
    if output.absolute() != local_report.absolute():
        atomic_text(output, render(old_rows + rows))
    atomic_text(local_report, render(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hyperparams", default="-")
    args = parser.parse_args()
    if args.requests < 0:
        parser.error("requests cannot be negative")
    write_reports(args.container, args.dataset, args.shards, args.requests, args.output, args.hyperparams)


if __name__ == "__main__":
    main()
