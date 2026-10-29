"""Operational journals that distinguish interrupted work from lost measurements."""
import json
from pathlib import Path

from .core import write_json


class OperationTimer:
    def __init__(self, path, clock):
        self.path, self.clock = Path(path), clock
        self.record = json.loads(self.path.read_text()) if self.path.exists() else {
            "timing_scope": "update_v1", "update_time_sec": 0.0, "training_time_sec": 0.0,
            "status": "new", "measurement_complete": True}
        if self.record["status"] == "running":
            self.record["measurement_complete"] = False

    def __enter__(self):
        self.record["status"] = "running"
        write_json(self.path, self.record)
        self.excluded = 0.0
        self.started = self.clock()
        return self

    def exclude(self, seconds):
        self.excluded += seconds

    def add_training(self, seconds):
        self.record["training_time_sec"] += seconds

    def __exit__(self, exc_type, exc, traceback):
        self.record["update_time_sec"] += max(0.0, self.clock() - self.started - self.excluded)
        self.record["status"] = "complete" if exc_type is None else "interrupted"
        write_json(self.path, self.record)


def total_timing(paths):
    paths = list(paths)
    records = [json.loads(Path(path).read_text()) for path in paths if Path(path).exists()]
    complete = bool(paths) and len(records) == len(paths) and all(
        item.get("status") == "complete" and item.get("measurement_complete")
        and item.get("timing_scope") == "update_v1" for item in records)
    return {"timing_scope": "update_v1" if records else "unmeasured",
            "update_time_sec": sum(item["update_time_sec"] for item in records) if complete else None,
            "training_time_sec": sum(item["training_time_sec"] for item in records) if complete else None,
            "measurement_complete": complete}
