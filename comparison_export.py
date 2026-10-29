"""Export SISA/S3T saved records without loading models or original datasets."""
import argparse
import json
from pathlib import Path

from reporting import digest, manifest, step_record, write_bundle
from reporting.core import read_jsonl
from reporting.timing import total_timing


def _json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def sisa_run_id(directory):
    directory = Path(directory).resolve()
    config = _json(directory / "config.json", {})
    return config.get("comparison_run_id") or _json(directory / "run.json", {}).get("run_id") or "sisa-" + digest([str(directory), config])[:24]


def export_sisa(directory):
    import numpy as np
    directory = Path(directory).resolve()
    config = _json(directory / "config.json")
    if config is None:
        raise ValueError(f"Missing saved SISA config.json: {directory}")
    split = _json(directory / "dataset_split.json", {})
    workload = _json(directory / "deletion_requests.json", {})
    run_id = sisa_run_id(directory)
    requested = config["deletion_steps"]
    run = manifest(run_id=run_id, method="sisa", dataset=split.get("dataset_id", Path(config["dataset"]).parent.name),
                   model={"vit": "vit_b_16", "roberta": "roberta-base"}.get(config["model"], config["model"]),
                   config={key: value for key, value in config.items() if key not in ("comparison_run_id", "provenance", "hardware")},
                   split_sha256=split.get("manifest_sha256"),
                   workload_sha256=workload.get("workload_sha256"), requested_steps=requested,
                   seed=config.get("seed"), provenance=config.get("provenance", {}),
                   hardware_info=config.get("hardware", {}), precision="fp32",
                   measurement_origin="native" if config.get("comparison_run_id") else "historical_export")
    evaluations = read_jsonl(directory / "evaluations.jsonl")
    # Native SISA CSV accuracy is defined on [0, 1] and the test pool. Keep
    # older scalar measurements useful without inventing predictions or F1.
    if (directory / "report.csv").exists():
        import csv
        with (directory / "report.csv").open() as stream:
            for row in csv.DictReader(stream):
                label = int(row["nb_requests"])
                if not any(item["state_id"] == f"step-{label}" and item["split"] == "test" for item in evaluations):
                    evaluations.append({"schema_version": 1, "run_id": run_id, "state_id": f"step-{label}",
                                        "step": label, "split": "test", "status": "ready",
                                        "evaluation_sha256": None, "identity_status": "unverified",
                                        "examples": config.get("dataset_metadata", {}).get("nb_test"),
                                        "metrics": {"accuracy": float(row["accuracy"])}, "confusion_matrix": None,
                                        "evaluation_time_sec": None, "method_details": {"measurement_origin": "historical_scalar"}})
    steps, previous = [], set()
    for label in range(requested + 1):
        path = directory / f"requestfile:{label}.npy"
        if not path.exists():
            break
        deleted = {int(value) for shard in np.load(path, allow_pickle=True) for value in shard}
        if workload and label:
            source_ids = workload["batches"][label - 1]
        else:
            source_ids = None  # Independent random labels are not an ordered portable workload.
        timings = [_json(directory / "times" / f"shard-{index}:{label}.json") for index in range(config["shards"])]
        complete = all(value and value.get("status") == "complete" and value.get("measurement_complete")
                       and value.get("timing_scope") == "update_v1" for value in timings)
        evaluated = any(item["step"] == label for item in evaluations)
        if not complete and not evaluated:
            break
        legacy = not any(timings)
        steps.append(step_record(run_id, label, deleted_count=len(deleted) - len(previous), deleted_total=len(deleted),
                                 source_ids=source_ids, status="ready" if complete or evaluated else "incomplete",
                                 update_time_sec=sum(item["update_time_sec"] for item in timings) if complete else None,
                                 training_time_sec=sum(item["training_time_sec"] for item in timings) if complete else None,
                                 timing_scope="legacy_batch_last_slice" if legacy else "update_v1",
                                 method_details={"shards": config["shards"],
                                                 "request_mode": "ordered_workload" if workload else "independent_deletion_states",
                                                 "shard_timings": timings}))
        previous = deleted
    write_bundle(directory, run, steps, evaluations)
    return directory


def export_s3t(directory):
    directory = Path(directory).resolve()
    config = _json(directory / "config.json")
    meta = _json(directory / "metadata.json")
    state = _json(directory / "state.json")
    split = _json(directory / "dataset_split.json")
    workload = _json(directory / "deletion_requests.json", {})
    runtime = _json(directory / "comparison_runtime.json", {})
    run_id = "s3t-" + meta["run_sha256"][:24]
    run = manifest(run_id=run_id, method="s3t", dataset=split["dataset_id"],
                   model={"roberta": "roberta-base"}.get(config["model"], config["model"]),
                   config=config, seed=config["seed"], split_sha256=split["manifest_sha256"],
                   workload_sha256=state["workload_sha256"],
                   requested_steps=len(workload["batches"]) if workload else state["step"],
                   provenance=meta["provenance"], hardware_info=runtime.get("hardware", meta.get("hardware", {})), precision=config["precision"],
                   measurement_origin="native" if meta.get("comparison_schema_version") else "historical_export")
    def member_timings(generation):
        return [directory / "comparison_timing" / f"g{generation}-s{shard}-q{sequence}.json"
                for shard in range(config["shards"]) for sequence in range(config["budget"])]
    initial = total_timing(member_timings(0) + list((directory / "comparison_timing").glob("initial-*.json")))
    steps = [step_record(run_id, 0, status="ready" if initial["measurement_complete"] else "incomplete",
                         **{key: initial[key] for key in ("update_time_sec", "training_time_sec", "timing_scope")})]
    # Older native evaluations are imported conservatively; no new timing is inferred.
    evaluations = read_jsonl(directory / "evaluations.jsonl")
    for path in sorted((directory / "evaluations").glob("*.json")):
        saved = _json(path)
        if saved.get("comparison_evaluation"):
            value = saved["comparison_evaluation"]
        else:
            value = {"schema_version": 1, "run_id": run_id, "state_id": f"g{saved['generation']}-step-{saved['step']}",
                     "step": saved["step"], "split": saved["split"], "status": saved["status"],
                     "evaluation_sha256": None, "identity_status": "unverified", "examples": saved["examples"],
                     "metrics": {"accuracy": saved["accuracy"]} if saved["accuracy"] is not None else {},
                     "evaluation_time_sec": saved["evaluation_seconds"], "confusion_matrix": None}
        evaluations = [item for item in evaluations if (item["state_id"], item["split"]) != (value["state_id"], value["split"])] + [value]
    steps[0]["state_id"] = "g0-step-0"
    if any(item["step"] == 0 and item["status"] == "ready" for item in evaluations):
        steps[0]["status"] = "ready"
    for event in state["history"]:
        index, generation = event["step"], event["generation"]
        paths = [directory / "comparison_timing" / f"delete-{index}.json"]
        for metadata in sorted((directory / "generations").glob("*/generation.json")):
            value = _json(metadata)
            if value["generation"] and value["parent_step"] == index:
                generation = max(generation, value["generation"])
                paths.extend(member_timings(value["generation"]))
                paths.append(directory / "comparison_timing" / f"publish-{value['generation']}.json")
        timing = total_timing(paths)
        status = "ready" if generation != event["generation"] else event["status"]
        if generation != event["generation"] and state["generation"] < generation:
            status = "incomplete"
        steps.append(step_record(run_id, index, deleted_count=len(event["source_ids"]),
                                 deleted_total=len(event["cumulative_source_ids"]), source_ids=event["source_ids"],
                                 state_id=f"g{generation}-step-{index}", status=status,
                                 **{key: timing[key] for key in ("update_time_sec", "training_time_sec", "timing_scope")},
                                 method_details={"generation": generation, "selected": event["selected"],
                                                 "planning_time_sec": event["planning_seconds"],
                                                 "measurement_complete": timing["measurement_complete"]}))
    write_bundle(directory, run, steps, evaluations)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("sisa", "s3t"))
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for directory in args.runs:
        print((export_sisa if args.method == "sisa" else export_s3t)(directory))


if __name__ == "__main__":
    main()
