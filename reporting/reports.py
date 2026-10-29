"""Build disposable tables and figures from saved comparison records only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics

from .core import atomic_text, classification_metrics, digest, load_bundle, write_json


def discover(inputs):
    directories = set()
    for value in inputs:
        path = Path(value).expanduser().resolve()
        if path.is_file() and path.name == "run.json":
            directories.add(path.parent)
        elif (path / "run.json").is_file():
            directories.add(path)
        elif path.is_dir():
            directories.update(p.parent for p in path.rglob("run.json")
                               if not {"launch_source", "source_snapshot"}.intersection(p.parts))
        else:
            raise ValueError(f"Input does not exist: {path}")
    if not directories:
        raise ValueError("No comparison run.json records found; export native runs first")
    return sorted(directories)


def _percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _csv(path, rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


BASE_FIELDS = ["run_id", "method", "dataset", "model", "seed", "config_sha256", "split_sha256",
               "workload_sha256", "execution_mode", "precision", "hardware", "measurement_origin"]
STEP_FIELDS = BASE_FIELDS + ["step", "operation", "request_sha256", "deleted_count", "deleted_total",
                            "state_id", "status", "update_time_sec", "training_time_sec", "timing_scope",
                            "split", "evaluation_sha256", "evaluation_identity_status", "examples",
                            "metric_name", "metric_value", "evaluation_time_sec"]
SUMMARY_FIELDS = BASE_FIELDS + ["split", "metric_name", "timing_scope", "initial_metric", "final_metric",
                               "metric_change", "mean_post_delete_metric", "initial_train_time_sec",
                               "initial_training_time_sec", "requested_steps", "attempted_steps", "ready_steps",
                               "scored_steps", "measured_steps", "final_step", "final_status",
                               "mean_update_time_sec", "median_update_time_sec", "p95_update_time_sec",
                               "total_update_time_sec", "total_training_time_sec"]


def _base(run):
    return {key: json.dumps(run.get(key), sort_keys=True) if isinstance(run.get(key), (dict, list)) else run.get(key)
            for key in BASE_FIELDS}


def build_tables(bundles, *, split="test", metric="accuracy"):
    step_rows, summaries = [], []
    for directory, (run, steps, evaluations) in bundles:
        final_evals = {(value["state_id"], value["split"]): value for value in evaluations}
        base, values = _base(run), {}
        for step in steps:
            evaluation = final_evals.get((step["state_id"], split))
            metrics = dict(evaluation["metrics"]) if evaluation else {}
            if evaluation and evaluation.get("confusion_matrix") is not None:
                metrics.update(classification_metrics(evaluation["confusion_matrix"]))
            value = metrics.get(metric) if evaluation and evaluation["status"] == "ready" else None
            values[step["step"]] = value
            step_rows.append({**base, **{key: step.get(key) for key in STEP_FIELDS if key in step},
                              "split": split, "evaluation_sha256": evaluation.get("evaluation_sha256") if evaluation else None,
                              "evaluation_identity_status": evaluation.get("identity_status") if evaluation else None,
                              "examples": evaluation.get("examples") if evaluation else None,
                              "metric_name": metric, "metric_value": value,
                              "evaluation_time_sec": evaluation.get("evaluation_time_sec") if evaluation else None})
        deletions = [item for item in steps if item["operation"] == "delete"]
        scopes = {item["timing_scope"] for item in deletions}
        if len(scopes) > 1:
            raise ValueError(f"{run['run_id']}: mixed deletion timing scopes; use separate runs")
        measured = [item for item in deletions if item["update_time_sec"] is not None
                    and item["status"] not in ("failed", "incomplete")]
        times = [item["update_time_sec"] for item in measured]
        training = [item["training_time_sec"] for item in deletions
                    if item["training_time_sec"] is not None and item["status"] not in ("failed", "incomplete")]
        first = steps[0] if steps else None
        last = steps[-1] if steps else None
        requested = run.get("requested_steps")
        expected_final = requested if requested is not None else (last["step"] if last else None)
        initial, final = values.get(0), values.get(expected_final)
        scores = [value for index, value in values.items() if index > 0 and value is not None]
        complete = len(deletions) == requested if requested is not None else True
        summaries.append({**base, "split": split, "metric_name": metric,
                          "timing_scope": next(iter(scopes), first["timing_scope"] if first else "unmeasured"),
                          "initial_metric": initial, "final_metric": final,
                          "metric_change": final - initial if final is not None and initial is not None else None,
                          "mean_post_delete_metric": statistics.mean(scores) if scores else None,
                          "initial_train_time_sec": first["update_time_sec"] if first else None,
                          "initial_training_time_sec": first["training_time_sec"] if first else None,
                          "requested_steps": requested, "attempted_steps": len(deletions),
                          "ready_steps": sum(item["status"] == "ready" for item in deletions),
                          "scored_steps": len(scores), "measured_steps": len(times),
                          "final_step": last["step"] if last else None,
                          "final_status": last["status"] if last else "incomplete",
                          "mean_update_time_sec": statistics.mean(times) if times else None,
                          "median_update_time_sec": statistics.median(times) if times else None,
                          "p95_update_time_sec": _percentile(times, .95),
                          "total_update_time_sec": sum(times) if complete and len(times) == len(deletions) else None,
                          "total_training_time_sec": sum(training) if complete and len(training) == len(deletions) else None})
    return step_rows, summaries


def comparison_notes(bundles, split):
    notes = []
    datasets = {run["dataset"] for _, (run, _, _) in bundles}
    for dataset in sorted(datasets):
        items = [(run, steps, evaluations) for _, (run, steps, evaluations) in bundles if run["dataset"] == dataset]
        for field in ("split_sha256", "workload_sha256", "model", "precision", "execution_mode", "hardware"):
            values = {digest(run.get(field)) for run, _, _ in items}
            if len(values) > 1:
                notes.append(f"{dataset}: different {field}; review before comparing")
            if field in ("split_sha256", "workload_sha256") and any(run.get(field) is None for run, _, _ in items):
                notes.append(f"{dataset}: missing {field}; protocol equality is unverified")
        identities = {item.get("evaluation_sha256") for _, _, evs in items for item in evs if item["split"] == split}
        if len(identities) > 1:
            notes.append(f"{dataset}/{split}: different evaluation identities")
        if any(item.get("identity_status") != "verified_source" for _, _, evs in items for item in evs if item["split"] == split):
            notes.append(f"{dataset}/{split}: some evaluation identities bind labels/protocol only; raw input identity is unverified")
        scopes = {step["timing_scope"] for _, steps, _ in items for step in steps if step["step"]}
        if len(scopes) > 1:
            notes.append(f"{dataset}: different timing scopes; timings must not be pooled")
    for _, (run, steps, evs) in bundles:
        if run.get("requested_steps") is not None and len(steps) - 1 < run["requested_steps"]:
            notes.append(f"{run['run_id']}: workload incomplete; final metric and full-workload totals are unavailable")
    return notes


def _plots(output, step_rows, summaries, *, x, y, metric, split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = []
    for index, dataset in enumerate(sorted({row["dataset"] for row in summaries})):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
        rows = [row for row in summaries if row["dataset"] == dataset]
        for row in rows:
            label = f"{row['method']} · {row['run_id'][:12]} · {row['timing_scope']}"
            steps = [step for step in step_rows if step["run_id"] == row["run_id"]]
            # Keep missing points as gaps rather than joining across exhausted requests.
            axes[0].plot([s["step"] for s in steps],
                         [s["metric_value"] if s["metric_value"] is not None else float("nan") for s in steps], label=label)
            deletes = [s for s in steps if s["operation"] == "delete"]
            axes[1].plot([s["step"] for s in deletes],
                         [s["update_time_sec"] if s["update_time_sec"] is not None else float("nan") for s in deletes], label=label)
            if isinstance(row.get(x), (int, float)) and isinstance(row.get(y), (int, float)):
                axes[2].scatter(row[x], row[y], label=label)
        axes[0].set(xlabel="Deletion step (0 = initial)", ylabel=f"{split} {metric}")
        axes[1].set(xlabel="Deletion step", ylabel="Update time (seconds)")
        axes[2].set(xlabel=x, ylabel=y)
        for axis in axes:
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6)
        fig.suptitle(dataset)
        name = f"comparison-{index + 1}"
        for suffix in ("svg", "png"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=160)
            plots.append(path.name)
        plt.close(fig)
    return plots


def refresh(inputs, output, *, split="test", metric="accuracy", x="mean_update_time_sec",
            y="final_metric", plots=True, strict=False):
    if split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test")
    if x not in SUMMARY_FIELDS or y not in SUMMARY_FIELDS:
        raise ValueError("Plot axes must name summary.csv columns")
    directories = discover(inputs)
    output = Path(output).expanduser().resolve()
    if any(output == directory or directory.is_relative_to(output) for directory in directories):
        raise ValueError("Report output must not contain or replace saved source records")
    bundles, ids = [], set()
    for directory in directories:
        bundle = load_bundle(directory)
        if bundle[0]["run_id"] in ids:
            raise ValueError(f"Duplicate run_id: {bundle[0]['run_id']}")
        ids.add(bundle[0]["run_id"])
        bundles.append((directory, bundle))
    notes = comparison_notes(bundles, split)
    if strict and notes:
        raise ValueError("Comparison validation failed: " + "; ".join(notes))
    steps, summaries = build_tables(bundles, split=split, metric=metric)
    if not any(row["metric_value"] is not None for row in steps):
        raise ValueError(f"No saved {split}/{metric} measurements; retain evaluation data or evaluate the models")
    output.mkdir(parents=True, exist_ok=True)
    previous = json.loads((output / "report_manifest.json").read_text()) if (output / "report_manifest.json").exists() else {}
    _csv(output / "steps.csv", steps, STEP_FIELDS)
    _csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    figures = _plots(output, steps, summaries, x=x, y=y, metric=metric, split=split) if plots else []
    for name in set(previous.get("figures", [])) - set(figures):
        if Path(name).name == name and name.startswith("comparison-") and Path(name).suffix in (".png", ".svg"):
            (output / name).unlink(missing_ok=True)
    settings = {"split": split, "metric": metric, "x": x, "y": y, "plots": plots, "strict": strict}
    write_json(output / "report_settings.json", settings)
    write_json(output / "report_manifest.json", {
        "schema_version": 1, "settings": settings, "notes": notes, "figures": figures,
        "inputs": [{"directory": str(directory), "run_id": run["run_id"],
                    "sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in ("run.json", "steps.jsonl", "evaluations.jsonl")}}
                   for directory, (run, _, _) in bundles]})
    atomic_text(output / "README.md", f"# Generated comparison\n\nSplit: {split}; metric: {metric}.\n\n"
                "steps.csv joins each operation with its final-state evaluation. summary.csv has one row per run.\n"
                "Initial training is excluded from deletion statistics. Missing outcomes remain empty; measured_steps,\n"
                "scored_steps and ready_steps show coverage. Totals require a complete workload and all timings.\n\n"
                + ("\n".join(f"- {note}" for note in notes) + "\n" if notes else "Comparison identities match.\n"))
    return {"output": str(output), "runs": len(bundles), "steps": len(steps), "figures": figures, "notes": notes}
