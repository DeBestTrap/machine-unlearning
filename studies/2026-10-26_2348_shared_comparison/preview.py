"""Run ViT/CIFAR-10+CIFAR-100 reporting previews across SISA, SU and BM."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from threading import Lock

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(ROOT))

from dataset_splits import load_manifest
from deletion_requests import checksum, load_workload, validate_workload
from reporting.core import write_json
from reporting.reports import refresh
from studies.artifacts import capture_run, create_sisa_records, publish_reports, source_inventory, verify_run_source
from studies.shared_comparison.run import build_plan, load_protocol


def block_sources(root):
    paths = list((root / "lib").glob("*.py")) + list((root / "reporting").glob("*.py"))
    paths += [root / name for name in ("run_sweep.py", "dataset_splits.py", "deletion_requests.py",
                                       "src_experiments/test_vit.py")]
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


def prepare_plan(tag, devices, block_root, requests=None, *, resume=False):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Use a tag containing letters, numbers, underscores or hyphens")
    settings = json.loads((STUDY / "configs/vit_preview.json").read_text())
    if resume:
        saved = json.loads((STUDY / "runs" / tag / "plan.json").read_text())
        saved_devices = [next(item["physical_device"] for item in saved["commands"] if item["dataset"] == name)
                         for name in settings["datasets"]]
        if (saved["preview_settings"] != settings or saved_devices != devices or saved["block_root"] != str(block_root)
                or (requests is not None and requests != saved["protocol"]["steps"])):
            raise ValueError("Resume arguments/settings differ from the original preview")
        return saved
    protocol = load_protocol(STUDY / "configs/quick25.json", ROOT)
    count = settings["requests"] if requests is None else requests
    if type(count) is not int or not 1 <= count <= protocol["steps"]:
        raise ValueError("Requests must be between 1 and 25")
    protocol.update(name="vit_reporting_preview", steps=count, epochs=settings["epochs"],
                    training_seed=settings["training_seed"], sisa=settings["sisa"])
    protocol["datasets"] = {name: protocol["datasets"][name] for name in settings["datasets"]}
    output = STUDY / "runs" / tag
    block_output = block_root / "studies" / STUDY.name / "runs" / tag
    workloads, commands = {}, []
    for name, device in zip(settings["datasets"], devices):
        entry = protocol["datasets"][name]
        split = load_manifest(ROOT / "splits" / entry["split"])
        block_split = load_manifest(block_root / "splits" / entry["split"])
        if block_split != split:
            raise ValueError(f"{name}: repositories have different split manifests")
        workload = load_workload(ROOT / "deletions" / entry["workload"], split)
        workload["batches"] = workload["batches"][:count]
        workload["workload_sha256"] = checksum(workload)
        workloads[name] = validate_workload(workload, split)
        entry["workload"] = str(output / "inputs" / f"{name}-requests.json")
        entry["workload_sha256"] = workload["workload_sha256"]
        # Each process sees only its assigned physical GPU, addressed as cuda:0.
        sisa = build_plan(protocol, ROOT, ["sisa"], [name], tag, "0")[0]
        sisa.update(cwd=str(ROOT), experiment=None, physical_device=device)
        sisa["env"]["CUDA_VISIBLE_DEVICES"] = device
        sisa["env"]["SISA_SEED"] = str(settings["training_seed"])
        commands.append(sisa)
        for item in build_plan(protocol, block_root, ["fc", "fcm"], [name], tag, "0"):
            method = {"fc": "su", "fcm": "bm"}[item["method"]]
            experiment = f"ViTPreview_{tag}_{method}_{name}"
            argv = item["argv"]
            for index, argument in enumerate(argv):
                if argument.startswith("experiment_name="):
                    argv[index] = "experiment_name=" + json.dumps(experiment)
            argv[argv.index("--workdir") + 1] = str(block_output / "sweeps" / method / name)
            argv.extend(["--set", "train_args.future_group_merge_mode=" + json.dumps(
                "joint_warmup" if method == "su" else "naive_weight_average")])
            item.update(method=method, cwd=str(block_root), experiment=experiment, physical_device=device)
            item["env"]["CUDA_VISIBLE_DEVICES"] = device
            commands.append(item)
    return dict(tag=tag, preview_settings=settings, protocol=protocol, workloads=workloads,
                commands=commands, block_root=str(block_root), block_output=str(block_output),
                block_sources=block_sources(block_root))


def render_reports(output, results, block_output):
    inputs = [Path(item["records"]) for item in results]
    if not inputs:
        return
    for metric in ("accuracy", "macro_f1"):
        name = "comparison" if metric == "accuracy" else "macro_f1-comparison"
        refresh(inputs, output / "reports" / name, metric=metric)
        refresh(inputs, block_output / "reports" / name, metric=metric)
    # The report command can subsequently regenerate this combined view from either repo.
    write_json(output / "results.json", results)
    write_json(block_output / "results.json", results)
    publish_reports(STUDY, output)
    import csv
    with (output / "reports/comparison/summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    lines = ["# ViT reporting preview", "", "One epoch, three shared deletions per dataset, one seed, full training/test pools.",
             "This previews the reporting pipeline; it is not a convergence benchmark.", "",
             "| Dataset | Method | Initial test accuracy | Final test accuracy | Mean update (s) | Initial training (s) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        def value(key, percent=False):
            return f"{float(row[key]) * (100 if percent else 1):.2f}" + ("%" if percent else "") if row[key] else "unavailable"
        lines.append(f"| {row['dataset']} | {row['method'].upper()} | {value('initial_metric', True)} | "
                     f"{value('final_metric', True)} | {value('mean_update_time_sec')} | {value('initial_train_time_sec')} |")
    lines += ["", "[Accuracy tables and figures](reports/comparison/README.md)", "",
              "[Macro F1 tables and figures](reports/macro_f1-comparison/README.md)", ""]
    (output / "report.md").write_text("\n".join(lines))
    publish_reports(STUDY, output)


def verify_preview_source(output, *, resume=False):
    if not resume:
        verify_run_source(output, ROOT)
        return
    recorded = json.loads((output / "study_run.json").read_text())["sources"]
    current = source_inventory(ROOT, STUDY)
    changed = {name for name in set(recorded["files"]) | set(current["files"])
               if recorded["files"].get(name) != current["files"].get(name)}
    # Resume can repair orchestration/report publication without changing any
    # code that trained the retained SISA models or produced their predictions.
    allowed = {str(Path(__file__).resolve().relative_to(ROOT))}
    if changed - allowed or recorded["symlinks"] != current["symlinks"]:
        raise ValueError("Training/evaluation source changed; preserve this run and use a fresh preview")
    write_json(output / "continuation.json", dict(
        resumed_at=datetime.now(timezone.utc).isoformat(), changed_orchestration_files=sorted(changed),
        note="Original launch/training evidence retained; completed SISA states are reused without retraining",
        current_driver_sha256=current["files"][next(iter(allowed))]))


def run(plan, *, resume=False):
    import torch
    devices = {item["physical_device"] for item in plan["commands"]}
    if not torch.cuda.is_available() or any(int(value) >= torch.cuda.device_count() for value in devices):
        raise ValueError("Requested CUDA devices are unavailable; this study requires GPU access")
    output = STUDY / "runs" / plan["tag"]
    block_root, block_output = Path(plan["block_root"]), Path(plan["block_output"])
    if resume:
        if not (output / "plan.json").exists() or json.loads((output / "plan.json").read_text()) != plan:
            raise ValueError("Resume settings differ from the saved plan")
        verify_preview_source(output, resume=True)
    else:
        if output.exists() or block_output.exists():
            raise ValueError("Preview output exists; use a fresh tag to preserve prior runs")
        capture_run(output, STUDY, plan, ROOT)
        write_json(output / "plan.json", plan)
        write_json(block_output / "plan.json", plan)
        for name, workload in plan["workloads"].items():
            write_json(output / "inputs" / f"{name}-requests.json", workload)
            create_sisa_records(output / f"sisa-{name}", ROOT)
        for name in plan["block_sources"]:
            target = output / "block_source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(block_root / name, target)
        write_json(output / "block_source.json", plan["block_sources"])
    (output / "reports").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    results = json.loads((output / "results.json").read_text()) if (output / "results.json").exists() else []
    lock = Lock()
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                       MPLCONFIGDIR=str(output / "runtime_cache/matplotlib"))

    def dataset_worker(dataset):
        for item in (item for item in plan["commands"] if item["dataset"] == dataset):
            if any(result["method"] == item["method"] and result["dataset"] == dataset for result in results):
                print(f"Reusing completed {item['method']}-{dataset}", flush=True)
                continue
            verify_preview_source(output, resume=resume)
            if block_sources(block_root) != plan["block_sources"]:
                raise ValueError("BlockUnlearn source changed since launch; use a fresh preview")
            started = datetime.now(timezone.utc).isoformat()
            label = f"{item['method']}-{dataset}"
            print(f"Starting {label} on GPU {item['physical_device']}", flush=True)
            argv = item["argv"]
            if resume and item["method"] == "sisa":
                from sisa_timing import summarize
                records = output / f"sisa-{dataset}"
                config = json.loads((records / "config.json").read_text())
                for step in range(plan["protocol"]["steps"] + 1):
                    summarize(str(records), step, config["shards"])
                    for shard in range(config["shards"]):
                        if not (records / "outputs" / f"shard-{shard}:{step}.npy").is_file():
                            raise ValueError("Resume requires complete retained SISA predictions")
                argv = ["bash", "example-scripts/common/data.sh", config["dataset"], str(config["shards"]),
                        str(config["deletion_steps"]), str(records), str(output / "reports" / label),
                        json.dumps(config["hyperparams"], separators=(",", ":"))]
                print(f"Reusing all trained {label} states and predictions; regenerating reports only", flush=True)
            log = output / "logs" / f"{label}{'-resume' if resume else ''}.log"
            with log.open("w") as stream:
                process = subprocess.run(argv, cwd=item["cwd"], env=dict(environment, **item["env"]),
                                         stdout=stream, stderr=subprocess.STDOUT)
            event = dict(method=item["method"], dataset=dataset, returncode=process.returncode,
                         started_at=started, finished_at=datetime.now(timezone.utc).isoformat(), log=str(log))
            with lock:
                for directory in (output, block_output):
                    with (directory / "execution.jsonl").open("a") as stream:
                        stream.write(json.dumps(event) + "\n")
            if process.returncode:
                raise RuntimeError(f"{label} failed ({process.returncode}); see {log}")
            if item["method"] == "sisa":
                records = output / f"sisa-{dataset}"
            else:
                matches = list((block_root / "experiments/VIT" / item["experiment"]).glob("*/comparison/repeat-000/run.json"))
                if len(matches) != 1:
                    raise ValueError(f"Expected one exported {label} run, found {len(matches)}")
                records = matches[0].parent
            with lock:
                results.append(dict(method=item["method"], dataset=dataset, records=str(records)))
                render_reports(output, results, block_output)
            print(f"Finished {label}; combined reports refreshed ({len(results)}/6)", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dataset_worker, dataset) for dataset in plan["protocol"]["datasets"]]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
                print(str(error), file=sys.stderr, flush=True)
    write_json(output / "completion.json", dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                                completed_runs=len(results), failures=failures))
    print(f"Preview finished: {len(results)}/6 runs; report: {output / 'report.md'}", flush=True)
    return int(bool(failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="vit_preview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--devices", nargs=2, default=["0", "1"], metavar=("CIFAR10_GPU", "CIFAR100_GPU"))
    parser.add_argument("--block-root", type=Path, default=ROOT.parent / "BlockUnlearn/unlearning")
    parser.add_argument("--requests", type=int, help="Override the preview's three deletion requests")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed trials and retained SISA predictions")
    args = parser.parse_args(argv)
    if any(not value.isdigit() for value in args.devices) or len(set(args.devices)) != 2:
        parser.error("Assign two different numeric GPU indices")
    try:
        plan = prepare_plan(args.tag, args.devices, args.block_root.resolve(), args.requests, resume=args.resume)
        print(f"Tag: {args.tag}; six runs, {plan['protocol']['epochs']} epoch, {plan['protocol']['steps']} deletions/dataset", flush=True)
        for item in plan["commands"]:
            print(f"[{item['method']}/{item['dataset']}, GPU {item['physical_device']}] {shlex.join(item['argv'])}", flush=True)
        return 0 if args.dry_run else run(plan, resume=args.resume)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
