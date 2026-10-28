"""Move completed experiment records into canonical study locations."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from studies.artifacts import atomic_json


SHARED = "studies/2026-10-26_2348_shared_comparison"
LR = "studies/2026-10-26_1332_s3t_lr_screening"
VALIDATION = "studies/2026-10-26_0948_s3t_validation"


def mappings():
    result = [("runs/shared_comparison", f"{SHARED}/runs"),
              ("runs/s3t-lr-bms-first-slice-20261026", f"{LR}/runs/s3t-lr-bms-first-slice-20261026"),
              ("runs/s3t-lr-bms-first-slice-cifar100-20261026", f"{LR}/runs/s3t-lr-bms-first-slice-cifar100-20261026"),
              ("runs/s3t-validation", f"{VALIDATION}/runs/validation_20261026")]
    for dataset in ("cifar10", "cifar100", "sst2"):
        result.extend([(f"containers/comparison_v1_sisa_{dataset}",
                        f"{SHARED}/runs/comparison_v1/sisa-{dataset}"),
                       (f"report-shared25-comparison_v1-{dataset}.csv",
                        f"{SHARED}/runs/comparison_v1/reports/sisa-{dataset}.csv")])
    for old, study in (
        ("report-purchase.csv", "2026-01-09_1800_sisa_purchase"),
        ("report-roberta.csv", "2026-01-09_1605_sisa_roberta_sst2"),
        ("report-vit.csv", "2026-01-09_1617_sisa_vit_cifar10"),
        ("r-vit.csv", "2026-01-09_1617_sisa_vit_cifar10"),
    ):
        result.append((old, f"studies/{study}/reports/historical/{old}"))
    return result


def inventory(path):
    path = Path(path)
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    result = {}
    for item in files:
        if item.is_symlink() or not item.is_file():
            continue
        stat = item.stat()
        record = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
        if item.suffix in (".json", ".jsonl", ".csv", ".md", ".py", ".sh", ".time", ".log"):
            record["sha256"] = sha256(item.read_bytes()).hexdigest()
        result[str(item.relative_to(path)) if item != path else "."] = record
    return result


def archive_record(path, old_locations):
    marker = path / "study_run.json"
    if marker.exists():
        return
    snapshots = sorted(str(item.relative_to(path)) for item in path.rglob("source_snapshot") if item.is_dir())
    commits = set()
    for item in path.rglob("metadata.json"):
        metadata = json.loads(item.read_text())
        commit = metadata.get("provenance", {}).get("git_commit")
        if commit:
            commits.add(commit)
    atomic_json(marker, {"schema_version": 1, "kind": "historical_import", "compatibility": "unverified",
                         "migrated_at": datetime.now(timezone.utc).isoformat(), "original_locations": old_locations,
                         "recorded_git_commits": sorted(commits), "source_snapshots": snapshots,
                         "note": "Original records are preserved. Current source is not asserted to have produced them; use a fresh run for current code."})


def migrate(root=ROOT, apply=False):
    root = Path(root).resolve()
    pending = []
    old_links = []
    for old, new in mappings():
        source, destination = root / old, root / new
        if source.is_symlink():
            if source.resolve() != destination.resolve():
                raise ValueError(f"Unexpected compatibility link: {source}")
            old_links.append((source, destination))
            continue
        if not source.exists():
            continue
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"Migration destination exists: {destination}")
        pending.append((source, destination))
    if not apply:
        return [{"old": str(old.relative_to(root)), "new": str(new.relative_to(root))} for old, new in pending + old_links]
    evidence = []
    with ExitStack() as stack:
        # Hold existing locks while renaming; an active training process blocks
        # migration, and a concurrent invocation cannot acquire these same locks.
        for source, _ in pending + old_links:
            if source.is_dir():
                for lock in sorted(source.rglob("*.lock")):
                    if lock.is_file() and not lock.is_symlink():
                        stream = stack.enter_context(lock.open("r"))
                        try:
                            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError as exc:
                            raise ValueError(f"An active process holds {lock}; leave this run in place") from exc
        for source, destination in old_links:
            source.unlink()
            evidence.append({"old": str(source.relative_to(root)), "new": str(destination.relative_to(root)),
                             "action": "remove_old_link", "files": {}})
        for source, destination in pending:
            before = inventory(source)
            cache = source / "cache" if source.is_dir() else None
            original_link = os.readlink(cache) if cache is not None and cache.is_symlink() else None
            checkpoint_target = cache.resolve() if original_link is not None else None
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            try:
                if original_link is not None:
                    (destination / "cache").unlink()
                    (destination / "cache").symlink_to(os.path.relpath(checkpoint_target, destination), target_is_directory=True)
                if inventory(destination) != before:
                    raise RuntimeError(f"Migrated file identity/content changed: {destination}")
            except BaseException:
                if original_link is not None:
                    (destination / "cache").unlink(missing_ok=True)
                    (destination / "cache").symlink_to(original_link, target_is_directory=True)
                destination.rename(source)
                raise
            evidence.append({"old": str(source.relative_to(root)), "new": str(destination.relative_to(root)),
                             "files": before, "checkpoint_target": str(checkpoint_target) if checkpoint_target else None})
    if evidence:
        log = root / "studies/.migration" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ.json")
        atomic_json(log, {"moved": evidence})
    for path, aliases in (
        (root / SHARED / "runs/comparison_v1", ["runs/shared_comparison/comparison_v1", "containers/comparison_v1_sisa_*", "report-shared25-comparison_v1-*.csv"]),
        (root / LR / "runs/s3t-lr-bms-first-slice-20261026", ["runs/s3t-lr-bms-first-slice-20261026"]),
        (root / LR / "runs/s3t-lr-bms-first-slice-cifar100-20261026", ["runs/s3t-lr-bms-first-slice-cifar100-20261026"]),
        (root / VALIDATION / "runs/validation_20261026", ["runs/s3t-validation"]),
    ):
        if path.is_dir():
            archive_record(path, aliases)
    return evidence


def remove_source_links(root=ROOT, apply=False):
    root = Path(root).resolve()
    links = [
        ("run_shared_comparison.py", f"{SHARED}/run.py"),
        ("comparison/README.md", f"{SHARED}/README.md"),
        ("comparison/quick25.json", f"{SHARED}/configs/quick25.json"),
        ("docs/s3t-validation.md", f"{VALIDATION}/README.md"),
        ("s3t/lr_search.py", f"{LR}/run.py"),
        ("filter_report_purchase.py", "studies/2026-01-09_1800_sisa_purchase/summarize.py"),
        ("example-scripts/purchase-sharding/README.txt", "studies/2026-01-09_1800_sisa_purchase/README.md"),
    ]
    for directory, study, names in (
        ("purchase-sharding", "2026-01-09_1800_sisa_purchase", ("run.sh", "run_lrs.sh")),
        ("vit-cifar10", "2026-01-09_1617_sisa_vit_cifar10", ("run.sh", "run_shards.sh")),
        ("roberta-sst2", "2026-01-09_1605_sisa_roberta_sst2", ("run.sh", "run_shards.sh", "run_lrs.sh")),
    ):
        links.extend((f"example-scripts/{directory}/{name}", f"studies/{study}/{name}") for name in names)
    removed = []
    for old, new in links:
        path, target = root / old, root / new
        if not path.is_symlink():
            continue
        if path.resolve() != target.resolve():
            raise ValueError(f"Unexpected old source link: {path}")
        if apply:
            path.unlink()
        removed.append({"old": old, "new": new})
    if apply:
        for name in ("comparison", "example-scripts/purchase-sharding", "example-scripts/vit-cifar10",
                     "example-scripts/roberta-sst2", "runs", "containers"):
            directory = root / name
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the displayed moves; otherwise only preview")
    args = parser.parse_args()
    result = migrate(apply=args.apply) + remove_source_links(apply=args.apply)
    for item in result:
        print(f"{item['old']} -> {item['new']}")
    print(f"{'Updated' if args.apply else 'Would update'} {len(result)} paths; only canonical study locations are retained")


if __name__ == "__main__":
    main()
