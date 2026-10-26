"""Keep SISA run records locally and disposable checkpoints in saved_models/."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile


def record_directory(container):
    if not container or container in (".", "..") or Path(container).name != container:
        raise ValueError("Container must be a single directory name")
    root = Path("containers")
    if root.is_symlink():
        raise ValueError("containers/ still points to model storage; run python container_storage.py separate first")
    directory = root / container
    if directory.is_symlink():
        raise ValueError(f"Run records must be local: {directory} is a symlink")
    return directory


def initialize_container(container):
    directory = record_directory(container)
    target = (Path("saved_models") / container / "cache").absolute()
    cache = directory / "cache"
    if cache.is_symlink():
        if cache.resolve() != target.resolve():
            raise ValueError(f"Checkpoint cache points to a different model store: {cache}")
    elif cache.exists():
        raise ValueError(f"Checkpoint cache is still a local directory: {cache}")
    if target.resolve().is_relative_to(Path("containers").resolve()):
        raise ValueError("saved_models/ must be outside containers/")
    target.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    if not cache.is_symlink():
        cache.symlink_to(os.path.relpath(target, directory), target_is_directory=True)
    for name in ("times", "outputs"):
        (directory / name).mkdir(exist_ok=True)
    null_time = directory / "times/null.time"
    if not null_time.exists():
        null_time.write_text("0\n")
    return directory


def separate_storage():
    """Copy legacy records locally, retaining every file in the external store."""
    containers = Path("containers")
    models = Path("saved_models")
    if not containers.is_symlink():
        containers.mkdir(exist_ok=True)
        return
    source = containers.resolve(strict=True)
    if models.exists() or models.is_symlink():
        if not models.is_symlink() or models.resolve() != source:
            raise ValueError("saved_models/ already exists and does not point to the legacy model store")
    # Stage copies before changing the old link. Dereference timing/output links
    # so deleting external files cannot invalidate the retained records.
    with tempfile.TemporaryDirectory(prefix=".sisa-storage-", dir=".") as temporary:
        staged = Path(temporary) / "containers"
        staged.mkdir()
        for entry in source.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                raise ValueError(f"Unexpected legacy container entry: {entry}")
            destination = staged / entry.name
            shutil.copytree(entry, destination, ignore=lambda path, names: ["cache"] if Path(path) == entry else [])
            (destination / "cache").symlink_to(entry / "cache", target_is_directory=True)
        if not models.is_symlink():
            models.symlink_to(source, target_is_directory=True)
        original_link = os.readlink(containers)
        containers.unlink()
        try:
            staged.rename(containers)
        except BaseException:
            containers.symlink_to(original_link, target_is_directory=True)
            raise
    for directory in containers.iterdir():
        checkpoint_summary(directory.name)


def checkpoint_summary(container):
    directory = record_directory(container)
    snapshot = directory / "checkpoint_storage.json"
    files = [path for path in (directory / "cache").glob("*.pt") if not path.is_symlink() and path.is_file()]
    # Keep the last measurement after models have been pruned or disconnected.
    if not files and snapshot.exists():
        return json.loads(snapshot.read_text())
    total = sum(path.stat().st_size for path in files)
    value = float(total)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB", "EB"):
        if value < 1024 or unit == "EB":
            human = f"{value:.1f}{unit}"
            break
        value /= 1024
    result = {"checkpoint_files": len(files), "checkpoint_bytes": total,
              "checkpoint_human": human, "captured_at": datetime.now(timezone.utc).isoformat()}
    directory.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(snapshot)
    return result


def clear_models(containers):
    removed = 0
    for container in containers:
        directory = record_directory(container)
        if not directory.is_dir():
            raise ValueError(f"Container does not exist: {directory}")
        checkpoint_summary(container)
        # Unlink checkpoint files and aliases only; never traverse run records.
        for path in (directory / "cache").glob("*.pt"):
            if path.is_file() or path.is_symlink():
                path.unlink()
                removed += 1
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("separate", help="Separate a legacy containers/ symlink without deleting external files")
    for name in ("init", "size"):
        command = commands.add_parser(name)
        command.add_argument("container")
    command = commands.add_parser("clear-models", help="Remove checkpoints, preserving all local run records")
    command.add_argument("containers", nargs="*", help="Container names; omit to clear models for all local containers")
    args = parser.parse_args()
    if args.command == "separate":
        separate_storage()
        print("Run records: containers/; checkpoints: saved_models/<container>/cache/")
    elif args.command == "init":
        initialize_container(args.container)
    elif args.command == "size":
        result = checkpoint_summary(args.container)
        for key in ("checkpoint_files", "checkpoint_bytes", "checkpoint_human"):
            print(f"{key}={result[key]}")
    else:
        # Check the root before enumerating it, including when it is empty.
        record_directory("validation")
        containers = args.containers or sorted(path.name for path in Path("containers").iterdir() if path.is_dir())
        print(f"Removed {clear_models(containers)} checkpoint files/links; retained run records in containers/")


if __name__ == "__main__":
    main()
