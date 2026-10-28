"""Study output paths, launch provenance, and source-safe experiment resume."""
from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def study_directory(slug, root=ROOT):
    matches = sorted(path for path in (Path(root) / "studies").glob(f"????-??-??_????_{slug}") if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one dated study directory for {slug}, found {len(matches)}")
    return matches[0]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".study-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_inventory(root, study):
    root, study = Path(root), Path(study)
    files = set(root.glob("*.py"))
    for name in ("s3t", "architectures", "datasets", "example-scripts", "configs"):
        directory = root / name
        if directory.exists():
            files.update(path for path in directory.rglob("*")
                         if path.is_file() and (path.suffix in (".py", ".sh", ".json", ".yaml", ".yml")
                                                or path.name == "datasetfile") and "__pycache__" not in path.parts)
    files.update(path for path in study.glob("*") if path.suffix in (".py", ".sh"))
    if (study / "configs").exists():
        files.update(path for path in (study / "configs").rglob("*") if path.is_file())
    files.update(path for path in (root / "studies/__init__.py", root / "studies/artifacts.py", root / "studies/sisa.py") if path.exists())
    for path in list(files):
        if path.is_symlink() and path.resolve().is_relative_to(root.resolve()):
            files.add(path.resolve())
    hashes, links = {}, {}
    for path in sorted(files):
        name = str(path.relative_to(root))
        hashes[name] = sha256(path.read_bytes()).hexdigest()
        if path.is_symlink():
            links[name] = os.readlink(path)
    return {"files": hashes, "symlinks": links}


def capture_run(directory, study, settings, root=ROOT):
    """Create immutable launch evidence, or verify it before resuming a study.

    Existing runs without launch evidence require a new output directory. A
    migration record never substitutes today's source for the historical code.
    """
    directory, root, study = Path(directory), Path(root), Path(study)
    path = directory / "study_run.json"
    sources = source_inventory(root, study)
    identity = {"study": str(study.relative_to(root)), "source_sha256": digest(sources),
                "settings_sha256": digest(settings)}
    if path.exists():
        recorded = json.loads(path.read_text())
        if recorded.get("kind") != "launch":
            raise ValueError("This is a historical study run; use a new output directory for current code")
        if any(recorded.get(key) != value for key, value in identity.items()):
            raise ValueError("Study source or settings changed; preserve this run and use a new output directory")
        return recorded
    if directory.exists() and any(path.name not in (".sweep.lock", ".study.lock") for path in directory.iterdir()):
        raise ValueError("Existing output has no study launch provenance; use a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    for name in sources["files"]:
        destination = directory / "launch_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in sources["symlinks"]:
            if not destination.is_symlink():
                destination.symlink_to(sources["symlinks"][name])
        else:
            shutil.copy2(root / name, destination)
    for name, expected in sources["files"].items():
        if sha256((directory / "launch_source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Source changed while capturing the study launch; use a new output directory")
    if source_inventory(root, study) != sources:
        raise ValueError("Source changed while capturing the study launch; use a new output directory")
    packages = {}
    for name in ("torch", "torchvision", "transformers", "numpy", "scipy", "pandas", "datasets", "PyYAML"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=root, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    record = {"schema_version": 1, "kind": "launch", "created_at": datetime.now(timezone.utc).isoformat(),
              **identity, "settings": settings, "sources": sources,
              "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
              "python": platform.python_version(), "python_executable": sys.executable, "packages": packages}
    atomic_json(path, record)
    return record


def verify_run_source(directory, root=ROOT):
    """Guard native training/replay commands as well as the study launchers."""
    root = Path(root).resolve()
    directory = Path(directory).resolve()
    for parent in (directory, *directory.parents):
        marker = parent / "study_run.json"
        if marker.is_file():
            recorded = json.loads(marker.read_text())
            if recorded.get("kind") != "launch":
                raise ValueError("Historical study run: preserve its records and use a fresh run for current code")
            study = root / recorded["study"]
            if digest(source_inventory(root, study)) != recorded["source_sha256"]:
                raise ValueError("Study source changed since launch; restore its launch source or use a new run")
            return
        if parent == root:
            return


def execute_plan(directory, study, commands, settings, root=ROOT, prepare=None):
    directory, root = Path(directory), Path(root)
    capture_run(directory, study, settings, root)
    with (directory / "plan.json").open("x") as stream:
        json.dump(settings, stream, indent=2)
        stream.write("\n")
    if prepare is not None:
        prepare()
    # SISA's shell entry points use `python`; retain the launching environment.
    environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    for index, item in enumerate(commands):
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(item["argv"], cwd=root, env=dict(environment, **item.get("env", {})))
        with (directory / "execution.jsonl").open("a") as stream:
            stream.write(json.dumps({"command": index, "returncode": result.returncode,
                                     "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}) + "\n")
        publish_reports(study, directory)
        if result.returncode:
            return result.returncode
    return 0


def publish_reports(study, directory):
    """Expose a study's saved reports beside its launchers without copying data."""
    study, directory = Path(study).resolve(), Path(directory).resolve()
    if not directory.is_relative_to(study / "runs"):
        return
    sources = {}
    for name in ("report.md", "results.csv", "verification.json", "cuda-resume-verification.json"):
        path = directory / name
        if path.is_file():
            sources[name] = path
    for path in sorted((directory / "reports").glob("*.csv")):
        sources[path.name] = path
    for path in sorted(directory.glob("*/results.csv")):
        sources[f"{path.parent.name}.csv"] = path
    for name, source in sources.items():
        target = study / "reports" / directory.name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Report link points to a different result: {target}")
        elif target.exists():
            raise ValueError(f"Report location already contains data: {target}")
        else:
            target.symlink_to(os.path.relpath(source, target.parent))


def create_sisa_records(destination, root=ROOT):
    """Create SISA records at their canonical study path."""
    root, destination = Path(root).resolve(), Path(destination)
    if (root / "studies").is_symlink() or not destination.resolve().is_relative_to(root / "studies"):
        raise ValueError("Study run records must stay under studies/")
    destination.mkdir(parents=True, exist_ok=False)
