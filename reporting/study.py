"""Study-local interface for regenerating comparison reports."""
import argparse
import json
from pathlib import Path
import re

from .reports import refresh


def report_settings(study):
    path = Path(study) / "configs/report.json"
    return json.loads(path.read_text()) if path.exists() else {}


def main(study, argv=None, *, publish=None):
    study = Path(study).resolve()
    parser = argparse.ArgumentParser(description="Rebuild this study's comparison tables and figures; no training")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--include", nargs="+", type=Path, default=[], help="Additional native comparison records, including another repo")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--metric")
    parser.add_argument("--x")
    parser.add_argument("--y")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--strict", action="store_true", default=None)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("Tag must contain only letters, numbers, underscores or hyphens")
    directory = study / "runs" / args.tag
    inputs = ([directory] if directory.exists() else []) + args.include
    previous = directory / "reports/comparison/report_manifest.json"
    if previous.exists():
        inputs.extend(Path(item["directory"]) for item in json.loads(previous.read_text()).get("inputs", []))
    settings = report_settings(study)
    for key in ("split", "metric", "x", "y", "strict"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    if args.no_plots:
        settings["plots"] = False
    try:
        result = refresh(inputs, directory / "reports/comparison", **settings)
        if publish:
            publish(study, directory)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0
