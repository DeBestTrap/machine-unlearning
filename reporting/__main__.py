"""python -m reporting refresh INPUT... --output STUDY/reports/TAG"""
import argparse
import json
from pathlib import Path

from .reports import refresh


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("refresh", help="Regenerate tables and plots without datasets or checkpoints")
    command.add_argument("inputs", nargs="+", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command.add_argument("--settings", type=Path, help="Saved report_settings.json or a custom settings object")
    command.add_argument("--split", choices=("val", "test"))
    command.add_argument("--metric")
    command.add_argument("--x", help="summary.csv column for the comparison plot's horizontal axis")
    command.add_argument("--y", help="summary.csv column for the comparison plot's vertical axis")
    command.add_argument("--no-plots", action="store_true", default=None)
    command.add_argument("--strict", action="store_true", default=None)
    args = parser.parse_args(argv)
    try:
        settings = json.loads(args.settings.read_text()) if args.settings else {}
        if not isinstance(settings, dict) or set(settings) - {"split", "metric", "x", "y", "plots", "strict"}:
            raise ValueError("Unsupported report settings")
        for key in ("split", "metric", "x", "y", "strict"):
            if getattr(args, key) is not None:
                settings[key] = getattr(args, key)
        if args.no_plots:
            settings["plots"] = False
        result = refresh(args.inputs, args.output, **settings)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
