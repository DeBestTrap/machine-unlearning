"""Command-line interface for training, deletion, and replay."""
import argparse
import json
from pathlib import Path

from .artifacts import atomic_json
from .config import Config
from .engine import Run


def _print(value):
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="S3T: sequence-aware sharded sliced training")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "train"):
        item = sub.add_parser(command, help="Initialize a run" if command == "init" else "Train or resume a run")
        item.add_argument("--config", help="JSON or YAML configuration")
        item.add_argument("--output", help="New run directory; identical reruns resume")
        item.add_argument("--run", help="Existing run directory")
        item.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="Override a config value")
        if command == "train":
            item.add_argument("--device", default="auto")
            item.add_argument("--shard", type=int, help="Train only this zero-based shard")
            item.add_argument("--sequence", type=int, help="Train only this zero-based sequence")
            item.add_argument("--max-epochs", type=int, help="Stop after this many additional epochs; resumable")
    for command in ("evaluate", "delete", "replay", "retrain", "status", "export", "audit"):
        item = sub.add_parser(command)
        item.add_argument("--run", required=True)
        if command in ("evaluate", "replay", "retrain"):
            item.add_argument("--device", default="auto")
        if command in ("evaluate", "replay"):
            item.add_argument("--split", choices=("val", "test"), default="test")
            item.add_argument("--no-cache", action="store_true", help="Recompute predictions for timing")
        if command == "evaluate":
            item.add_argument("--step", type=int, help="Evaluate an earlier deletion step")
        if command == "delete":
            item.add_argument("--source-ids", nargs="+", type=int, required=True)
        if command == "replay":
            item.add_argument("--requests", required=True, help="Shared deletion workload JSON")
            item.add_argument("--steps", type=int, help="Replay through this absolute step number")
            item.add_argument("--on-exhaustion", choices=("stop", "continue", "retrain"), default="stop")
    smoke = sub.add_parser("smoke", help="Offline end-to-end training and deletion on a tiny transformer")
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--model", choices=("tiny_vit", "tiny_roberta"), default="tiny_vit")
    args = parser.parse_args(argv)
    if args.command in ("init", "train"):
        if args.run:
            if args.config or args.output or args.set:
                parser.error("Use --run alone to resume, or --config with --output for a configured run")
            run = Run(args.run)
        else:
            if not args.config or not args.output:
                parser.error("Supply --config and --output, or --run")
            values = Config.load(args.config).to_dict()
            for override in args.set:
                if "=" not in override:
                    parser.error("--set requires KEY=JSON, e.g. --set epochs_per_slice=2")
                key, value = override.split("=", 1)
                try:
                    values[key] = json.loads(value)
                except json.JSONDecodeError:
                    values[key] = value
            run = Run.create(args.output, Config.from_dict(values))
        if args.command == "train":
            if args.max_epochs is not None and args.max_epochs < 1:
                parser.error("--max-epochs must be positive")
            _print(run.train(args.device, args.shard, args.sequence, max_epochs=args.max_epochs))
        else:
            _print(run.status())
    elif args.command == "smoke":
        from deletion_requests import generate_workload
        import torch
        torch.set_num_threads(1)
        config = Config(dataset="synthetic", model=args.model, pretrained=False, shards=2, slices=2, budget=2,
                        rank=4, alpha=8, epochs_per_slice=1, learning_rate=0.01, lora_dropout=0.0,
                        batch_size=8, eval_batch_size=16, num_workers=0)
        run = Run.create(args.output, config)
        if run.status()["status"] == "training":
            run.train(args.device)
        requests = Path(args.output) / "smoke_requests.json"
        atomic_json(requests, generate_workload(run.split, seed=123, steps=6))
        results = run.replay(requests, args.device, on_exhaustion="continue")
        _print({"status": run.status(), "final_evaluation": results[-1], "results_csv": run.export_csv()})
    else:
        run = Run(args.run)
        if args.command == "evaluate":
            _print(run.evaluate(args.device, args.split, step=args.step, cache=not args.no_cache))
            run.export_csv()
        elif args.command == "delete":
            _print(run.delete(args.source_ids))
        elif args.command == "replay":
            _print(run.replay(args.requests, args.device, args.split, steps=args.steps, cache=not args.no_cache,
                              on_exhaustion=args.on_exhaustion)[-1])
        elif args.command == "retrain":
            _print(run.retrain(args.device))
        elif args.command == "status":
            _print(run.status())
        elif args.command == "audit":
            _print(run.audit())
        else:
            _print({"results_csv": run.export_csv()})
