"""Run the SISA recipes owned by an individual study."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shlex

from .artifacts import ROOT, create_sisa_records, execute_plan


def build_plan(settings, mode, shards, requests, tag, study, root=ROOT, device=None):
    root, study = Path(root), Path(study)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Tag must contain only letters, numbers, underscores, or hyphens")
    if requests < 0 or shards is not None and shards < 1:
        raise ValueError("Requests must be nonnegative and shards must be positive")
    recipe = {**settings["defaults"], **settings["modes"][mode]}
    counts = recipe.get("shard_grid", [shards])
    if any(count is None for count in counts):
        raise ValueError("This recipe requires --shards")
    rates = recipe.get("learning_rate_grid", [recipe["learning_rate"]])
    output = study / "runs" / tag
    if output.exists():
        raise ValueError("Study run exists; choose a new tag")
    dataset = root / settings["dataset"]
    metadata = json.loads(dataset.read_text())
    commands = []
    for count in counts:
        for rate in rates:
            name = f"shards-{count}-lr-{rate:g}"
            container = output / name
            if container.exists():
                raise ValueError(f"SISA records already exist: {container}")
            argv = ["bash", "example-scripts/common/run_all.sh", settings["model"], str(dataset),
                    str(count), str(requests), str(container), str(recipe["slices"]), str(recipe["epochs"]),
                    str(recipe["batch_size"]), str(rate), recipe["optimizer"], str(recipe["chkpt_interval"]),
                    str(recipe["predict_batch_size"]), str(output / "reports" / name)]
            if "dropout_rate" in recipe:
                argv.append(str(recipe["dropout_rate"]))
            environment = {}
            if device is not None:
                if device == "cpu":
                    environment["CUDA_VISIBLE_DEVICES"] = ""
                elif re.fullmatch(r"(?:cuda:)?[0-9]+", device):
                    environment["CUDA_VISIBLE_DEVICES"] = device.removeprefix("cuda:")
                else:
                    raise ValueError("Device must be cpu, a GPU index, or cuda:<index>")
            commands.append({"argv": argv, "env": environment, "container": str(container),
                             "records": str(output / name),
                             "resolved_config": {**recipe, "shards": count, "learning_rate": rate,
                                                 "requests": requests, "dataset_metadata": metadata}})
    return commands


def main(study, argv=None):
    study = Path(study).resolve()
    settings = json.loads((study / "configs/recipes.json").read_text())
    parser = argparse.ArgumentParser(description=settings["description"])
    parser.add_argument("--mode", choices=tuple(settings["modes"]), default="single")
    parser.add_argument("--shards", type=int)
    parser.add_argument("--requests", type=int, default=0)
    parser.add_argument("--device", help="cpu or GPU index; omitted to preserve the current CUDA environment")
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S%fZ"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        commands = build_plan(settings, args.mode, args.shards, args.requests, args.tag, study, ROOT, args.device)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for command in commands:
        environment = " ".join(f"{key}={shlex.quote(value)}" for key, value in command["env"].items())
        print(f"{environment} {shlex.join(command['argv'])}".strip(), flush=True)
    if args.dry_run:
        return 0
    output = study / "runs" / args.tag
    def prepare():
        (output / "reports").mkdir()
        for command in commands:
            create_sisa_records(command["records"], ROOT)
    return execute_plan(output, study, commands, {"commands": commands, "recipe": settings, "mode": args.mode},
                        ROOT, prepare=prepare)
