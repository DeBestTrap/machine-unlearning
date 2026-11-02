#!/usr/bin/env python3
"""Monitor real ProtoSISA trials on a fixed 10% CIFAR-10 training sample."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import protosisa as proto
from container_storage import initialize_container, checkpoint_summary
from dataset_loading import load_dataset_module
from dataset_splits import load_manifest, split_indices
from reporting.core import hardware, runtime_provenance, write_json
from studies.artifacts import capture_run


def prepare_selection(group):
    path = group / "selection.json"
    metadata = ROOT / "datasets/cifar10/datasetfile"
    info = json.loads(metadata.read_text())
    loader = load_dataset_module(metadata, info)
    split = load_manifest(ROOT / "splits/cifar10-val2500-seed42.json")
    source_labels = np.asarray(loader.train_set.targets)
    train_labels = source_labels[loader.train_indices]
    val_labels = source_labels[loader.val_indices]
    _, selected = split_indices(train_labels, 4750, seed=42, stratify=True)
    confirmation, scoring = split_indices(val_labels, 1000, seed=42, stratify=True)
    selection = dict(dataset="cifar10", seed=42, training_fraction=0.1,
        split_sha256=split["manifest_sha256"], train_local_indices=selected,
        train_source_indices=[int(loader.train_indices[i]) for i in selected],
        scoring_val_indices=scoring, confirmation_val_indices=confirmation,
        train_class_counts=np.bincount(train_labels[selected], minlength=10).tolist())
    if path.exists():
        if json.loads(path.read_text()) != selection:
            raise ValueError("Saved training selection changed")
    else:
        write_json(path, selection)
    return selection, loader, info, split


def diagnose(directory, selection):
    arrays = np.load(directory / "outputs/components-step-0-val.npz")
    chosen = np.asarray(selection["scoring_val_indices"])
    labels = arrays["labels"][chosen]
    ensemble, prototype = arrays["ensemble"][chosen], arrays["prototype"][chosen]
    accuracy = lambda scores: float(np.mean(scores.argmax(1) == labels))
    config = json.loads((directory / "config.json").read_text())
    mixture = proto.mixture_parameters(config, len(selection["train_local_indices"]), config["shards"])
    scores = proto.mix_logits(torch.from_numpy(ensemble), torch.from_numpy(prototype),
                              mixture["prototype_weight"]).numpy()
    result = dict(scoring_examples=len(chosen), sisa_accuracy=accuracy(ensemble),
        prototype_accuracy=accuracy(prototype), protosisa_accuracy=accuracy(scores), **mixture,
        ensemble_std=float(np.std(ensemble - ensemble.mean(1, keepdims=True))),
        prototype_std=float(np.std(prototype - prototype.mean(1, keepdims=True))))
    write_json(directory / "scoring.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="cifar10_10pct_v1")
    parser.add_argument("--trial", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--delete-count", type=int, default=0)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        from studies.protosisa_pilot.finalize import main as finalize
        return finalize(["--group", args.group, "--trial", args.trial, "--device", args.device])
    os.chdir(ROOT)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU experiment")
    print(f"device={device}; name={torch.cuda.get_device_name(device)}", flush=True)
    group = STUDY / "runs" / args.group
    group.mkdir(parents=True, exist_ok=True)
    selection, loader, info, split = prepare_selection(group)
    directory = group / args.trial
    settings = dict(epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                    subset_sha256=proto.digest(selection), device=str(device), delete_count=args.delete_count,
                    prototype_weighting="safe")
    capture_run(directory, STUDY, settings, ROOT)
    initialize_container(str(directory))
    config = dict(model="vit", dataset=str(ROOT / "datasets/cifar10/datasetfile"), dataset_metadata=info,
        seed=1, shards=10, slices=1, epochs=args.epochs, optimizer="adamw", learning_rate=args.learning_rate,
        batch_size=args.batch_size, dropout_rate=0.4, prototype_weighting="safe", prototype_weight=None,
        eval_split="val",
        training_fraction=0.1, selection_sha256=proto.digest(selection), deletion_steps=args.delete_count)
    config["training_sha256"] = proto.digest(config)
    config["comparison_run_id"] = "protosisa-pilot-" + proto.digest([args.group, args.trial])[:24]
    config.update(provenance=runtime_provenance(ROOT), hardware=hardware(device))
    write_json(directory / "config.json", config)
    write_json(directory / "dataset_split.json", split)
    shards = np.empty(10, dtype=object)
    shards[:] = np.array_split(np.random.default_rng(1).permutation(selection["train_local_indices"]), 10)
    np.save(directory / "splitfile.npy", shards, allow_pickle=True)
    started = perf_counter()
    paths = [proto.train_member(directory, config, shard_id, shard, [], 0, loader, device)
             for shard_id, shard in enumerate(shards)]
    proto.seed_all(1)
    encoder = proto.build_encoder("vit", device)
    prototypes = proto.update_prototypes(directory, config, shards, [[] for _ in shards], 0, loader, encoder, device)
    evaluation = proto.evaluate(directory, config, paths, prototypes, 0, loader, encoder, device)
    write_json(directory / "validation.json", evaluation)
    result = diagnose(directory, selection)
    for step in range(1, args.delete_count + 1):
        deletions = selection["train_local_indices"][:step]
        requests = [[int(index) for index in shard if index in deletions] for shard in shards]
        paths = [proto.train_member(directory, config, shard_id, shard, deleted, step, loader, device)
                 for shard_id, (shard, deleted) in enumerate(zip(shards, requests))]
        prototypes = proto.update_prototypes(directory, config, shards, requests, step, loader, encoder, device)
        value = proto.evaluate(directory, config, paths, prototypes, step, loader, encoder, device)
        write_json(directory / f"validation-step-{step}.json", value)
    checkpoint_summary(str(directory))
    result.update(elapsed_sec=perf_counter() - started, completed_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / "result.json", result)


if __name__ == "__main__":
    main()
