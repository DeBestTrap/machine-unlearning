#!/usr/bin/env python3
"""Confirm a frozen pilot candidate, replay a deletion, and score the official test set."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import protosisa as proto
from dataset_loading import load_dataset_module
from reporting.core import write_json
from studies.artifacts import capture_run


def component_metrics(arrays, indices, config, training_examples):
    labels = arrays["labels"][indices]
    ensemble = arrays["ensemble"][indices]
    prototype = arrays["prototype"][indices]
    mixture = proto.mixture_parameters(config, training_examples, config["shards"])
    scores = proto.mix_logits(torch.from_numpy(ensemble), torch.from_numpy(prototype), mixture["prototype_weight"])
    return dict(examples=len(labels),
        sisa_accuracy=float(np.mean(ensemble.argmax(1) == labels)),
        prototype_accuracy=float(np.mean(prototype.argmax(1) == labels)),
        protosisa_accuracy=float(np.mean(scores.numpy().argmax(1) == labels)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="cifar10_10pct_v1")
    parser.add_argument("--trial", required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    group = STUDY / "runs" / args.group
    parent = group / args.trial
    directory = parent / "finalization_safe"
    selection = json.loads((group / "selection.json").read_text())
    config = json.loads((parent / "config.json").read_text())
    config.update(prototype_weighting="safe", prototype_weight=None)
    config.pop("prototype_temperature", None)  # Historical pilot metadata only.
    settings = dict(training_run=str(parent), training_sha256=config["training_sha256"],
        prototype_weighting="safe", deletion_count=1,
        selection_sha256=proto.digest(selection), device=str(device))
    capture_run(directory, STUDY, settings, ROOT)
    confirmation_indices = np.asarray(selection["confirmation_val_indices"])
    arrays = np.load(parent / "outputs/components-step-0-val.npz")
    before = component_metrics(arrays, confirmation_indices, config, len(selection["train_local_indices"]))
    write_json(directory / "confirmation-before.json", before)
    print("Frozen candidate confirmation:", json.dumps(before), flush=True)
    loader = load_dataset_module(config["dataset"], config["dataset_metadata"])
    shards = np.load(parent / "splitfile.npy", allow_pickle=True)
    # One fixed selected source ID; this choice does not depend on any scores.
    deleted = int(selection["train_local_indices"][0])
    requests = [np.intersect1d(shard, [deleted]) for shard in shards]
    affected = [i for i, values in enumerate(requests) if len(values)]
    write_json(directory / "deletion.json", dict(local_id=deleted,
        source_id=int(loader.train_indices[deleted]), affected_shards=affected))
    print(f"Deleting local_id={deleted}; affected_shards={affected}", flush=True)
    paths = [proto.train_member(parent, config, i, shard, values, 1, loader, device)
             for i, (shard, values) in enumerate(zip(shards, requests))]
    proto.seed_all(1)
    encoder = proto.build_encoder("vit", device)
    prototypes = proto.update_prototypes(parent, config, shards, requests, 1, loader, encoder, device)
    if int(prototypes["counts"].sum()) != 4749:
        raise ValueError("Deletion did not remove exactly one prototype contribution")
    validation = proto.evaluate(directory, config, paths, prototypes, 1, loader, encoder, device)
    write_json(directory / "validation-after.json", validation)
    arrays = np.load(directory / "outputs/components-step-1-val.npz")
    after = component_metrics(arrays, confirmation_indices, config, int(prototypes["counts"].sum()))
    print("Post-deletion confirmation:", json.dumps(after), flush=True)
    test_config = dict(config, eval_split="test")
    test = proto.evaluate(directory, test_config, paths, prototypes, 1, loader, encoder, device)
    write_json(directory / "test-after.json", test)
    arrays = np.load(directory / "outputs/components-step-1-test.npz")
    test_metrics = component_metrics(arrays, np.arange(len(arrays["labels"])), config, int(prototypes["counts"].sum()))
    result = dict(settings=settings, confirmation_before=before, confirmation_after=after,
        test_after=test_metrics, prototype_counts_after=prototypes["counts"].tolist(),
        completed_at=datetime.now(timezone.utc).isoformat())
    write_json(directory / "result.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
