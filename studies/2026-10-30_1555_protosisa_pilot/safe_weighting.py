#!/usr/bin/env python3
"""Evaluate SAFE's published ProtoSISA mixture on saved real-model outputs.

Only the inference weighting changes. Reusing the unscaled component outputs
is equivalent to repeating the same backbone inference, without retraining.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STUDY = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import protosisa as proto
from reporting import classification_evaluation
from reporting.core import atomic_text, write_json
from studies.artifacts import capture_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="cifar10_10pct_v1")
    parser.add_argument("--output-name", default="safe_weighting_v1")
    args = parser.parse_args(argv)
    group = STUDY / "runs" / args.group
    selection = json.loads((group / "selection.json").read_text())
    scoring = np.asarray(selection["scoring_val_indices"])
    confirmation = np.asarray(selection["confirmation_val_indices"])
    cases = []
    for trial in ("baseline_1ep", "three_epochs"):
        cases.append((trial, 0, "val", group / trial, group / trial / "validation.json"))
    for split in ("val", "test"):
        parent = group / "three_epochs/finalization_safe"
        if not parent.exists():
            parent = group / "three_epochs/finalization"  # Historical model outputs.
        cases.append(("three_epochs", 1, split, parent, parent / f"{'validation' if split == 'val' else 'test'}-after.json"))
    inputs = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
              for trial, step, split, parent, evaluation in cases
              for path in (parent / "outputs" / f"components-step-{step}-{split}.npz", evaluation)}
    output = group / args.output_name
    settings = dict(prototype_weighting="safe",
        formula="lambda = exp(-mean_retained_shard_size / 100)", shard_graph_degree=1,
        evaluation_backend="saved_unscaled_component_outputs", inputs_sha256=inputs,
        training_selection_sha256=proto.digest(selection),
        scope="SAFE prototype weighting on the existing independent full-finetuning ProtoSISA proxy")
    capture_run(output, STUDY, settings, ROOT)
    results = []
    for trial, step, split, parent, saved_evaluation in cases:
        saved = json.loads(saved_evaluation.read_text())
        arrays = np.load(parent / "outputs" / f"components-step-{step}-{split}.npz")
        ensemble, prototype = torch.from_numpy(arrays["ensemble"]), torch.from_numpy(arrays["prototype"])
        labels = arrays["labels"]
        config = json.loads((group / trial / "config.json").read_text())
        config.update(prototype_weighting="safe", prototype_weight=None)
        config.pop("prototype_temperature", None)  # Historical pilot metadata only.
        count = sum(saved["method_details"]["prototype_counts"])
        mixture = proto.mixture_parameters(config, count, config["shards"])
        scores = proto.mix_logits(ensemble, prototype, mixture["prototype_weight"]).numpy()
        case = f"{trial}-step-{step}-{split}"
        evaluation = classification_evaluation(output, run_id="protosisa-safe-" + proto.digest(settings)[:24],
            state_id=f"{trial}-step-{step}", step=step, split=split, dataset="cifar10",
            labels=labels, predictions=scores.argmax(1), scores=scores, num_classes=10,
            split_sha256=selection["split_sha256"], source_sha256=saved.get("source_sha256"),
            method_details={**mixture, "evaluation_backend": "saved_component_recombination",
                "source_evaluation": str(saved_evaluation.relative_to(ROOT))})
        write_json(output / f"{case}.json", evaluation)
        def accuracy(values, indices):
            return float(np.mean(values[indices].argmax(1) == labels[indices]))
        entry = dict(trial=trial, step=step, split=split, examples=len(labels), **mixture,
            safe_accuracy=float(np.mean(scores.argmax(1) == labels)),
            sisa_accuracy=accuracy(arrays["ensemble"], np.arange(len(labels))),
            prototype_accuracy=accuracy(arrays["prototype"], np.arange(len(labels))))
        if split == "val":
            entry.update(scoring_accuracy=accuracy(scores, scoring), confirmation_accuracy=accuracy(scores, confirmation))
        results.append(entry)
    result = {"settings": settings, "results": results}
    write_json(output / "result.json", result)
    lines = ["# SAFE-weighted ProtoSISA: 10% CIFAR-10", "",
        "Published weighting with unscaled cosine scores.", "",
        "This recombines saved real-model outputs from the existing independent full-finetuning ProtoSISA proxy. It does not reproduce SAFE's InCA adapters or shard-graph training.", "",
        "| Training | State / pool | Examples | Prototype weight | SISA | SAFE-weighted ProtoSISA |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for item in results:
        lines.append(f"| {item['trial']} | step {item['step']} / {item['split']} | {item['examples']} | "
                     f"{item['prototype_weight']:.8f} | {100 * item['sisa_accuracy']:.2f}% | {100 * item['safe_accuracy']:.2f}% |")
    lines += ["", "The per-state weight uses d=1 and mean retained shard size. Before deletion: 4,750 / 10 = 475 examples per shard; after deletion: 4,749 / 10 = 474.9.",
              "", "Source component files and their SHA256 checksums are recorded in result.json and study_run.json. Predictions, scores, labels, and confusion matrices are retained alongside this report.", ""]
    atomic_text(output / "report.md", "\n".join(lines))
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    main()
