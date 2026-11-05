"""ProtoSISA: independent SISA models plus global frozen-feature prototypes.

See PROTOSISA.md for the prescribed training procedure and SAFE references.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from importlib import import_module
import io
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from container_storage import checkpoint_summary, initialize_container, record_directory
from dataset_loading import load_dataset_module
from deletion_support import export_deletion_requests
from reporting import classification_evaluation, digest, manifest, step_record, write_bundle
from reporting.core import atomic_text, hardware, runtime_provenance, write_json
from reporting.timing import OperationTimer, total_timing
from sisa_timing import UpdateTimer
from sisa_schedule import epochs_per_slice
from sisa_training import synchronized_time
from split_support import bind_container_split, configured_manifest, validate_container_split
from studies.artifacts import verify_run_source


ROOT = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def retained_indices(shard, deleted):
    shard = np.asarray(shard, dtype=np.int64)
    return shard[~np.isin(shard, np.asarray(deleted, dtype=np.int64))]


def batches(loader, indices, batch_size, category="train"):
    for offset in range(0, len(indices), batch_size):
        yield loader.load(indices[offset:offset + batch_size], category=category)


class RobertaFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import RobertaModel
        self.backbone = RobertaModel.from_pretrained("roberta-base", add_pooling_layer=False)

    def forward(self, x):
        mask = (x != self.backbone.config.pad_token_id).long()
        return self.backbone(input_ids=x, attention_mask=mask).last_hidden_state[:, 0]


def build_encoder(model_name, device):
    """A separate pretrained encoder; never load task-trained shard weights."""
    if model_name == "vit":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        encoder.heads = nn.Identity()
    elif model_name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        encoder.fc = nn.Identity()
    elif model_name == "roberta":
        encoder = RobertaFeatures()
    else:
        raise ValueError("ProtoSISA frozen features support vit, resnet50 and roberta")
    return encoder.to(device).requires_grad_(False).eval()


def build_model(config, device):
    module = import_module(f"architectures.{config['model']}")
    info = config["dataset_metadata"]
    return module.Model(tuple(info["input_shape"]), info["nb_classes"],
                        dropout_rate=config["dropout_rate"]).to(device)


def prototype_statistics(encoder, loader, indices, classes, batch_size, device):
    """Sum unit embeddings in float64; prototypes are global, count-weighted means."""
    encoder.requires_grad_(False).eval()
    sums, counts = None, torch.zeros(classes, dtype=torch.int64)
    with torch.inference_mode():
        for inputs, labels in batches(loader, indices, batch_size, "train_eval"):
            z = F.normalize(encoder(torch.as_tensor(inputs, device=device)), dim=1).cpu().double()
            labels = torch.as_tensor(labels, dtype=torch.int64)
            if z.ndim != 2 or len(z) != len(labels) or not torch.isfinite(z).all():
                raise ValueError("Invalid frozen embeddings")
            if labels.ndim != 1 or (labels < 0).any() or (labels >= classes).any():
                raise ValueError("Training labels outside class range")
            if sums is None:
                sums = torch.zeros(classes, z.shape[1], dtype=torch.float64)
            sums.index_add_(0, labels, z)
            counts += torch.bincount(labels, minlength=classes)
    # Empty shards carry no vector; merge_statistics obtains the feature dimension
    # from another shard. A completely empty retained pool is rejected explicitly.
    return {"sums": sums, "counts": counts}


def merge_statistics(statistics):
    nonempty = [item["sums"] for item in statistics if item["sums"] is not None]
    if not nonempty:
        raise ValueError("ProtoSISA needs at least one retained training example")
    sums = sum(nonempty, torch.zeros_like(nonempty[0]))
    counts = sum((item["counts"] for item in statistics), torch.zeros_like(statistics[0]["counts"]))
    means = sums / counts.clamp_min(1)[:, None]
    return {"vectors": F.normalize(means, dim=1).float(), "counts": counts}


def prototype_logits(embeddings, prototypes):
    vectors = prototypes["vectors"].to(embeddings.device, embeddings.dtype)
    logits = F.normalize(embeddings, dim=1) @ vectors.T
    # An absent class has no prototype and receives minimum cosine similarity.
    return logits.masked_fill(prototypes["counts"].to(embeddings.device)[None, :] == 0, -1.0)


def mix_logits(ensemble, prototype, weight=0.5):
    if not 0 <= weight <= 1 or ensemble.shape != prototype.shape:
        raise ValueError("Expected matching logits and a prototype weight in [0, 1]")
    return (1 - weight) * ensemble + weight * prototype


def mixture_parameters(config, training_examples, shards):
    """Resolve SAFE section 7 for independent shards (d=1), or an explicit variant.

    SAFE: lambda = exp(-d * |S| / 100). We use the current retained mean
    shard size, including empty shards, to define the retained-data procedure.
    Older saved configurations without a mode retain their fixed-weight rule.
    """
    if type(training_examples) is not int or type(shards) is not int or training_examples < 1 or shards < 1:
        raise ValueError("Mixture requires positive integer example and shard counts")
    mode = config.get("prototype_weighting", "fixed")
    if config.get("prototype_temperature", 1.0) != 1.0:
        raise ValueError("Temperature-scaled configurations are historical; use their saved results or archived launch source")
    mean_size = training_examples / shards
    if mode == "safe":
        if config.get("prototype_weight") is not None:
            raise ValueError("SAFE weighting computes its own weight")
        weight = math.exp(-mean_size / 100.0)
    elif mode == "fixed":
        weight = config.get("prototype_weight", 0.5)
        if not 0 <= weight <= 1:
            raise ValueError("Invalid fixed mixture weight")
    else:
        raise ValueError("prototype weighting must be safe or fixed")
    return {"prototype_weighting": mode, "prototype_weight": weight,
            "retained_training_examples": training_examples,
            "mean_shard_size": mean_size, "shard_graph_degree": 1}


def slice_plan(shard, deleted, slices, epochs):
    """SISA cumulative prefixes with whole epochs at each stage, including the tail."""
    passes = epochs_per_slice(epochs, slices)
    size = len(shard)
    if not 1 <= slices <= size:
        raise ValueError("slices must be between 1 and the initial shard size")
    return [(retained_indices(shard[:size if i == slices - 1 else (i + 1) * (size // slices)], deleted),
             passes) for i in range(slices)]


def checkpoint_paths(directory, config, shard_id, shard, deleted):
    paths = []
    for index, (indices, epochs) in enumerate(slice_plan(shard, deleted, config["slices"], config["epochs"])):
        key = digest([config["training_sha256"], shard_id, index, indices.tolist()])
        paths.append((directory / "cache" / f"model-{key}.pt", indices, epochs))
    return paths


def train_member(directory, config, shard_id, shard, deleted, step, loader, device,
                 model_factory=build_model):
    """Reuse unaffected prefixes; preserve Adam/momentum state at each slice boundary.

    Each slice has an independent deterministic RNG stream. Retrying an interrupted
    slice repeats that slice's work; the timing journal retains the failed effort.
    """
    paths = checkpoint_paths(directory, config, shard_id, shard, deleted)
    binding = {"training_sha256": config["training_sha256"],
               "memberships": [path.stem for path, _, _ in paths]}
    timer = UpdateTimer(directory, shard_id, str(step), binding, lambda: synchronized_time(device))
    if timer.record["status"] == "complete":
        if not all(path.exists() for path, _, _ in paths):
            raise ValueError("Recorded models were removed; use a new container to retrain")
        return paths[-1][0]
    with timer:
        model = optimizer = None
        for index, (path, indices, epochs) in enumerate(paths):
            item = timer.slice(index, path.stem, reused=path.exists())
            if not path.exists():
                if model is None:
                    seed_all((config["seed"] + shard_id) % 2**32)
                    model = model_factory(config, device)
                    optimizer_cls = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam,
                                     "adamw": torch.optim.AdamW}[config["optimizer"]]
                    optimizer = optimizer_cls(model.parameters(), lr=config["learning_rate"])
                    if index:
                        previous = torch.load(paths[index - 1][0], map_location=device, weights_only=True)
                        model.load_state_dict(previous["model"])
                        optimizer.load_state_dict(previous["optimizer"])
                        del previous
                model.train()
                seed_all((config["seed"] + shard_id + 1000003 * (index + 1)) % 2**32)
                for epoch in range(epochs):
                    started = synchronized_time(device)
                    loss_total, correct, examples = 0.0, 0, 0
                    try:
                        for inputs, labels in batches(loader, indices, config["batch_size"]):
                            optimizer.zero_grad(set_to_none=True)
                            logits = model(torch.as_tensor(inputs, device=device))
                            loss = F.cross_entropy(logits, torch.as_tensor(labels, dtype=torch.long, device=device))
                            loss.backward()
                            optimizer.step()
                            loss_total += loss.detach().item() * len(labels)
                            correct += (logits.detach().argmax(1) == torch.as_tensor(labels, device=device)).sum().item()
                            examples += len(labels)
                    except BaseException:
                        timer.add_training(item, synchronized_time(device) - started, complete=False)
                        raise
                    elapsed = synchronized_time(device) - started
                    timer.add_training(item, elapsed)
                    print(f"shard={shard_id} slice={index} epoch={epoch + 1}/{epochs} "
                          f"examples={examples} loss={loss_total / max(examples, 1):.4f} "
                          f"train_accuracy={correct / max(examples, 1):.4f} seconds={elapsed:.2f}", flush=True)
                atomic_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            else:
                # The next missing slice must restore this exact prefix, even if
                # an earlier missing slice was trained in the current invocation.
                model = optimizer = None
            timer.checkpoint()
    return paths[-1][0]


def update_prototypes(directory, config, shards, deleted, step, loader, encoder, device):
    path = directory / "cache" / f"prototypes-step-{step}.pt"
    journal = directory / "times" / f"prototypes:{step}.json"
    if journal.exists() and json.loads(journal.read_text())["status"] == "complete":
        if not path.exists():
            raise ValueError("Recorded prototypes were removed; use a new container to retrain")
        return torch.load(path, map_location="cpu", weights_only=True)
    with OperationTimer(journal, lambda: synchronized_time(device)):
        statistics = []
        for shard_id, (shard, requests) in enumerate(zip(shards, deleted)):
            indices = retained_indices(shard, requests)
            key = digest([config["training_sha256"], shard_id, indices.tolist()])
            cache = directory / "cache" / f"prototype-{key}.pt"
            if not cache.exists():
                value = prototype_statistics(encoder, loader, indices,
                    config["dataset_metadata"]["nb_classes"], config["batch_size"], device)
                atomic_checkpoint(cache, value)
            statistics.append(torch.load(cache, map_location="cpu", weights_only=True))
        prototypes = merge_statistics(statistics)
        atomic_checkpoint(path, prototypes)
    return prototypes


def evaluate(directory, config, paths, prototypes, step, loader, encoder, device,
             model_factory=build_model):
    split = config["eval_split"]
    classes = config["dataset_metadata"]["nb_classes"]
    size = config["dataset_metadata"][f"nb_{split}"]
    indices = np.arange(size)
    started = synchronized_time(device)
    mixture = mixture_parameters(config, int(prototypes["counts"].sum()), len(paths))
    ensemble = torch.zeros(size, classes)
    # Keep only one shard model in accelerator memory at a time.
    with torch.inference_mode():
        for path in paths:
            model = model_factory(config, device)
            # Lazy CPU mapping reads model tensors without loading the saved
            # optimizer tensors into accelerator memory during inference.
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True, mmap=True)["model"])
            model.eval()
            offset = 0
            for inputs, _ in batches(loader, indices, config["batch_size"], split):
                output = model(torch.as_tensor(inputs, device=device)).cpu()
                ensemble[offset:offset + len(output)] += output / len(paths)
                offset += len(output)
            del model
        prototype, labels = [], []
        encoder.eval()
        for inputs, target in batches(loader, indices, config["batch_size"], split):
            z = encoder(torch.as_tensor(inputs, device=device))
            prototype.append(prototype_logits(z, prototypes).cpu())
            labels.extend(np.asarray(target, dtype=np.int64).tolist())
        prototype = torch.cat(prototype)
        scores = mix_logits(ensemble, prototype, mixture["prototype_weight"]).numpy()
    elapsed = synchronized_time(device) - started
    components = directory / "outputs" / f"components-step-{step}-{split}.npz"
    components.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(components, ensemble=ensemble.numpy(), prototype=prototype.numpy(), labels=labels)
    saved_split = configured_manifest(config["dataset"])
    dataset_id = saved_split["dataset_id"] if saved_split else Path(config["dataset"]).parent.name
    source_hash = None
    source = getattr(loader, "test_set" if split == "test" else "val_set", None)
    if source is not None and hasattr(source, "data") and hasattr(source, "targets"):
        from dataset_splits import fingerprint_cifar
        source_hash = fingerprint_cifar(source.data, source.targets)
    evaluation = classification_evaluation(directory, run_id=config["comparison_run_id"],
        state_id=f"step-{step}", step=step, split=split, dataset=dataset_id,
        labels=labels, predictions=scores.argmax(1), scores=scores, num_classes=classes,
        split_sha256=saved_split["manifest_sha256"] if saved_split else None,
        source_sha256=source_hash, evaluation_time_sec=elapsed,
        method_details={**mixture,
                        "prototype_counts": prototypes["counts"].tolist(),
                        "ensemble_accuracy": float(np.mean(ensemble.numpy().argmax(1) == labels)),
                        "prototype_accuracy": float(np.mean(prototype.numpy().argmax(1) == labels)),
                        "component_scores": str(components.relative_to(directory)),
                        "aggregation": "mean_logits_plus_cosine"})
    return evaluation


def validate_options(args, info):
    if args.model not in ("vit", "resnet50", "roberta"):
        raise ValueError("ProtoSISA supports vit, resnet50 and roberta")
    if args.shards is None:
        args.shards = info["nb_classes"]
    if args.slices is None:
        args.slices = 2 if args.model == "roberta" else 1
    epochs_per_slice(args.epochs, args.slices)
    if (args.shards < 1 or args.slices < 1 or args.shards * args.slices > info["nb_train"]
            or args.epochs < 1 or args.batch_size < 1 or info[f"nb_{args.eval_split}"] < 1):
        raise ValueError("Invalid shard, slice, epoch, batch or evaluation size")
    if not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    mode = getattr(args, "prototype_weighting", "fixed")
    if mode == "safe":
        if args.prototype_weight is not None:
            raise ValueError("SAFE weighting computes its own weight; use --prototype-weighting fixed for variants")
    elif mode == "fixed":
        if args.prototype_weight is None:
            args.prototype_weight = 0.5
        if not 0 <= args.prototype_weight <= 1:
            raise ValueError("prototype weight must be in [0, 1]")
    else:
        raise ValueError("prototype weighting must be safe or fixed")
    if not 0 < args.learning_rate < float("inf") or not 0 <= args.dropout_rate < 1:
        raise ValueError("Invalid learning rate or dropout rate")


def prepare(args, device):
    info = json.loads(Path(args.dataset).read_text())
    validate_options(args, info)
    directory = record_directory(args.container)
    verify_run_source(directory)
    split = configured_manifest(args.dataset)
    workload = None
    if args.requests:
        from deletion_requests import load_workload
        if split is None:
            raise ValueError("Shared deletion requests require a split manifest")
        workload = load_workload(args.requests, split)
    config = {key: getattr(args, key) for key in ("model", "shards", "slices", "epochs", "batch_size",
        "learning_rate", "optimizer", "dropout_rate", "seed", "prototype_weight", "eval_split")}
    config.update(method="protosisa", dataset=str(Path(args.dataset).resolve()), dataset_metadata=info,
                  prototype_weighting=getattr(args, "prototype_weighting", "fixed"),
                  loader_sha256=digest({path.name: sha256(path.read_bytes()).hexdigest()
                      for path in sorted(Path(args.dataset).resolve().parent.glob("*.py"))}),
                  split_sha256=split["manifest_sha256"] if split else None,
                  workload_sha256=workload["workload_sha256"] if workload else None,
                  deletion_steps=len(workload["batches"]) if workload else 0,
                  prototype_features="pretrained_frozen_unit_embeddings",
                  prototype_preprocessing="train_eval", aggregation="mean_logits_plus_cosine")
    config["training_sha256"] = digest(config)
    config_path = directory / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if any(previous.get(key) != value for key, value in config.items()):
            raise ValueError("Saved ProtoSISA settings differ; use a new container")
        config = previous
        if config["provenance"]["source_sha256"] != runtime_provenance(ROOT)["source_sha256"]:
            raise ValueError("ProtoSISA source changed since launch; use a new container")
        validate_container_split(args.container, args.dataset)
        if config["input_sha256"] != input_digest(directory, config["deletion_steps"]):
            raise ValueError("Saved shard membership or deletion requests changed")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Existing container has no ProtoSISA config; use a new container")
        directory = initialize_container(args.container)
        bind_container_split(args.container, args.dataset)
        shards = np.empty(args.shards, dtype=object)
        shards[:] = np.array_split(np.random.default_rng(args.seed).permutation(info["nb_train"]), args.shards)
        np.save(directory / "splitfile.npy", shards, allow_pickle=True)
        if workload:
            export_deletion_requests(args.container, args.dataset, args.requests)
        else:
            requests = np.empty(args.shards, dtype=object)
            requests[:] = [np.array([], dtype=np.int64) for _ in shards]
            np.save(directory / "requestfile:0.npy", requests, allow_pickle=True)
        config.update(comparison_run_id="protosisa-" + uuid.uuid4().hex,
                      provenance=runtime_provenance(ROOT), hardware=hardware(device),
                      input_sha256=input_digest(directory, config["deletion_steps"]))
        write_json(config_path, config)
    return directory, config, split, workload


def input_digest(directory, steps):
    names = ["splitfile.npy", *[f"requestfile:{step}.npy" for step in range(steps + 1)]]
    names += [name for name in ("dataset_split.json", "deletion_requests.json") if (directory / name).exists()]
    return digest({name: sha256((directory / name).read_bytes()).hexdigest() for name in names})


def run(args, *, model_factory=build_model, encoder_factory=build_encoder):
    device = torch.device(args.device if args.device != "auto" else "cuda:0" if torch.cuda.is_available() else "cpu")
    directory, config, split, workload = prepare(args, device)
    loader = load_dataset_module(config["dataset"], config["dataset_metadata"])
    seed_all(config["seed"])
    encoder = encoder_factory(config["model"], device).requires_grad_(False).eval()
    shards = np.load(directory / "splitfile.npy", allow_pickle=True)
    run_manifest = manifest(run_id=config["comparison_run_id"], method="protosisa",
        dataset=split["dataset_id"] if split else Path(args.dataset).parent.name,
        model={"vit": "vit_b_16", "resnet50": "resnet50", "roberta": "roberta-base"}[config["model"]],
        config={key: value for key, value in config.items() if key not in ("comparison_run_id", "provenance", "hardware")},
        split_sha256=config["split_sha256"], workload_sha256=config["workload_sha256"],
        requested_steps=config["deletion_steps"], seed=config["seed"],
        provenance=config["provenance"], hardware_info=config["hardware"], precision="fp32")
    steps, evaluations, rows = [], [], []
    previous_deleted = 0
    for step in range(config["deletion_steps"] + 1):
        deleted = np.load(directory / f"requestfile:{step}.npy", allow_pickle=True)
        print(f"ProtoSISA step {step}/{config['deletion_steps']}", flush=True)
        paths = [train_member(directory, config, shard_id, shard, requests, step, loader, device, model_factory)
                 for shard_id, (shard, requests) in enumerate(zip(shards, deleted))]
        prototypes = update_prototypes(directory, config, shards, deleted, step, loader, encoder, device)
        mixture = mixture_parameters(config, int(prototypes["counts"].sum()), len(shards))
        timing_paths = [directory / "times" / f"shard-{shard_id}:{step}.json" for shard_id in range(len(shards))]
        timing_paths.append(directory / "times" / f"prototypes:{step}.json")
        total = total_timing(timing_paths)
        deleted_total = sum(map(len, deleted))
        steps.append(step_record(config["comparison_run_id"], step,
            deleted_count=deleted_total - previous_deleted, deleted_total=deleted_total,
            source_ids=workload["batches"][step - 1] if workload and step else None,
            status="ready" if total["measurement_complete"] else "incomplete",
            **{key: total[key] for key in ("update_time_sec", "training_time_sec", "timing_scope")},
            method_details={"prototype_counts": prototypes["counts"].tolist(),
                            **mixture, "shards": len(shards),
                            "measurement_complete": total["measurement_complete"]}))
        previous_deleted = deleted_total
        evaluation = evaluate(directory, config, paths, prototypes, step, loader, encoder, device, model_factory)
        evaluations.append(evaluation)
        write_bundle(directory, run_manifest, steps, evaluations)
        rows.append({"nb_shards": len(shards), "nb_requests": step, "accuracy": evaluation["metrics"]["accuracy"],
                     "prototype_weighting": mixture["prototype_weighting"],
                     "prototype_weight": mixture["prototype_weight"],
                     **{key: total[key] for key in ("update_time_sec", "training_time_sec", "timing_scope")}})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        atomic_text(directory / "report.csv", buffer.getvalue())
    checkpoint_summary(args.container)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("run", help="Train, replay shared deletions, and evaluate every state")
    command.add_argument("--model", choices=("vit", "resnet50", "roberta"), required=True)
    command.add_argument("--dataset", required=True)
    command.add_argument("--container", required=True, help="Container name or local study run directory")
    command.add_argument("--requests", help="Portable ordered deletion workload; omit for initial training only")
    command.add_argument("--shards", type=int, help="Default: number of classes")
    command.add_argument("--slices", type=int, help="Default: 1 for ViT, 2 for RoBERTa")
    command.add_argument("--epochs", type=int, default=1)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--learning-rate", type=float, default=1e-4)
    command.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    command.add_argument("--dropout-rate", type=float, default=0.4)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument("--prototype-weighting", choices=("safe", "fixed"), default="safe",
                         help="SAFE's published size-dependent weighting (default), or an explicit fixed variant")
    command.add_argument("--prototype-weight", type=float,
                         help="Fixed mode only: prototype coefficient; default 0.5")
    command.add_argument("--device", default="auto")
    command.add_argument("--eval-split", choices=("test", "val"), default="test")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
