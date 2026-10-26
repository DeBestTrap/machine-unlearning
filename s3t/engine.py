"""Resumable S3T experiments with one compact checkpoint per trained sequence."""
from contextlib import nullcontext
import csv
import io
import math
from pathlib import Path
import shutil
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from deletion_requests import load_workload, local_batches
from dataset_splits import load_manifest
from .adapters import S3TModel, cpu_state
from .artifacts import (atomic_json, atomic_torch, choose_device, cpu_tree, file_hash, lock,
                        provenance, read_json, seed_everything, seed_for, seed_worker, synchronize)
from .config import Config
from .data import load_data
from .models import build_backbone
from .partition import Partition, digest
from .sequences import select_sequences


def _loader(dataset, config, device, *, train=False, seed=0):
    return DataLoader(dataset, batch_size=config.batch_size if train else config.eval_batch_size,
                      shuffle=train, num_workers=config.num_workers, pin_memory=device.type == "cuda",
                      worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))


def _autocast(config, device):
    return torch.autocast(device.type, dtype=torch.bfloat16) if config.precision == "bf16" else nullcontext()


def _optimizer(config, parameters):
    cls = {"adamw": torch.optim.AdamW, "adam": torch.optim.Adam, "sgd": torch.optim.SGD}[config.optimizer]
    return cls(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def _scheduler(config, optimizer, steps):
    warmup = int(config.warmup_ratio * steps)

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
        if config.scheduler == "linear":
            return 1.0 - progress
        if config.scheduler == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.inference_mode()
def predict(model, dataset, config, device):
    model.eval()
    logits, targets = [], []
    for inputs, labels in _loader(dataset, config, device):
        with _autocast(config, device):
            output = model(inputs.to(device, non_blocking=True))
        if not torch.isfinite(output).all():
            raise FloatingPointError("Nonfinite evaluation logits")
        logits.append(output.float().cpu())
        targets.append(labels.long().cpu())
    return torch.cat(logits), torch.cat(targets)


class Run:
    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        self.config = Config.from_dict(read_json(self.directory / "config.json"))
        self.metadata = read_json(self.directory / "metadata.json")
        if self.metadata.get("run_sha256") != digest({key: value for key, value in self.metadata.items()
                                                      if key != "run_sha256"}):
            raise ValueError("Run metadata checksum mismatch")
        if self.metadata["config_sha256"] != digest(self.config.to_dict()):
            raise ValueError("Run configuration changed; create a new run")
        self.spec = read_json(self.directory / "backbone.json")
        self.split = load_manifest(self.directory / "dataset_split.json")
        if self.metadata["split_sha256"] != self.split["manifest_sha256"]:
            raise ValueError("Saved dataset split differs from run metadata")
        if self.metadata["backbone_sha256"] != file_hash(self.directory / "backbone.pt"):
            raise ValueError("Frozen backbone checksum mismatch")
        if self.metadata.get("backbone_spec_sha256", digest(self.spec)) != digest(self.spec):
            raise ValueError("Frozen backbone architecture changed")
        self._data = None

    @classmethod
    def create(cls, directory, config):
        config.validate()
        directory = Path(directory).expanduser().resolve()
        with lock(directory / ".lifecycle.lock"):
            if (directory / "metadata.json").exists():
                run = cls(directory)
                if run.config != config:
                    raise ValueError("Existing run has a different configuration; choose a new output directory")
                return run
            if (directory / "config.json").exists():
                if read_json(directory / "config.json") != config.to_dict():
                    raise ValueError("Incomplete run has a different configuration")
            elif any(path.name != ".lifecycle.lock" for path in directory.iterdir()):
                raise ValueError("Output directory is nonempty and is not an S3T run")
            atomic_json(directory / "config.json", config.to_dict())
            data = load_data(config)
            partition = Partition.create(data.source_ids, config.shards, config.slices,
                                         select_sequences(config), config.partition_seed)
            seed_everything(config.seed, config.deterministic)
            backbone, spec = build_backbone(config, data.num_classes)
            # Validate adapter geometry before writing a large base checkpoint.
            base_state = cpu_state(backbone)
            probe = S3TModel(backbone, spec["kind"], config)
            architecture = {"layer_groups": probe.layer_groups,
                            "adapter_parameters_per_sequence": probe.adapter_parameter_count(),
                            "head_parameters": sum(parameter.numel() for parameter in probe.head.parameters())}
            atomic_torch(directory / "backbone.pt", base_state)
            atomic_json(directory / "backbone.json", spec)
            atomic_json(directory / "dataset_split.json", data.split)
            atomic_json(directory / "generations/0000/partition.json", partition.to_dict())
            atomic_json(directory / "generations/0000/generation.json", {"generation": 0, "parent_step": 0,
                        "excluded_source_ids": [], "partition_sha256": partition.to_dict()["partition_sha256"]})
            state = {"schema_version": 1, "step": 0, "generation": 0, "deleted_source_ids": [],
                     "history": [], "workload_sha256": None}
            atomic_json(directory / "state.json", state)
            metadata = {"schema_version": 1, "config_sha256": digest(config.to_dict()),
                        "split_sha256": data.split["manifest_sha256"], "preprocessing": data.preprocessing,
                        "source_ids": data.source_ids, "backbone_sha256": file_hash(directory / "backbone.pt"),
                        "backbone_spec_sha256": digest(spec),
                        "architecture": architecture, "provenance": provenance()}
            # Preserve the implementation behind this run, including uncommitted code.
            from .config import ROOT
            for relative in metadata["provenance"]["source_files"]:
                destination = directory / "source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            metadata["run_sha256"] = digest(metadata)
            atomic_json(directory / "metadata.json", metadata)
        run = cls(directory)
        run._data = data
        return run

    @property
    def data(self):
        if self._data is None:
            self._data = load_data(self.config, self.metadata["preprocessing"])
            if (self._data.split != self.split or self._data.source_ids != self.metadata["source_ids"]
                    or self._data.preprocessing != self.metadata["preprocessing"]):
                raise ValueError("Dataset or source mapping changed since this run was initialized")
        return self._data

    def state(self):
        state = read_json(self.directory / "state.json")
        if state["schema_version"] != 1 or state["step"] != len(state["history"]):
            raise ValueError("Invalid deletion state")
        batches = [source_id for event in state["history"] for source_id in event["source_ids"]]
        if sorted(batches) != state["deleted_source_ids"] or len(set(batches)) != len(batches):
            raise ValueError("Deletion history/cumulative IDs disagree")
        if not set(batches).issubset(self.metadata["source_ids"]):
            raise ValueError("Deletion history contains IDs outside training")
        return state

    def generation_dir(self, generation):
        return self.directory / "generations" / f"{generation:04d}"

    def partition(self, generation):
        directory = self.generation_dir(generation)
        partition = Partition.from_dict(read_json(directory / "partition.json"))
        generation_meta = read_json(directory / "generation.json")
        if generation_meta["partition_sha256"] != partition.to_dict()["partition_sha256"]:
            raise ValueError("Generation partition changed")
        if set(partition.locations) != set(self.metadata["source_ids"]) - set(generation_meta["excluded_source_ids"]):
            raise ValueError("Partition does not cover exactly the retained training pool")
        return partition

    def member_path(self, generation, shard, sequence):
        return self.generation_dir(generation) / "members" / f"shard-{shard:04d}-sequence-{sequence:04d}.pt"

    def _binding(self, generation, shard, sequence):
        return {"run_sha256": self.metadata["run_sha256"], "generation": generation,
                "partition_sha256": self.partition(generation).to_dict()["partition_sha256"],
                "shard": shard, "sequence": sequence}

    def checkpoint(self, generation, shard, sequence):
        value = torch.load(self.member_path(generation, shard, sequence), map_location="cpu", weights_only=True)
        if value.get("schema_version") != 1 or value["binding"] != self._binding(generation, shard, sequence):
            raise ValueError("Checkpoint belongs to a different run/partition/member")
        return value

    def _model(self, generation, shard, sequence, device):
        seed_everything(seed_for(self.config.seed, generation, shard, sequence), self.config.deterministic)
        backbone, _ = build_backbone(self.config, self.spec["num_classes"], self.spec)
        backbone.load_state_dict(torch.load(self.directory / "backbone.pt", map_location="cpu", weights_only=True))
        return S3TModel(backbone, self.spec["kind"], self.config).to(device)

    def train(self, device="auto", shard=None, sequence=None, *, max_epochs=None):
        """Resume at an epoch boundary. max_epochs is a bounded-run/testing hook."""
        if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 1):
            raise ValueError("max_epochs must be a positive integer or None")
        device = choose_device(device)
        if self.config.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bfloat16")
        with lock(self.directory / ".lifecycle.lock", shared=True):
            state = self.state()
            generation = state["generation"]
            if state["deleted_source_ids"]:
                self.require_complete(generation)
                raise ValueError("Deletion has started. Use retrain to build a new retained-data generation")
            return self._train_generation(generation, device, shard, sequence, max_epochs=max_epochs)

    def _train_generation(self, generation, device, shard=None, sequence=None, *, max_epochs=None):
        if shard is not None and not 0 <= shard < self.config.shards:
            raise ValueError("Shard index out of range")
        if sequence is not None and not 0 <= sequence < self.config.budget:
            raise ValueError("Sequence index out of range")
        remaining = max_epochs
        for shard_id in range(self.config.shards) if shard is None else [shard]:
            for sequence_id in range(self.config.budget) if sequence is None else [sequence]:
                path = self.member_path(generation, shard_id, sequence_id)
                with lock(path.with_suffix(".lock")):
                    spent = self._train_member(generation, shard_id, sequence_id, device, remaining)
                    if remaining is not None:
                        remaining -= spent
                        if remaining <= 0:
                            return self.status()
        return self.status()

    def _train_member(self, generation, shard, sequence, device, max_epochs):
        config, path = self.config, self.member_path(generation, shard, sequence)
        previous = self.checkpoint(generation, shard, sequence) if path.exists() else None
        if previous and previous["complete"]:
            return 0
        partition = self.partition(generation)
        model = self._model(generation, shard, sequence, device)
        start_stage = previous["stage"] if previous else 0
        history = previous["history"] if previous else []
        scores = previous["validation_scores"] if previous else []
        train_seconds = previous["train_seconds"] if previous else 0.0
        evaluation_seconds = previous["validation_seconds"] if previous else 0.0
        peak_memory = previous.get("peak_gpu_allocated_bytes", 0) if previous else 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epochs_run = 0
        if previous:
            model.load_exported_state(previous["model"])
        binding = self._binding(generation, shard, sequence)
        for stage in range(start_stage, config.slices):
            params = model.begin_stage(stage)
            optimizer = _optimizer(config, params)
            training_data = self.data.subset(partition.training_ids(shard, sequence, stage + 1))
            batches = math.ceil(len(training_data) / config.batch_size)
            scheduler = _scheduler(config, optimizer, batches * config.epochs_per_slice)
            start_epoch = 0
            if previous and stage == start_stage and previous["next_epoch"]:
                model.head.load_state_dict(previous["model"]["head_current"])
                optimizer.load_state_dict(previous["optimizer"])
                scheduler.load_state_dict(previous["scheduler"])
                start_epoch = previous["next_epoch"]
            for epoch in range(start_epoch, config.epochs_per_slice):
                epoch_seed = seed_for(config.seed, generation, shard, sequence, stage, epoch)
                seed_everything(epoch_seed, config.deterministic)
                model.train()
                synchronize(device)
                started = perf_counter()
                loss_sum, correct, examples = 0.0, 0, 0
                for inputs, labels in _loader(training_data, config, device, train=True, seed=epoch_seed):
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(config, device):
                        logits = model(inputs)
                        loss = F.cross_entropy(logits, labels)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at shard={shard}, sequence={sequence}, stage={stage}")
                    loss.backward()
                    if config.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm, error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    examples += len(labels)
                    loss_sum += float(loss.detach()) * len(labels)
                    correct += int((logits.detach().argmax(dim=-1) == labels).sum())
                synchronize(device)
                seconds = perf_counter() - started
                train_seconds += seconds
                record = {"stage": stage, "epoch": epoch, "examples": examples, "loss": loss_sum / examples,
                          "accuracy": correct / examples, "seconds": seconds, "lr": optimizer.param_groups[0]["lr"]}
                history.append(record)
                finished_stage = epoch + 1 == config.epochs_per_slice
                if finished_stage:
                    model.finish_stage()
                    synchronize(device)
                    eval_started = perf_counter()
                    val_logits, val_labels = predict(model, self.data.val, config, device)
                    synchronize(device)
                    evaluation_seconds += perf_counter() - eval_started
                    score = float((val_logits.argmax(-1) == val_labels).float().mean())
                    scores.append(score)
                    record["validation_accuracy"] = score
                if device.type == "cuda":
                    peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
                checkpoint = {"schema_version": 1, "binding": binding, "complete": finished_stage and stage + 1 == config.slices,
                              "stage": stage + int(finished_stage), "next_epoch": 0 if finished_stage else epoch + 1,
                              "model": model.export_state(), "optimizer": None if finished_stage else cpu_tree(optimizer.state_dict()),
                              "scheduler": None if finished_stage else scheduler.state_dict(), "history": history,
                              "validation_scores": scores, "train_seconds": train_seconds, "validation_seconds": evaluation_seconds,
                              "peak_gpu_allocated_bytes": peak_memory}
                atomic_torch(path, checkpoint)
                atomic_json(path.with_suffix(".json"), {key: value for key, value in checkpoint.items()
                                                       if key not in ("model", "optimizer", "scheduler")})
                print(f"generation={generation} shard={shard} sequence={sequence} stage={stage + 1}/{config.slices} "
                      f"epoch={epoch + 1}/{config.epochs_per_slice} loss={record['loss']:.4f} "
                      f"accuracy={record['accuracy']:.4f} seconds={seconds:.2f}", flush=True)
                epochs_run += 1
                if max_epochs is not None and epochs_run >= max_epochs:
                    return epochs_run
            previous = None
        return epochs_run

    def require_complete(self, generation):
        for shard in range(self.config.shards):
            for sequence in range(self.config.budget):
                path = self.member_path(generation, shard, sequence)
                if not path.exists() or not self.checkpoint(generation, shard, sequence)["complete"]:
                    raise ValueError(f"Training is incomplete: shard={shard}, sequence={sequence}")

    def _selection(self, state):
        generation = state["generation"]
        partition = self.partition(generation)
        deleted = set(state["deleted_source_ids"]) & set(partition.locations)
        scores = None
        if self.config.selection == "validation":
            scores = [[self.checkpoint(generation, shard, sequence)["validation_scores"]
                       for sequence in range(self.config.budget)] for shard in range(self.config.shards)]
        return partition.select(deleted, scores)

    def costs(self, generation):
        checkpoints = [self.checkpoint(generation, shard, sequence)
                       for shard in range(self.config.shards) for sequence in range(self.config.budget)]
        return {"generation_training_seconds": sum(value["train_seconds"] for value in checkpoints),
                "generation_validation_seconds": sum(value["validation_seconds"] for value in checkpoints),
                "generation_training_examples": sum(epoch["examples"] for value in checkpoints for epoch in value["history"]),
                "backbone_bytes": (self.directory / "backbone.pt").stat().st_size,
                "generation_checkpoint_bytes": sum(self.member_path(generation, shard, sequence).stat().st_size
                                                   for shard in range(self.config.shards) for sequence in range(self.config.budget)),
                "peak_gpu_allocated_bytes": max(value.get("peak_gpu_allocated_bytes", 0) for value in checkpoints)}

    def status(self):
        state = self.state()
        generation = state["generation"]
        complete, epochs = 0, 0
        for shard in range(self.config.shards):
            for sequence in range(self.config.budget):
                path = self.member_path(generation, shard, sequence)
                if path.exists():
                    checkpoint = self.checkpoint(generation, shard, sequence)
                    complete += int(checkpoint["complete"])
                    epochs += len(checkpoint["history"])
        selected = self._selection(state) if complete == self.config.shards * self.config.budget else None
        return {"run": str(self.directory), "generation": generation, "completed_members": complete,
                "total_members": self.config.shards * self.config.budget, "completed_epochs": epochs,
                "deletion_step": state["step"], "deleted_examples": len(state["deleted_source_ids"]),
                "selected": selected,
                "status": "training" if selected is None else ("ready" if any(selected) else "requires_retraining")}

    def _prediction(self, generation, member, split, device, cache):
        shard, sequence, prefix = member["shard"], member["sequence"], member["prefix"]
        member_path = self.member_path(generation, shard, sequence)
        binding = dict(self._binding(generation, shard, sequence), prefix=prefix, split=split,
                       checkpoint_sha256=file_hash(member_path))
        cache_path = self.generation_dir(generation) / "predictions" / f"s{shard}-q{sequence}-p{prefix}-{split}.pt"
        if cache and cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu", weights_only=True)
            if saved["binding"] != binding:
                raise ValueError("Prediction cache does not match its model; remove this cache file to recompute")
            return saved["logits"], saved["labels"], True
        model = self._model(generation, shard, sequence, device)
        model.load_exported_state(self.checkpoint(generation, shard, sequence)["model"])
        model.activate_prefix(prefix)
        logits, labels = predict(model, getattr(self.data, split), self.config, device)
        if cache:
            atomic_torch(cache_path, {"binding": binding, "logits": logits, "labels": labels})
        return logits, labels, False

    def evaluate(self, device="auto", split="test", *, step=None, cache=True):
        if split not in ("val", "test"):
            raise ValueError("Evaluation split must be val or test")
        with lock(self.directory / ".lifecycle.lock", shared=True):
            state = self.state()
            if step is not None and step != state["step"]:
                if not 0 <= step <= state["step"]:
                    raise ValueError("Requested deletion step does not exist")
                if step == 0:
                    state = dict(state, step=0, generation=0, deleted_source_ids=[])
                else:
                    event = state["history"][step - 1]
                    state = dict(state, step=step, generation=event["generation"],
                                 deleted_source_ids=event["cumulative_source_ids"])
            return self._evaluate_state(state, choose_device(device), split, cache)

    def _evaluate_state(self, state, device, split, cache):
        self.require_complete(state["generation"])
        # Validate current raw data even if all member predictions are cached.
        data = self.data
        selected = self._selection(state)
        active = [member for member in selected if member is not None]
        synchronize(device)
        started = perf_counter()
        total, reference, hits = None, None, 0
        for member in active:
            logits, labels, hit = self._prediction(state["generation"], member, split, device, cache)
            hits += int(hit)
            if len(logits) != len(getattr(data, split)) or logits.shape[1] != data.num_classes:
                raise ValueError("Invalid prediction shape")
            if reference is None:
                reference = labels
            elif not torch.equal(reference, labels):
                raise ValueError("Ensemble members used different evaluation examples")
            if self.config.aggregation == "probabilities":
                logits = logits.softmax(-1)
            elif self.config.aggregation == "vote":
                logits = F.one_hot(logits.argmax(-1), data.num_classes).float()
            total = logits if total is None else total + logits
        synchronize(device)
        result = {"schema_version": 1, "run_sha256": self.metadata["run_sha256"], "method": "s3t",
                  "dataset": self.config.dataset, "model": self.config.model, "seed": self.config.seed,
                  "shards": self.config.shards, "slices": self.config.slices, "budget": self.config.budget,
                  "generation": state["generation"], "step": state["step"], "split": split,
                  "deleted_examples": len(state["deleted_source_ids"]), "active_shards": len(active),
                  "selected": selected, "status": "ready" if active else "requires_retraining",
                  "accuracy": None if total is None else float((total.argmax(-1) == reference).float().mean()),
                  "examples": len(getattr(data, split)), "evaluation_seconds": perf_counter() - started,
                  "prediction_cache_hits": hits, "aggregation": self.config.aggregation,
                  "selection": self.config.selection, "precision": self.config.precision,
                  "split_manifest_sha256": self.split["manifest_sha256"],
                  "workload_sha256": state["workload_sha256"], "capped_training": self.config.max_train_examples is not None,
                  "capped_evaluation": self.config.max_eval_examples is not None,
                  "deletion_planning_seconds": state["history"][state["step"] - 1]["planning_seconds"] if state["step"] else 0.0,
                  **self.costs(state["generation"])}
        path = self.directory / "evaluations" / f"g{state['generation']:04d}-step-{state['step']:06d}-{split}.json"
        atomic_json(path, result)
        return result

    def delete(self, source_ids, *, workload_sha256=None):
        with lock(self.directory / ".lifecycle.lock"):
            state = self.state()
            self.require_complete(state["generation"])
            if (self.generation_dir(state["generation"] + 1) / "generation.json").exists():
                raise ValueError("A retraining generation is pending; resume retrain before further deletions")
            if (not source_ids or any(type(index) is not int for index in source_ids)
                    or len(set(source_ids)) != len(source_ids)):
                raise ValueError("Deletion batch must contain distinct integer source IDs")
            if not set(source_ids).issubset(self.metadata["source_ids"]):
                raise ValueError("Deletion ID is outside this run's training pool")
            if set(source_ids) & set(state["deleted_source_ids"]):
                raise ValueError("Deletion ID was already deleted")
            if state["workload_sha256"] != workload_sha256 and (state["step"] or state["workload_sha256"] is not None):
                raise ValueError("Cannot mix different workloads or manual deletion batches in one run")
            started = perf_counter()
            state["deleted_source_ids"] = sorted(set(state["deleted_source_ids"]) | set(source_ids))
            state["step"] += 1
            state["workload_sha256"] = workload_sha256
            selected = self._selection(state)
            partition = self.partition(state["generation"])
            event = {"step": state["step"], "generation": state["generation"], "source_ids": list(source_ids),
                     "cumulative_source_ids": state["deleted_source_ids"], "selected": selected,
                     "prefixes": partition.prefixes(set(state["deleted_source_ids"]) & set(partition.locations)),
                     "status": "ready" if any(selected) else "requires_retraining",
                     "planning_seconds": perf_counter() - started, "workload_sha256": workload_sha256}
            state["history"].append(event)
            # One atomic state contains both the authoritative history and current masks.
            atomic_json(self.directory / "state.json", state)
            atomic_json(self.directory / "deletion_trace.json", state["history"])
            return event

    def replay(self, workload_path, device="auto", split="test", *, steps=None, cache=True,
               on_exhaustion="stop"):
        if on_exhaustion not in ("stop", "continue", "retrain"):
            raise ValueError("Invalid on_exhaustion policy")
        workload = load_workload(workload_path, self.split)
        local_batches(workload, self.split, self.metadata["source_ids"])
        total = len(workload["batches"]) if steps is None else steps
        if not 0 <= total <= len(workload["batches"]):
            raise ValueError("steps must be between zero and workload length")
        with lock(self.directory / ".replay.lock"):
            state = self.state()
            if state["step"] > total:
                raise ValueError("Run has already passed the requested replay step")
            for event, batch in zip(state["history"], workload["batches"]):
                if event["source_ids"] != batch or event["workload_sha256"] != workload["workload_sha256"]:
                    raise ValueError("Existing deletion history differs from this workload")
            marker = self.directory / "deletion_requests.json"
            if marker.exists() and read_json(marker) != workload:
                raise ValueError("Run is already bound to a different deletion workload")
            with lock(self.directory / ".lifecycle.lock"):
                latest = self.state()
                if latest != state:
                    raise ValueError("Deletion state changed while preparing replay; retry")
                self.require_complete(state["generation"])
                state["workload_sha256"] = workload["workload_sha256"]
                atomic_json(marker, workload)
                atomic_json(self.directory / "state.json", state)
            results = [self.evaluate(device, split, cache=cache)]
            if results[-1]["status"] == "requires_retraining":
                if on_exhaustion == "stop":
                    self.export_csv()
                    return results
                if on_exhaustion == "retrain":
                    self.retrain(device)
                    results.append(self.evaluate(device, split, cache=cache))
            for index in range(state["step"], total):
                event = self.delete(workload["batches"][index], workload_sha256=workload["workload_sha256"])
                result = self.evaluate(device, split, cache=cache)
                results.append(result)
                print(f"deletion_step={event['step']} deleted={result['deleted_examples']} "
                      f"active_shards={result['active_shards']} accuracy={result['accuracy']}", flush=True)
                if result["status"] == "requires_retraining":
                    if on_exhaustion == "stop":
                        break
                    if on_exhaustion == "retrain":
                        self.retrain(device)
                        results.append(self.evaluate(device, split, cache=cache))
            self.export_csv()
            return results

    def retrain(self, device="auto", *, max_epochs=None):
        """Start/resume a new generation on all retained examples, then publish it."""
        if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 1):
            raise ValueError("max_epochs must be a positive integer or None")
        with lock(self.directory / ".lifecycle.lock"):
            state = self.state()
            if not state["deleted_source_ids"]:
                raise ValueError("No deletions to retrain after")
            active_meta = read_json(self.generation_dir(state["generation"]) / "generation.json")
            if active_meta["excluded_source_ids"] == state["deleted_source_ids"]:
                return read_json(self.generation_dir(state["generation"]) / "retraining.json")
            generation = state["generation"] + 1
            partition = Partition.create(
                sorted(set(self.metadata["source_ids"]) - set(state["deleted_source_ids"])),
                self.config.shards, self.config.slices, select_sequences(self.config),
                seed_for(self.config.partition_seed, generation),
            )
            directory = self.generation_dir(generation)
            generation_meta = {"generation": generation, "parent_step": state["step"],
                               "excluded_source_ids": state["deleted_source_ids"],
                               "partition_sha256": partition.to_dict()["partition_sha256"]}
            if (directory / "generation.json").exists():
                if read_json(directory / "generation.json") != generation_meta:
                    raise ValueError("Pending retraining generation was started from different deletions")
            else:
                atomic_json(directory / "partition.json", partition.to_dict())
                atomic_json(directory / "generation.json", generation_meta)
            started = perf_counter()
            self._train_generation(generation, choose_device(device), max_epochs=max_epochs)
            if max_epochs is not None:
                try:
                    self.require_complete(generation)
                except ValueError:
                    return {"status": "retraining", "generation": generation}
            self.require_complete(generation)
            # Publication is atomic: incomplete generations never serve predictions.
            record = {"generation": generation, "parent_step": state["step"],
                      "excluded_source_ids": state["deleted_source_ids"],
                      "invocation_seconds": perf_counter() - started,
                      "training_seconds": sum(self.checkpoint(generation, shard, sequence)["train_seconds"]
                                              for shard in range(self.config.shards)
                                              for sequence in range(self.config.budget))}
            atomic_json(directory / "retraining.json", record)
            state["generation"] = generation
            atomic_json(self.directory / "state.json", state)
            return record

    def export_csv(self):
        records = [read_json(path) for path in sorted((self.directory / "evaluations").glob("*.json"))]
        fields = ["method", "dataset", "model", "seed", "shards", "slices", "budget", "generation", "step", "split",
                  "deleted_examples", "active_shards", "accuracy", "status", "evaluation_seconds", "prediction_cache_hits",
                  "deletion_planning_seconds", "generation_training_seconds", "generation_validation_seconds",
                  "generation_training_examples", "backbone_bytes", "generation_checkpoint_bytes", "peak_gpu_allocated_bytes",
                  "selection", "aggregation", "precision", "split_manifest_sha256", "workload_sha256"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        from .artifacts import _atomic
        path = self.directory / "results.csv"
        _atomic(path, lambda stream: stream.write(buffer.getvalue().encode()))
        return str(path)

    def audit(self):
        """Verify saved deletion dependencies and source identities without inference."""
        with lock(self.directory / ".lifecycle.lock", shared=True):
            state = self.state()
            self.require_complete(state["generation"])
            workload_path = self.directory / "deletion_requests.json"
            workload = load_workload(workload_path, self.split) if workload_path.exists() else None
            if workload is not None:
                local_batches(workload, self.split, self.metadata["source_ids"])
                if state["workload_sha256"] != workload["workload_sha256"]:
                    raise ValueError("State and workload identity disagree")
                if state["step"] > len(workload["batches"]):
                    raise ValueError("Trace is longer than the workload")
            elif state["workload_sha256"] is not None:
                raise ValueError("State refers to a missing workload")
            cumulative, checked_members = set(), 0
            for index, event in enumerate(state["history"], 1):
                if event["step"] != index:
                    raise ValueError("Trace step order is invalid")
                if event["workload_sha256"] != state["workload_sha256"]:
                    raise ValueError("Trace workload identity differs from run state")
                if workload is not None and event["source_ids"] != workload["batches"][index - 1]:
                    raise ValueError("Trace batch differs from saved workload")
                cumulative.update(event["source_ids"])
                if event["cumulative_source_ids"] != sorted(cumulative):
                    raise ValueError("Trace cumulative source IDs disagree")
                partition = self.partition(event["generation"])
                expected_prefixes = partition.prefixes(cumulative & set(partition.locations))
                if event["prefixes"] != expected_prefixes:
                    raise ValueError("Trace contains stale/incorrect prefix masks")
                historical = dict(state, generation=event["generation"], deleted_source_ids=sorted(cumulative))
                if self._selection(historical) != event["selected"]:
                    raise ValueError("Trace selection differs from the configured policy")
                for shard, prefixes in enumerate(expected_prefixes):
                    for sequence, prefix in enumerate(prefixes):
                        dependencies = partition.training_ids(shard, sequence, prefix)
                        if set(dependencies) & cumulative:
                            raise ValueError("A surviving prefix includes deleted training sources")
                        checkpoint = self.checkpoint(event["generation"], shard, sequence)
                        if len(checkpoint["model"]["head_history"]) < prefix:
                            raise ValueError("A surviving prefix has no matching head snapshot")
                        checked_members += 1
            result = {"status": "passed", "run_sha256": self.metadata["run_sha256"],
                      "steps_checked": state["step"], "deleted_examples": len(cumulative),
                      "prefixes_checked": checked_members,
                      "workload_sha256": state["workload_sha256"],
                      "scope": "saved source identities, prefix dependencies, selections, and head availability"}
            atomic_json(self.directory / "deletion_audit.json", result)
            return result
