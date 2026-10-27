"""SISA shard training with complete request timing and slice-level accounting."""
from hashlib import sha256
from time import perf_counter

import torch

from container_storage import record_directory
from sharded import fetchShardBatch, getShardHash, sizeOfShard
from sisa_timing import UpdateTimer, atomic_text


def synchronized_time(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def _link(path, target):
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        raise ValueError(f"Expected a checkpoint alias at {path}")
    path.symlink_to(target)


def train_shard(args, model, optimizer, loss_fn, device):
    directory = record_directory(args.container)
    cache, times = directory / "cache", directory / "times"
    model_link = cache / f"shard-{args.shard}:{args.label}.pt"
    binding = {key: getattr(args, key) for key in
               ("model", "dataset", "shard", "label", "slices", "epochs", "batch_size",
                "learning_rate", "optimizer", "dropout_rate", "chkpt_interval")}
    for name in ("splitfile.npy", f"requestfile:{args.label}.npy"):
        binding[name] = sha256((directory / name).read_bytes()).hexdigest()
    timer = UpdateTimer(directory, args.shard, args.label, binding, lambda: synchronized_time(device))
    if timer.record["status"] == "complete":
        if not model_link.exists():
            raise ValueError("This update was recorded but its models were removed; use a new container to retrain")
        return timer.record
    if model_link.exists() and timer.record["status"] == "new":
        raise ValueError("This label has legacy checkpoints without complete update timing; use a new container")
    with timer:
        shard_size = sizeOfShard(args.container, args.shard)
        if args.slices < 1 or shard_size < args.slices:
            raise ValueError("slices must be between 1 and the shard size")
        slice_size = shard_size // args.slices
        avg_epochs = 2 * args.slices / (args.slices + 1) * args.epochs / args.slices
        loaded = False
        for sl in range(args.slices):
            slice_hash = getShardHash(args.container, args.label, args.shard, until=(sl + 1) * slice_size)
            checkpoint = cache / f"{slice_hash}.pt"
            item = timer.slice(sl, slice_hash, reused=checkpoint.exists())
            if not checkpoint.exists():
                start_epoch, elapsed_training = 0, 0.0
                slice_epochs = int((sl + 1) * avg_epochs) - int(sl * avg_epochs)
                if not loaded:
                    recovery = sorted(cache.glob(f"{slice_hash}_*.pt"),
                                      key=lambda path: int(path.stem.rsplit("_", 1)[1]))
                    if recovery:
                        if not item["epochs_trained"]:
                            raise ValueError("Recovery checkpoint has no matching timing journal; use a new container")
                        latest = recovery[-1]
                        model.load_state_dict(torch.load(latest, map_location=device, weights_only=True))
                        # The suffix names the last completed zero-based epoch.
                        start_epoch = int(latest.stem.rsplit("_", 1)[1]) + 1
                        elapsed_training = float((times / f"{latest.stem}.time").read_text())
                    elif sl > 0:
                        previous = getShardHash(args.container, args.label, args.shard, until=sl * slice_size)
                        model.load_state_dict(torch.load(cache / f"{previous}.pt", map_location=device, weights_only=True))
                    loaded = True
                for epoch in range(start_epoch, slice_epochs):
                    started = synchronized_time(device)
                    try:
                        for images, labels in fetchShardBatch(
                            args.container, args.label, args.shard, args.batch_size, args.dataset,
                            until=(sl + 1) * slice_size if sl < args.slices - 1 else None,
                        ):
                            gpu_images = torch.from_numpy(images).to(device)
                            gpu_labels = torch.from_numpy(labels).to(device)
                            logits = model(gpu_images)
                            loss = loss_fn(logits, gpu_labels)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                    except BaseException:
                        timer.add_training(item, synchronized_time(device) - started, complete=False)
                        raise
                    seconds = synchronized_time(device) - started
                    elapsed_training += seconds
                    timer.add_training(item, seconds)
                    if args.chkpt_interval != -1 and epoch % args.chkpt_interval == args.chkpt_interval - 1:
                        recovery = cache / f"{slice_hash}_{epoch}.pt"
                        torch.save(model.state_dict(), recovery)
                        atomic_text(times / f"{recovery.stem}.time", f"{elapsed_training}\n")
                        previous = cache / f"{slice_hash}_{epoch - args.chkpt_interval}.pt"
                        previous.unlink(missing_ok=True)
                        (times / f"{previous.stem}.time").unlink(missing_ok=True)
                        timer.checkpoint()
                torch.save(model.state_dict(), checkpoint)
                atomic_text(times / f"{slice_hash}.time", f"{elapsed_training}\n")
                for recovery in cache.glob(f"{slice_hash}_*.pt"):
                    recovery.unlink()
                    (times / f"{recovery.stem}.time").unlink(missing_ok=True)
            timer.checkpoint()
        _link(model_link, f"{slice_hash}.pt")
    return timer.record
