from split_support import validate_container_split
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD, AdamW
from torch.nn.functional import one_hot
from sharded import fetchTestBatch
from container_storage import record_directory
import json
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model", default="purchase", help="Architecture to use, default purchase"
)

parser.add_argument(
    "--train", action="store_true", help="Perform SISA training on the shard"
)
parser.add_argument("--test", action="store_true", help="Compute shard predictions")

parser.add_argument(
    "--epochs",
    default=20,
    type=int,
    help="Train for the specified number of epochs, default 20",
)
parser.add_argument(
    "--batch_size",
    default=16,
    type=int,
    help="Size of the batches, relevant for both train and test, default 16",
)
parser.add_argument(
    "--dropout_rate",
    default=0.4,
    type=float,
    help="Dropout rate, if relevant, default 0.4",
)
parser.add_argument(
    "--learning_rate", default=0.001, type=float, help="Learning rate, default 0.001"
)

parser.add_argument("--optimizer", default="sgd", help="Optimizer, default sgd")

parser.add_argument(
    "--output_type",
    default="argmax",
    help="Type of outputs to be used in aggregation, can be either argmax or softmax, default argmax",
)

parser.add_argument("--container", help="Study record directory or standalone container name")
parser.add_argument("--shard", type=int, help="Index of the shard to train/test")
parser.add_argument(
    "--slices", default=1, type=int, help="Number of slices to use, default 1"
)
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)

parser.add_argument(
    "--chkpt_interval",
    default=1,
    type=int,
    help="Interval (in epochs) between two chkpts, -1 to disable chackpointing, default 1",
)
parser.add_argument(
    "--label",
    default="latest",
    help="Label to be used on simlinks and outputs, default latest",
)
parser.add_argument("--eval_split", choices=("test", "val"), default="test")
parser.add_argument("--seed", type=int, default=os.environ.get("SISA_SEED"), help="Base RNG seed; each shard uses seed + shard")
args = parser.parse_args()
if args.seed is not None:
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an integer in [0, 2**32)")
    shard_seed = (args.seed + args.shard) % 2**32
    np.random.seed(shard_seed)
    torch.manual_seed(shard_seed)
validate_container_split(args.container, args.dataset)
directory = record_directory(args.container)
if args.train:
    from studies.artifacts import verify_run_source
    verify_run_source(directory)

# Import the architecture.
from importlib import import_module

model_lib = import_module("architectures.{}".format(args.model))

# Retrive dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
input_shape = tuple(datasetfile["input_shape"])
nb_classes = datasetfile["nb_classes"]

# Use GPU if available.
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # pylint: disable=no-member

# Instantiate model and send to selected device.
model = model_lib.Model(input_shape, nb_classes, dropout_rate=args.dropout_rate)
model.to(device)

# Instantiate loss and optimizer.
loss_fn = CrossEntropyLoss()
if args.optimizer == "adam":
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "sgd":
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
elif args.optimizer == "adamw":
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
else:
    raise "Unsupported optimizer"

if args.train:
    from sisa_training import train_shard
    train_shard(args, model, optimizer, loss_fn, device)


if args.test:
    # Load model weights from shard checkpoint (last slice).
    model.load_state_dict(
        torch.load(
            directory / "cache" / f"shard-{args.shard}:{args.label}.pt"
        )
    )

    # Compute predictions batch per batch.
    outputs = np.empty((0, nb_classes))
    for images, _ in fetchTestBatch(args.dataset, args.batch_size, category=args.eval_split):
        # Convert data to torch format and send to selected device.
        gpu_images = torch.from_numpy(images).to(device)  # pylint: disable=no-member

        if args.output_type == "softmax":
            # Actual batch prediction.
            logits = model(gpu_images)
            predictions = softmax(logits, dim=1).to("cpu")  # Send back to cpu.

            # Convert back to numpy and concatenate with previous batches.
            outputs = np.concatenate((outputs, predictions.numpy()))

        else:
            # Actual batch prediction.
            logits = model(gpu_images)
            predictions = torch.argmax(logits, dim=1)  # pylint: disable=no-member

            # Convert to one hot, send back to cpu, convert back to numpy and concatenate with previous batches.
            out = one_hot(predictions, nb_classes).to("cpu")
            outputs = np.concatenate((outputs, out.numpy()))

    # Save outputs in numpy format.
    outputs = np.array(outputs)
    output_dir = directory / "outputs"
    if args.eval_split == "val":
        output_dir /= "val"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / f"shard-{args.shard}:{args.label}.npy",
        outputs,
    )
