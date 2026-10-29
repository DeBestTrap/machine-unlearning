from split_support import validate_container_split
import numpy as np
import json
from dataset_loading import load_dataset_module
from container_storage import record_directory

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--strategy", default="uniform", help="Voting strategy, default uniform"
)
parser.add_argument("--container", help="Study record directory or standalone container name")
parser.add_argument("--shards", type=int, default=1, help="Number of shards, default 1")
parser.add_argument(
    "--dataset",
    default="datasets/purchase/datasetfile",
    help="Location of the datasetfile, default datasets/purchase/datasetfile",
)
parser.add_argument(
    "--baseline", type=int, help="Use only the specified shard (lone shard baseline)"
)
parser.add_argument("--label", default="latest", help="Label, default latest")
parser.add_argument("--eval_split", choices=("test", "val"), default="test")
parser.add_argument("--save-evaluation", action="store_true", help="Retain comparison metrics, predictions and labels")
args = parser.parse_args()
validate_container_split(args.container, args.dataset)
directory = record_directory(args.container)

# Load dataset metadata.
with open(args.dataset) as f:
    datasetfile = json.loads(f.read())
dataloader = load_dataset_module(args.dataset, datasetfile)

# Output files used for the vote.
if args.baseline != None:
    filenames = ["shard-{}:{}.npy".format(args.baseline, args.label)]
else:
    filenames = ["shard-{}:{}.npy".format(i, args.label) for i in range(args.shards)]

# Concatenate output files.
outputs = []
for filename in filenames:
    outputs.append(
        np.load(
            directory / "outputs" / ("val" if args.eval_split == "val" else "") / filename,
            allow_pickle=True,
        )
    )
outputs = np.array(outputs)

# Compute weight vector based on given strategy.
if args.strategy == "uniform":
    weights = (
        1 / outputs.shape[0] * np.ones((outputs.shape[0],))
    )  # pylint: disable=unsubscriptable-object
elif args.strategy.startswith("models:"):
    models = np.array(args.strategy.split(":")[1].split(",")).astype(int)
    weights = np.zeros((outputs.shape[0],))  # pylint: disable=unsubscriptable-object
    weights[models] = 1 / models.shape[0]  # pylint: disable=unsubscriptable-object
elif args.strategy == "proportional":
    split = np.load(
        directory / "splitfile.npy", allow_pickle=True
    )
    weights = np.array([shard.shape[0] for shard in split])

# Tensor contraction of outputs and weights (on the shard dimension).
votes = np.argmax(
    np.tensordot(weights.reshape(1, weights.shape[0]), outputs, axes=1), axis=2
).reshape(
    (outputs.shape[1],)
)  # pylint: disable=unsubscriptable-object

# Load labels.
source = getattr(dataloader, "test_set" if args.eval_split == "test" else "val_set", None)
if source is not None and hasattr(source, "targets"):
    indices = np.arange(datasetfile[f"nb_{args.eval_split}"]) if args.eval_split == "test" else dataloader.val_indices
    labels = np.asarray(source.targets)[indices]
elif hasattr(dataloader, "y_test") and args.eval_split == "test":
    labels = dataloader.y_test
elif hasattr(dataloader, "y_train") and args.eval_split == "val" and hasattr(dataloader, "val_indices"):
    labels = dataloader.y_train[dataloader.val_indices]
else:
    _, labels = dataloader.load(np.arange(datasetfile[f"nb_{args.eval_split}"]), category=args.eval_split)

# Compute and print accuracy.
accuracy = (
    np.where(votes == labels)[0].shape[0] / outputs.shape[1]
)  # pylint: disable=unsubscriptable-object
print(accuracy)

if args.save_evaluation:
    from comparison_export import sisa_run_id
    from reporting import classification_evaluation, record_evaluation
    split_path = directory / "dataset_split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    dataset_id = split.get("dataset_id", __import__("pathlib").Path(args.dataset).parent.name)
    # CIFAR loaders expose the original ordered pool, before transforms.
    source_hash = None
    source = getattr(dataloader, "test_set" if args.eval_split == "test" else "val_set", None)
    if source is not None and hasattr(source, "data") and hasattr(source, "targets"):
        from dataset_splits import fingerprint_cifar
        source_hash = fingerprint_cifar(source.data, source.targets)
    evaluation = classification_evaluation(
        directory, run_id=sisa_run_id(directory), state_id=f"step-{args.label}", step=int(args.label),
        split=args.eval_split, dataset=dataset_id, labels=labels, predictions=votes,
        scores=np.tensordot(weights, outputs, axes=1),
        num_classes=datasetfile["nb_classes"], split_sha256=split.get("manifest_sha256"),
        source_sha256=source_hash, method_details={"aggregation": args.strategy, "score_type": "votes_or_probabilities"})
    record_evaluation(directory, evaluation)
