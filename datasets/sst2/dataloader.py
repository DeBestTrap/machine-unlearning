import os
import numpy as np
from pathlib import Path
from split_support import configured_manifest

pwd = os.path.dirname(os.path.realpath(__file__))

train_path = os.path.join(pwd, "sst2_train.npz")
test_path = os.path.join(pwd, "sst2_test.npz")

if not (os.path.exists(train_path) and os.path.exists(test_path)):
    raise FileNotFoundError(
        "SST-2 arrays not found. Run datasets/sst2/prepare_data.py first."
    )

train_data = np.load(train_path)
test_data = np.load(test_path)

X_train = train_data["X"]
y_train = train_data["y"].astype(np.int64)

X_test = test_data["X"]
y_test = test_data["y"].astype(np.int64)


if "source_sha256" not in train_data:
    raise ValueError("SST-2 arrays predate verified splits. Re-run datasets/sst2/prepare_data.py.")
manifest = configured_manifest(
    Path(__file__).with_name("datasetfile"), dataset_id="glue/sst2",
    source_size=len(X_train), source_sha256=str(train_data["source_sha256"].item()),
)
train_indices = np.asarray(manifest["train_indices"], dtype=np.int64)
val_indices = np.asarray(manifest["val_indices"], dtype=np.int64)


def load(indices, category="train"):
    indices = np.asarray(indices, dtype=np.int64)
    if category == "train":
        source_indices = train_indices[indices]
        return X_train[source_indices], y_train[source_indices]
    if category in ("val", "validation"):
        source_indices = val_indices[indices]
        return X_train[source_indices], y_train[source_indices]
    if category == "test":
        return X_test[indices], y_test[indices]
    raise ValueError(f"Unknown dataset category: {category}")
