import os
import numpy as np

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


def load(indices, category="train"):
    int_indices = indices.astype(np.int64)

    if category == "train":
        return X_train[int_indices], y_train[int_indices]
    else:
        return X_test[int_indices], y_test[int_indices]
