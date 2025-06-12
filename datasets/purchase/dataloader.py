import os
import numpy as np

pwd = os.path.dirname(os.path.realpath(__file__))

# ── load & unwrap the dicts ---------------------------------------------------
train_data = np.load(os.path.join(pwd, "purchase2_train.npy"),
                     allow_pickle=True).item()
test_data  = np.load(os.path.join(pwd, "purchase2_test.npy"),
                     allow_pickle=True).item()

X_train = train_data["X"].astype(np.float32)
y_train = train_data["y"].astype(np.int64)

X_test  = test_data["X"].astype(np.float32)
y_test  = test_data["y"].astype(np.int64)

def load(indices, category="train"):
    int_indices = indices.astype(np.int64)

    if category == "train":
        return X_train[int_indices], y_train[int_indices]
    else:  # "test"
        return X_test[int_indices], y_test[int_indices]