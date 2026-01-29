import json
import os
from pathlib import Path

import numpy as np
from torchvision import datasets


def main():
    pwd = os.path.dirname(os.path.realpath(__file__))
    repo_root = Path(pwd).resolve().parents[1]

    train_set = datasets.CIFAR10(root=str(repo_root), train=True, download=True)
    test_set = datasets.CIFAR10(root=str(repo_root), train=False, download=False)

    X_train = np.asarray(train_set.data)
    y_train = np.asarray(train_set.targets, dtype=np.int64)
    X_test = np.asarray(test_set.data)
    y_test = np.asarray(test_set.targets, dtype=np.int64)

    np.savez_compressed(os.path.join(pwd, "cifar10_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(pwd, "cifar10_test.npz"), X=X_test, y=y_test)

    datasetfile = {
        "nb_train": int(X_train.shape[0]),
        "nb_test": int(X_test.shape[0]),
        "input_shape": [3, 224, 224],
        "nb_classes": 10,
        "dataloader": "dataloader",
    }

    with open(os.path.join(pwd, "datasetfile"), "w") as f:
        json.dump(datasetfile, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
