import json
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

pwd = os.path.dirname(os.path.realpath(__file__))

MAX_LEN = 128
TOKENIZER_NAME = "bert-base-uncased"


def _tokenize(batch, tokenizer):
    return tokenizer(
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
    )


def _to_arrays(split):
    input_ids = np.asarray(split["input_ids"], dtype=np.int64)
    labels = np.asarray(split["label"], dtype=np.int64)
    return input_ids, labels


def main():
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokenized = dataset.map(lambda b: _tokenize(b, tokenizer), batched=True)
    train = tokenized["train"]
    validation = tokenized["validation"]

    X_train, y_train = _to_arrays(train)
    X_test, y_test = _to_arrays(validation)

    np.savez_compressed(os.path.join(pwd, "sst2_train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(pwd, "sst2_test.npz"), X=X_test, y=y_test)

    datasetfile = {
        "nb_train": int(X_train.shape[0]),
        "nb_test": int(X_test.shape[0]),
        "input_shape": [int(X_train.shape[1])],
        "nb_classes": 2,
        "dataloader": "dataloader",
    }

    with open(os.path.join(pwd, "datasetfile"), "w") as f:
        json.dump(datasetfile, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
