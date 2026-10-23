import json
import os
from pathlib import Path
import sys
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset_splits import fingerprint_records, load_manifest

import numpy as np
from transformers import AutoTokenizer

pwd = os.path.dirname(os.path.realpath(__file__))

MAX_LEN = 128
TOKENIZER_NAME = "roberta-base"


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

    source_sha256 = fingerprint_records(
        {field: row[field] for field in ("sentence", "label")} for row in dataset["train"]
    )
    metadata_path = Path(pwd) / "datasetfile"
    metadata = json.loads(metadata_path.read_text())
    manifest_relative_path = metadata["split_manifest"]
    manifest_path = Path(pwd) / manifest_relative_path
    manifest = load_manifest(manifest_path, dataset_id="glue/sst2",
                             source_size=len(dataset["train"]), source_sha256=source_sha256)
    tokenized = dataset.map(lambda b: _tokenize(b, tokenizer), batched=True)
    train = tokenized["train"]
    validation = tokenized["validation"]

    X_train, y_train = _to_arrays(train)
    X_test, y_test = _to_arrays(validation)

    np.savez_compressed(os.path.join(pwd, "sst2_train.npz"), X=X_train, y=y_train, source_sha256=source_sha256)
    np.savez_compressed(os.path.join(pwd, "sst2_test.npz"), X=X_test, y=y_test)

    datasetfile = {
        "nb_train": len(manifest["train_indices"]),
        "nb_val": len(manifest["val_indices"]),
        "split_manifest": manifest_relative_path,
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
