# Shared validation pools

CIFAR-10, CIFAR-100, and SST-2 use the same saved split files as BlockUnlearn. These files select examples from the original training source; sharding and slicing operate on the remaining training pool. The default holdout is 5%, generated once with split seed 42.

| Dataset | Training | Validation | Final evaluation |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 47,500 | 2,500 | 10,000 official test examples |
| CIFAR-100 | 47,500 | 2,500 | 10,000 official test examples |
| SST-2 | 63,982 | 3,367 | 872 official development examples |

Each dataset's `datasetfile` points to a manifest in `splits/` and declares the reduced `nb_train` and new `nb_val`. The common loaders map local training indices to `train_indices`, local validation indices to `val_indices`, and test indices to the official final evaluation source. They verify source identity before loading. Training and validation are disjoint, including across different experiment seeds. Validation uses deterministic preprocessing.

## New experiments

Create a **new container** with the usual initialization and `distribution.py --shards` commands before training. New containers store `dataset_split.json` and refuse mismatched splits. Existing containers and checkpoints from before this change cannot be reused because training membership has changed. CIFAR final evaluation now explicitly loads `train=False`; old evaluation outputs should be regenerated with newly trained models.

SST-2's full source arrays include a fingerprint of the raw sentence/label sequence. Existing arrays must be prepared with `python datasets/sst2/prepare_data.py`; this command verifies the configured manifest against the raw source before saving. The local existing training array was verified against every raw label and tokenized sentence and given this provenance without changing its examples.

No change to existing run commands is required for final test evaluation. To score the fixed validation pool instead, add `--eval_split val` to both prediction and aggregation commands. For example, after training a 3-shard CIFAR-10 container named `example`:

```bash
for shard in 0 1 2; do
  python sisa.py --model vit --test --dataset datasets/cifar10/datasetfile \
    --container example --shard "$shard" --label 0 --eval_split val
done
python aggregation.py --dataset datasets/cifar10/datasetfile \
  --container example --shards 3 --label 0 --eval_split val
```

Validation predictions go under `containers/example/outputs/val/`; test predictions retain their usual output paths. Label 0 assumes the initial request file/checkpoints were created with that label.

## Sharing and custom pools

The checked-in JSON manifests are byte-identical to those in `../BlockUnlearn/unlearning/splits/`. They include original source indices, a raw-data fingerprint, dataset identity, split settings, format version and checksum. `dataset_splits.py` is the same portable implementation in both repos. Its loader uses only the standard library; generation requires PyTorch.

For another holdout size or seed, generate the manifest in BlockUnlearn, copy it into this repo's `splits/`, then update `split_manifest`, `nb_train` and `nb_val` in the dataset's `datasetfile`. Manifest paths are relative to that metadata file. Start a new container. SST-2 preprocessing follows the configured manifest as well.

Code for ProtoSISA, S3T, or other methods can use `configured_manifest(datasetfile)` and the common `load(..., category="train"|"val"|"test")` interface. Code with separate data loaders must apply the saved indices before making its own shards. The checkout currently contains the common SISA implementation; separate implementations must use this interface to share the pools.

Local positions map to original examples through the saved index list. For example, `manifest['train_indices'][local_training_index]` identifies an original source example. Matching deletion requests across methods requires sharing those source identities separately from the train/validation split.

Use `distribution.py --request-manifest` or `DELETION_REQUEST_MANIFEST` with the
common initialization scripts to replay the same deletion batches as BlockUnlearn.
See [Shared deletion requests](DELETIONS.md) for saved workloads, cumulative-label
semantics, and the cross-repository audit command.

Run the offline checks with:

```bash
python -m unittest discover -s tests -p test_dataset_splits.py
```
