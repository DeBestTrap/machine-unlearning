# Shared deletion requests

BlockUnlearn and `machine-unlearning` can replay the same saved deletion batches.
Requests identify **original training-source indices**, before the validation
holdout, sharding, or tokenization. A workload contains the dataset/source identity,
the shared split checksum, ordered batches, a deletion seed, and a workload checksum.
Both methods validate these before replay. Deletions are unique across all batches.

The workload JSON files in `BlockUnlearn/unlearning/deletions/` have byte-identical
copies in `machine-unlearning/deletions/`:

| Workload | Steps | Points per step | Total deleted |
| --- | ---: | ---: | ---: |
| `cifar10-seed123-100x1.json` | 100 | 1 | 100 |
| `cifar100-seed123-100x1.json` | 100 | 1 | 100 |
| `cifar100-seed123-100x10.json` | 100 | 10 | 1,000 |
| `glue-sst2-seed123-100x1.json` | 100 | 1 | 100 |

## BlockUnlearn

Run from `BlockUnlearn/unlearning/`. Add this to a CIFAR-100 experiment's existing
configuration (keep its strategy, replacement, and evaluation settings):

```yaml
dataset_args:
  max_train_examples: null
unlearning_args:
  type: random_datapoint
  request_manifest: deletions/cifar100-seed123-100x10.json
```

`request_manifest` determines the step count and each batch's size. In this mode,
`num_sequential`, the sequential ratios, and `num_unlearned_points` are ignored.
Training/experiment seeds do not affect the workload. Without `request_manifest`,
the previous selection behavior remains available. Paths are relative to the
process working directory, like explicit split paths.

Replay supports CIFAR (ViT, ResNet, DenseNet) and the shared GLUE loaders,
including SST-2. It works in regular, naive-retraining, and training-group
unlearning. For comparisons, use the same full training pool in both repos;
actual training caps are rejected. Expanded datasets such as SQuAD token windows
are not supported by this replay path.

Runs save `deletion_requests.json`, `deletion_replay.json` (local-to-source mapping),
and per-step local/source indices, cumulative source deletions, workload checksum,
and model-run number in `unlearning_trace.json`.

## SISA

Run from `machine-unlearning/`. Use a **new container**:

```bash
python distribution.py --shards 5 --label 0 \
  --dataset datasets/cifar100/datasetfile --container aligned-c100 \
  --request-manifest deletions/cifar100-seed123-100x10.json
```

This creates `requestfile:0.npy` (empty), then cumulative request files for labels
1 through 100. Label 1 excludes the first 10 points; label 2 excludes the first 20.
The stored indices are local training positions, distributed according to this
container's shards. Shard layouts need not match BlockUnlearn's layouts.

Existing `example-scripts/common/init.sh` and callers such as `run_all.sh` also
support an environment variable:

```bash
DELETION_REQUEST_MANIFEST=deletions/cifar100-seed123-100x10.json \
  bash example-scripts/common/init.sh datasets/cifar100/datasetfile 5 100 aligned-c100
```

Here `ul_reqs=100` means 100 **batches**, and must match the manifest's batch count.
Training, prediction, and report scripts continue to use labels 0..100. SISA stores
the workload in its container and rejects conflicting request files, different
workloads, random request overwrites, and reinitialization of shared containers.
To regenerate matching request files, omit `--shards`.

## Generate another workload

From `BlockUnlearn/unlearning/`:

```bash
python deletion_requests.py \
  --split-manifest splits/cifar100-val2500-seed42.json \
  --output deletions/cifar100-seed456-20x5.json \
  --seed 456 --steps 20 --batch-size 5
cp deletions/cifar100-seed456-20x5.json ../../machine-unlearning/deletions/
```

Generation uses its own RNG without advancing training randomness. Reuse or copy
the resulting file: never independently sample requests during either method's
unlearning. The generator refuses to overwrite a different saved workload.

## Audit actual run artifacts

From `BlockUnlearn/unlearning/`, after both methods have produced their artifacts:

```bash
python scripts/verify_shared_deletions.py \
  --split-manifest splits/cifar100-val2500-seed42.json \
  --request-manifest deletions/cifar100-seed123-100x10.json \
  --sisa-container ../../machine-unlearning/containers/aligned-c100 \
  --block-run /absolute/path/to/blockunlearn/run \
  --output deletion-audit.json
```

The audit independently decodes the actual local indices in every BlockUnlearn
trace event and SISA request file. It checks batch order, all cumulative sets,
unique deletion counts, split/workload identity, and SISA shard membership. Every
model run in the trace must contain the complete workload. Omitting `--block-run`
audits SISA's prepared files alone; it does not establish cross-method replay.

Cross-repository smoke tests exercise both production orchestration paths,
BlockUnlearn's removal sampler and SISA's `fetchShardBatch` on the real saved
split IDs. Model retraining is stubbed; they verify requests and filtering,
not model accuracy or unlearning quality:

```bash
python -m unittest discover -s tests -p 'test_deletion_*.py'
python -m unittest discover -s tests -p test_shared_deletions_integration.py
```

Set `SHARED_DELETION_VERIFICATION_DIR` to a new directory to retain smoke-test
traces, SISA request files, and per-case `report.json` files. The sibling SISA repo
must be present for cross-repo tests. Its own deletion tests run from that repo
with `python -m unittest discover -s tests -p 'test_deletion_*.py'`.

A separate offline DenseNet test uses real training and retraining on a small
synthetic dataset for both training-group and naive unlearning:

```bash
PYTHONPATH=tests python -m unittest \
  test_densenet.DenseNetTests.test_shared_manifest_with_real_training_and_retraining
```
