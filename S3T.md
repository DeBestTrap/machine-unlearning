# S3T experiments

This repository includes an independent implementation of Sequence-aware Sharded
Sliced Training for CIFAR-10/CIFAR-100 with ViT and SST-2 with RoBERTa. The method
is described in [Chowdhury et al., ICLR 2025](https://arxiv.org/abs/2406.16257v3).
Read [implementation decisions](docs/s3t-decisions.md) before interpreting results
as a reproduction of the paper's reported numbers.
The [validation report](docs/s3t-validation.md) records the tests and real-data
runs completed in this checkout.

## Install and check

Use a compatible PyTorch/torchvision installation for your CUDA version, then:

```bash
pip install -r requirements-s3t.txt
python -m unittest discover -s tests -p 'test_s3t_*.py'
python -m s3t smoke --output runs/s3t-smoke --device cpu
python -m s3t smoke --output runs/s3t-smoke-roberta --model tiny_roberta --device cpu
```

Smoke runs train real small transformers on generated data without downloading
weights. They exercise checkpointing, prediction, and saved deletion replay.

The default head policy is `first_slice`: train the classifier in the first
stage, then freeze it for every later stage. `snapshot` remains an explicit
alternative. Existing runs retain the head policy in their saved configuration.

Run commands from this repository. Dataset/config paths inside a config resolve
relative to this repository; CLI run/output paths resolve from the working
directory. CIFAR archives must already be extracted in `data_root` (default `.`).
SST-2 uses the verified arrays under `data_root/datasets/sst2/`; prepare missing
arrays with `python datasets/sst2/prepare_data.py`. Initializing a pretrained run
may download weights. Subsequent loading of its frozen backbone is offline.

## Train and resume

```bash
python -m s3t train --config configs/s3t/vit-cifar10.json \
  --output runs/s3t-c10 --device cuda:0

python -m s3t train --run runs/s3t-c10 --device cuda:0
python -m s3t status --run runs/s3t-c10
```

Training resumes from the last completed epoch and skips completed members.
Every stage starts a fresh optimizer. Recovery within a stage restores optimizer
and scheduler state. Epoch-local randomness reproduces shuffled batches,
augmentations, and dropout on the same device/software setup. An interrupted
epoch is rerun. Configurations, model identity, partitions, and saved split
identity are checked; a different configuration requires a new output directory.

The configs are starting settings, not validated optimal hyperparameters:

| Config under `configs/s3t/` | Backbone | Slices | Layers per slice |
| --- | --- | ---: | ---: |
| `vit-cifar10.json` | torchvision ViT-B/16, ImageNet-1K | 6 | 2 |
| `vit-cifar100.json` | torchvision ViT-B/16, ImageNet-1K | 6 | 2 |
| `hf-vit-cifar10.json` | HF ViT-B/16, ImageNet-21K | 6 | 2 |
| `roberta-sst2.json` | RoBERTa-base | 6 | 2 |
| `roberta-large-sst2.json` | RoBERTa-large | 7 | 3 |

Override a config before creating a run using `--set key=value`, for example:

```bash
python -m s3t train --config configs/s3t/vit-cifar10.json \
  --output runs/s3t-c10-small-budget --device cuda:0 \
  --set shards=2 --set slices=2 --set layers_per_slice=6 \
  --set budget=2 --set epochs_per_slice=1
```

`epochs_per_slice` means epochs over **each cumulative prefix**. `budget` is the
number of sequences trained **per shard**, bounded by `slices!`. For identical
partitions and deletions across methods, use the saved source-ID manifests; an
identical seed alone does not establish identical examples.

To run independent members on multiple GPUs, initialize once, then give each
process disjoint `--shard` and optionally `--sequence` indices (zero based):

```bash
python -m s3t init --config configs/s3t/vit-cifar10.json --output runs/s3t-c10
# In separate terminals/jobs:
python -m s3t train --run runs/s3t-c10 --shard 0 --device cuda:0
python -m s3t train --run runs/s3t-c10 --shard 1 --device cuda:1
```

Process locks reject duplicate writers. All configured members must finish
before ensemble evaluation/deletion. These example jobs cover two shards;
schedule the remaining configured shards as well.

## Evaluate and delete

```bash
python -m s3t evaluate --run runs/s3t-c10 --split val --device cuda:0
python -m s3t evaluate --run runs/s3t-c10 --split test --device cuda:0

python -m s3t replay --run runs/s3t-c10 \
  --requests deletions/cifar10-seed123-100x1.json --device cuda:0
```

Replay validates the shared split/workload checksums and applies saved batches
of original training-source IDs. It resumes in order after interruption; using
another workload or mixing manual deletions into a workload is rejected.
`--steps 10` means replay through step 10, not ten additional steps.

Default replay stops when every shard has lost every usable sequence. The final
row has `status=requires_retraining` and an empty accuracy. No random/untrained
model is silently substituted. Choose an explicit alternative when needed:

```bash
# Continue recording the workload after exhaustion, with empty accuracies.
python -m s3t replay --run runs/s3t-c10 \
  --requests deletions/cifar10-seed123-100x1.json --on-exhaustion continue

# Rebuild on retained examples whenever the ensemble is exhausted.
python -m s3t replay --run runs/s3t-c10 \
  --requests deletions/cifar10-seed123-100x1.json --on-exhaustion retrain --device cuda:0

# Or explicitly rebuild the current ensemble after any deletions.
python -m s3t retrain --run runs/s3t-c10 --device cuda:0
```

Retraining creates a new generation using the same frozen backbone and all
retained training examples. It deterministically reshuffles/repartitions that
pool. A generation becomes active only after every member finishes. Interrupted
retraining resumes with `retrain`. There must still be enough retained examples
to fill the configured shards and slices. Further deletions are rejected while
a generation is pending, so its training membership cannot change mid-run.

For a separate manual-deletion experiment:

```bash
python -m s3t delete --run runs/manual-example --source-ids 123 456
```

The IDs must belong to that run's original training pool. Deletion is cumulative;
duplicates, validation IDs, and unknown IDs are rejected. Earlier steps remain
available with `evaluate --step N` for research comparisons. This is model
unlearning, not a tool for purging historical artifacts from disk.

## Outputs and measurement

Each run contains:

- `config.json`, `metadata.json`, `dataset_split.json`: resolved settings, source
  identities, package versions, Git state, and architecture/parameter counts.
- `source_snapshot/`: the S3T implementation and shared manifest utilities used
  when the run was initialized, including uncommitted source files.
- `backbone.pt`, `backbone.json`: one shared frozen backbone and its architecture.
- `generations/NNNN/partition.json`: every source ID's shard/slice and all sequences.
- `generations/NNNN/members/*.pt`: one compact checkpoint per sequence, including
  all adapter groups and prefix head states. During training it also contains
  optimizer/scheduler recovery state. Small JSON sidecars expose progress.
- `state.json`, `deletion_trace.json`, `deletion_requests.json`: current generation,
  cumulative deletions, each surviving prefix, selections, and workload identity.
- `evaluations/*.json`, `results.csv`: metrics for each evaluated step and pool.

Evaluation caches predictions by generation/member/prefix/pool to make long
deletion sweeps practical. Use `--no-cache` for fresh inference timings. Each
result records cache hits. Timing fields deliberately distinguish:

- `generation_training_seconds`: sum of synchronized training-epoch wall times,
  including batch loading, excluding model initialization, validation, and saves.
- `generation_validation_seconds`: validation performed during training.
- `deletion_planning_seconds`: dependency/selection bookkeeping only. This does
  **not** measure a production deployment's complete unlearning latency.
- `evaluation_seconds`: ensemble loading, prefix activation, and evaluation, or
  cached prediction loading. Compare fresh evaluations separately from cache hits.

`generation_checkpoint_bytes` includes all sequence checkpoints/head histories;
`backbone_bytes` counts the shared base once. Prediction caches, source snapshots,
and retired generations are not included in those two storage metrics.
`peak_gpu_allocated_bytes` is the maximum PyTorch allocation observed while
training/validating a member, not total system GPU memory.

`max_train_examples` and `max_eval_examples` are diagnostic caps. Results label
their presence; shared deletion replay rejects an actual training-pool cap.
The split seed, partition seed, sequence seed, training seed, and workload seed
have separate roles and are all recorded.

To regenerate the aggregate CSV after evaluating more points:

```bash
python -m s3t export --run runs/s3t-c10
python -m s3t audit --run runs/s3t-c10
```

The audit checks saved source identities, every surviving prefix's dependencies,
head-snapshot availability, and sequence selections against the workload. It
does not rerun training or constitute a statistical privacy audit.

## Learning-rate screening

The sequential screening runner uses `first_slice`, BMS with a uniform deletion
prior, and the full shared validation pools. It trains one representative shard
and one sequence per trial while retaining a five-shard/six-slice partition.
Training length is 15 epochs per prefix for CIFAR-10 and 10 for SST-2. The grids
are `[2e-4, 1e-3, 2e-3]` for CIFAR-10 and `[5e-6, 1e-5, 2e-5]` for SST-2.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -u -m s3t.lr_search run --output runs/s3t-lr-screen --device cuda:0

python -m s3t.lr_search status --output runs/s3t-lr-screen
```

Rerunning `run` resumes incomplete training and skips completed trials. A lock
prevents two queue processes from training the same search. Stop the previous
queue and its active worker before restarting an interrupted queue.

The search directory contains a live `report.md`, `results.csv`, `status.json`,
per-trial logs under `logs/`, and exact trial configs. Each completed trial has
`pilot_result.json` with final-prefix validation accuracy and initialization,
training, validation, and total wall times. The test pool is not scored. The
remaining-time estimate extrapolates observed epoch times by cumulative-prefix
size and transformer depth; initial estimates use earlier local verification
runs when available.

These are single-member screening scores. Confirm promising learning rates with
the intended full ensemble and additional seeds before reporting final results.
BMS runs its matching procedure here; uniform priors do not test its advantage
from knowing nonuniform deletion probabilities.
