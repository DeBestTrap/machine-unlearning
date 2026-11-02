# ProtoSISA

`python -m protosisa run` implements the classification baseline described in
BlockUnlearn's `.paper/sections/experiments.tex`: uniform random SISA sharding,
an independent task model per shard, and a global prototype classifier computed
from the original pretrained backbone. The default shard count equals the task's
output size: 10 for CIFAR-10, 100 for CIFAR-100, and 2 for SST-2. The selected
slice counts from `.paper/sections/appendix_implementation.tex` are 1, 1, and 2.

For a query `x`, inference computes

```text
z(x)        = frozen_pretrained_encoder(x)
p[c]        = mean(normalize(z(x_i)) for retained training examples of class c)
proto[c]    = cosine_similarity(z(x), p[c])
sisa        = mean(shard_model(x) for all shards)   # raw logits
lambda      = exp(-mean_retained_shard_size / 100)
scores      = (1 - lambda) * sisa + lambda * proto
prediction  = argmax(scores)
```

The default `--prototype-weighting safe` implements the published ProtoSISA
weighting in [SAFE, section 7](https://arxiv.org/pdf/2304.13169):
`lambda = exp(-d * mean_shard_size / 100)`. Independent SISA shards have `d=1`.
This implementation uses the current retained example count divided by the
fixed number of shards, including empty shards. It recomputes the weight after
each deletion, consistent with applying the procedure to the retained data.
Prototype cosine scores are unscaled. There is no mixture-weight search in
SAFE mode.

There is no softmax, majority vote, or split of the training data between the
two components. Both components use the full retained training pool under their
respective training rules. Prototypes contribute only at inference and never
enter shard training or its loss.

The implementation follows the local paper's independent
model-per-shard procedure using the existing full ViT/RoBERTa task models;
it implements SAFE's prototype weighting, rather than its shard graphs or InCA
adapter training. Configuration records identify the weighting mode. Every
step/evaluation records the resolved weight, retained count, mean shard size,
and `d=1`.

An unscaled fixed mixture is available with
`--prototype-weighting fixed --prototype-weight 0.5`. SAFE mode computes its own
weight. Older saved configurations without a weighting-mode field retain their
fixed-weight interpretation. Temperature scaling has been removed; calibrated
historical runs remain archived and require their launch source to reproduce.

## Run

From the repository root, after preparing the ordinary CIFAR/SST-2 data and
pretrained model caches:

```bash
python -m protosisa run --model vit \
  --dataset datasets/cifar10/datasetfile --container proto-c10-v1 \
  --requests deletions/cifar10-seed123-25x1.json \
  --epochs 1 --batch-size 64 --learning-rate 0.0001 --device cuda:0

python -m protosisa run --model roberta \
  --dataset datasets/sst2/datasetfile --container proto-sst2-v1 \
  --requests deletions/glue-sst2-seed123-25x1.json \
  --epochs 1 --batch-size 32 --learning-rate 0.00001 --device cuda:0
```

Omit `--requests` for initial training and evaluation only. `--shards`,
`--slices`, `--prototype-weighting`, `--seed`, `--optimizer`, `--dropout-rate`, and
`--eval-split val|test` allow explicit variants. The defaults use seed 1, AdamW,
one nominal SISA epoch, and test evaluation. Standalone learning rate defaults
to 0.0001; set 0.00001 for the SST-2 comparison. Only ViT and RoBERTa feature
encoders are supported.

The [10% CIFAR-10 pilot](studies/2026-10-30_1555_protosisa_pilot/README.md)
compares the initial one-epoch setup with three epochs. Its SAFE-weighting
evaluation obtains 92.06% official test accuracy after deletion. At 4,749
retained examples across ten shards, SAFE gives the prototypes weight 0.00866035.
See the [SAFE-weighting results](studies/2026-10-30_1555_protosisa_pilot/runs/cifar10_10pct_v1/safe_weighting_v1/report.md).
The shared comparison preset now uses `protosisa.prototype_weighting="safe"`.
Custom manifests can select `fixed` and specify its weight explicitly.

The shared study accepts the method explicitly, with the same verified splits
and 25 ordered deletion source IDs as the other methods:

```bash
python studies/2026-10-26_2348_shared_comparison/run.py \
  --methods protosisa --datasets cifar10 cifar100 sst2 \
  --device 0 --tag proto25_v1 --dry-run
```

Remove `--dry-run` to execute. Select `--methods sisa protosisa s3t` for all
three baselines. The study's existing default selection remains SISA/S3T.
ProtoSISA's larger shard counts entail more models than the five-shard SISA
preset. Study records live at `runs/<tag>/protosisa-<dataset>/` under the study.

## Deletion and reproducibility

Sharding happens after applying the shared training holdout. A seeded uniform
permutation assigns every training example to exactly one shard. Portable
source IDs are translated to local training indices by the existing deletion
exporter; validation/test examples never enter either training component.

Shard training uses SISA's cumulative prefixes and integer epoch allocation
(`2 * nominal_epochs / (slices + 1)` epochs per slice before rounding).
The last prefix includes all remainder examples. Deletion restores the last
unaffected slice boundary and retrains later prefixes. Slice checkpoints include
model **and optimizer** state. Per-shard initialization and per-slice RNG seeds
are fixed, including augmentation/dropout; interrupted slices restart from their
boundary. This defines the retained-data reference training procedure, and CPU
tests compare deletion results to fresh retained-data training exactly. GPU
kernel nondeterminism can still affect bitwise reproducibility.

The frozen encoder is separate from all task-trained models. ViT uses the
ImageNet-pretrained CLS representation before its classification head; RoBERTa
uses the pretrained final-layer CLS token with a padding attention mask.
`train_eval` selects the same training indices with deterministic evaluation
preprocessing, so CIFAR prototypes do not depend on random crops. Unit feature
sums and class counts accumulate in float64. Combining shard contributions gives
global sample-weighted prototypes, including uneven shard sizes.

When a deletion changes a shard, its prototype statistics are rebuilt from its
retained members. Unchanged contributions are reused. This avoids subtraction
drift and needs no stored individual embeddings. An empty shard contributes zero
counts; an absent class gets cosine score -1. Deleting the entire training pool
is unsupported because no prototype classifier remains.

Re-running the identical command resumes completed checkpoints while preserving
their timing journals. Configuration, source, loader, split, shard membership,
and deletion identity checks reject incompatible reuse. Use a fresh container
after changing the experiment. Like the existing SISA cache, historical slice
models and prototype statistics remain available until explicit model cleanup;
deletion updates the active predictor rather than purging historical artifacts.

## Measurements and cleanup

The [expected replay cost calculator](TIMING.md#analytical-expected-replay-cost)
uses the paper's SISA formula for ProtoSISA's shard training as well. For example,
`python -m sisa_cost --shards 2 10 100 --slices 1 2` compares normalized costs
without training. Prototype maintenance adds work outside that analytical model.

Each state saves accuracy, macro/weighted F1, balanced accuracy, confusion matrix,
and compact prediction/label/score arrays using the shared reporting schema with
`method=protosisa`. `run.json`, `steps.jsonl`, `evaluations.jsonl`, `report.csv`,
and prototype counts remain available without model checkpoints.
`outputs/components-step-<step>-<split>.npz` additionally retains the separate
ensemble logits, unscaled prototype cosine scores, and ordered labels. Component
accuracies and the resolved mixture weight are included in evaluation metadata. Training
prints per-shard epoch loss, accuracy, example counts, and elapsed time.

`update_time_sec` sums serial shard update journals plus prototype construction,
cache lookup/load, aggregation, and checkpoint writes. `training_time_sec` is
gradient training only. Per-state evaluation time includes loading/evaluating
the shard models and prototype inference. Initial dataset/encoder setup and
report generation are outside those intervals. Failed slice work is retained
in timing totals; an uncleanly interrupted journal remains marked incomplete.

Model and prototype `.pt` files use the existing `saved_models/` cache and are
included in checkpoint-size measurement and ordinary cleanup:

```bash
bash clear_containers.sh proto-c10-v1
python -m reporting refresh containers/proto-c10-v1 --output /tmp/proto-report
```

Training/evaluation after checkpoint cleanup requires a new run. CPU tests with
small injected models exercise the full split/workload/reporting pipeline
without downloading pretrained models or training a full benchmark.
