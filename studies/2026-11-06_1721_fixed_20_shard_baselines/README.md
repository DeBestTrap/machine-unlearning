# Fixed-20-shard SISA, ProtoSISA, and S3T searches

This is the baseline side of the fixed-20-shard main-table study. It covers
ResNet-50/CIFAR-10, ViT-B/16/CIFAR-100, DenseNet-121/CIFAR-100, and
RoBERTa-base/SST-2. SU and BM
plans live in the sibling BlockUnlearn study
`unlearning/studies/2026-11-06_1716_fixed_20_shard_search/`. Both sides use
exactly the same copied seed-42 CIFAR source-index manifests: 45,000 final
training examples and a 5,000-example development pool split into 3,000 fitting,
1,000 search scoring, and 1,000 frozen-finalist confirmation examples. SST-2
uses 60,615 final-training examples and 4,042/1,346/1,346 development examples
from the original ordered training source. Prepared SST-2 token arrays are bound
to their raw-source fingerprint and their own token checksum. The
official test set is not accessed during these searches.

`baseline.py` is a study-local copy of the verified SISA/ProtoSISA development
driver. One trained SISA ensemble supplies both SISA and ProtoSISA scores.
At 20 shards, the 100 candidate configurations vary valid whole-epoch slice
schedules and learning rates; the expected-cost strata are 0.10, 0.25, and
0.50 relative to one full-data epoch. The top three per method and cost ceiling
are frozen before fresh three-seed confirmation. This analytical SISA cost
excludes prototype-maintenance work, which must be measured in the full-data
comparison. Completed model weights are
discarded after predictions and metrics have been saved. Prototype features
are built from development fitting examples only.

`s3t.py` is a study-local copy of the complete-ensemble S3T development driver.
Each candidate has exactly 20 shards and varies slice count, sequence budget,
whole epochs per slice, and LR. The 24 deterministic screening candidates have
20–80 ensemble members; three finalists are frozen and repeated at seeds 1/2/3
on the confirmation pool. The ResNet and DenseNet paths are explicit
convolutional-LoRA adaptations; DenseNet attaches independent branches to
each dense layer's `conv1` and `conv2`. The ViT and RoBERTa paths use attention
adapters. Batch-normalization statistics stay fixed in the CNN adapters. S3T sequence
switching does not share SU/BM's analytical deletion-work ceiling, so the study
records training work and member counts separately. Full-data deletion and
wall-clock comparisons remain separate experiments.

All eight configs and the exact parent/development manifests are under
`configs/`. The CIFAR training archive is expected under this repository root
(or use S3T's `--data-root` argument). The SISA driver checks the source
fingerprint during preparation. The configs leave the GPU power-limit guard
unset, allowing different hardware limits. They still require at least 20 GiB
free on the relevant disks. Native records remain under each engine's normal
record store; generated plans, logs, predictions, reports, and resume state are
under this study's `runs/<run-name>/`.

## Preparation only

From the repository root, these commands validate manifests and write plans;
they do **not** train models:

For SST-2, first run `python datasets/sst2/prepare_data.py`. The preparation
script loads `nyu-mll/glue` with the `sst2` configuration and writes the original
ordered RoBERTa token arrays to `datasets/sst2/sst2_train.npz`, including the raw
source fingerprint. Having only the dataset scripts or a Hugging Face cache is
not sufficient. S3T's `--data-root` points to the repository root containing
`datasets/`, not to `datasets/` itself. The searches consume the original training
arrays and their fixed 10% manifests; they do not load `sst2_test.npz`.

```bash
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh baseline prepare \
  --config studies/2026-11-06_1721_fixed_20_shard_baselines/configs/baseline_resnet50_cifar10.json \
  --name resnet50_cifar10_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh baseline prepare \
  --config studies/2026-11-06_1721_fixed_20_shard_baselines/configs/baseline_vit_b16_cifar100.json \
  --name vit_b16_cifar100_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh baseline prepare \
  --config studies/2026-11-06_1721_fixed_20_shard_baselines/configs/baseline_densenet121_cifar100.json \
  --name densenet121_cifar100_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh baseline prepare \
  --config studies/2026-11-06_1721_fixed_20_shard_baselines/configs/baseline_roberta_base_sst2.json \
  --name roberta_base_sst2_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh s3t prepare \
  --config configs/s3t_resnet50_cifar10.json --name s3t_resnet50_cifar10_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh s3t prepare \
  --config configs/s3t_vit_b16_cifar100.json --name s3t_vit_b16_cifar100_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh s3t prepare \
  --config configs/s3t_densenet121_cifar100.json --name s3t_densenet121_cifar100_fixed20_v1
bash studies/2026-11-06_1721_fixed_20_shard_baselines/run.sh s3t prepare \
  --config configs/s3t_roberta_base_sst2.json --name s3t_roberta_base_sst2_fixed20_v1
```

SISA/ProtoSISA training or resume uses `baseline execute --run
studies/2026-11-06_1721_fixed_20_shard_baselines/runs/<name> --devices cuda:0
cuda:1`. S3T training uses `s3t run --config configs/<config>.json --name
<name> --devices cuda:0 cuda:1`; it auto-prepares an absent run. Start these
complete workloads in a persistent session or background process. Use a run's
`STOP` file to finish active trials and pause. `baseline summarize --run PATH`
and `s3t status --run PATH` read saved results without training. The baseline
reports include `search_results.csv`, `confirmation_results.csv`, and
`selected_by_budget.csv`; S3T writes corresponding search, confirmation, and
`selected_by_band.csv` reports. Do not use confirmation scores to revise the
candidate search or use final test scores to select settings.

The folder prefix records preparation on 2026-11-06 17:21 America/New_York.
No training in this study had completed when the folder was created. Later
reruns belong under `runs/`.
