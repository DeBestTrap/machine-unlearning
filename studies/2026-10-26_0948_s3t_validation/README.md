# S3T validation study

Saved results are collected in the [report index](reports/README.md), alongside
the study’s `run.sh` / `run.py` entry points.

This study groups the original implementation checks and provides a fresh-run
launcher for the full-pool validation protocol:

```bash
bash studies/2026-10-26_0948_s3t_validation/run.sh --dry-run --tag validation_v2
bash studies/2026-10-26_0948_s3t_validation/run.sh --device cuda:0 --tag validation_v2
```

The launcher trains the two-shard/two-slice/two-sequence CIFAR-10 and SST-2
checks, replays all 100 shared deletions with `continue`, and audits the saved
state. `--datasets cifar10` selects one dataset. Each fresh run retains its launch
source/settings, engine configs, and outputs under this study's `runs/<tag>/`.
Historical snapshots are preserved in `runs/validation_20261026/`; their
current-code compatibility is unverified, so use a new tag for new training.

The fixed completion prefix uses the original `unit-tests.log` modification
time, October 26, 2026 at 09:48 America/New_York. It does not date the migration
or every future run. The original validation report follows, with artifact links
updated to their canonical study locations; its reported results are unchanged.

# S3T validation — 2026-10-26

The implementation passed **36 tests** with:

```bash
python -m unittest discover -s tests
```

This includes 23 S3T tests and the 13 existing split/deletion tests. Compilation
and `git diff --check` also passed. The test log is saved at
`studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/unit-tests.log`.

Tested environment: PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118,
Transformers 5.8.0, and NVIDIA RTX 4090 GPUs. No other repository's implementation
was changed to run these checks.

## Full-pool verification runs

These short runs used two shards, two slices, two sequences per shard, six
transformer blocks per slice, and one epoch per cumulative stage. They used the
complete shared training and validation pools and the full final evaluation pools.
Their purpose was to exercise the complete implementation, not to reproduce or
tune the paper's reported results.

| Run | Before deletion | After first deletion | First exhausted request | Training epochs, summed wall seconds |
| --- | ---: | ---: | ---: | ---: |
| torchvision ViT-B/16, CIFAR-10 | 96.36% | 96.16% | 4 | 338.36 |
| RoBERTa-base, SST-2 | 89.11% | 88.30% | 6 | 294.76 |

CIFAR-10 uses its 10,000 official test examples. SST-2 final evaluation uses its
872 labeled development examples; the separate shared validation holdout is used
during training. Training times sum synchronized epoch wall times and exclude
initialization, validation, and checkpoint saving.

Both runs replayed **all 100 saved single-example deletion requests** using
`--on-exhaustion continue`. After exhaustion, accuracy is null and the status is
`requires_retraining`. No predictor was substituted. The structural audits each
checked 400 prefix states against the original source IDs and passed.

Artifacts:

- `studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/vit-cifar10-full-pool/results.csv`
- `studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/roberta-sst2-full-pool/results.csv`
- Each directory's `deletion_audit.json`, `deletion_trace.json`, `config.json`,
  `metadata.json`, source snapshot, backbone, and sequence checkpoints.

The commands used for training were equivalent to:

```bash
python -m s3t train --config configs/s3t/vit-cifar10.json \
  --output studies/2026-10-26_0948_s3t_validation/runs/manual-c10-verification --device cuda:1 \
  --set shards=2 --set slices=2 --set budget=2 \
  --set layers_per_slice=6 --set epochs_per_slice=1

python -m s3t train --config configs/s3t/roberta-sst2.json \
  --output studies/2026-10-26_0948_s3t_validation/runs/manual-sst2-verification --device cuda:0 \
  --set shards=2 --set slices=2 --set budget=2 \
  --set layers_per_slice=6 --set epochs_per_slice=1
```

The corresponding workloads were `deletions/cifar10-seed123-100x1.json` and
`deletions/glue-sst2-seed123-100x1.json`. Use `python -m s3t replay` with the matching
`--run`, `--requests`, and `--on-exhaustion continue` arguments to repeat this
protocol. New experiments should use new output directories.

## Additional checks

- Pretrained-model training and manual deletion completed for torchvision ViT
  on CIFAR-10 and CIFAR-100, HF ImageNet-21K ViT on CIFAR-10, and RoBERTa-base on
  SST-2. These were capped diagnostic runs, with caps recorded in their configs.
- Across those four runs and the two full-pool runs, all **36 trained prefixes**
  were reloaded from final checkpoints. Their reconstructed validation
  accuracies exactly matched the accuracies recorded before later-stage training.
  The detailed results are in `studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/verification.json`.
- A pretrained ViT run with real CIFAR augmentation, two DataLoader workers,
  and CUDA was interrupted after the first epoch of stage two and resumed.
  All **52 adapter/head tensors** matched uninterrupted training bit for bit.
  See `studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/cuda-resume-verification.json`.
- The unit tests additionally compare earlier-prefix **logits** bit for bit
  after later training/deactivation, verify frozen weights under AdamW, exercise
  all epoch-boundary recovery positions, and check member scheduling order.
- Synthetic tests cover explicit and automatic retained-data retraining,
  incomplete-generation recovery, conflicting process locks, workload/cap
  rejection, and corrupted dependency traces. CUDA RoBERTa and bfloat16 smoke
  runs also completed.

The RoBERTa-large preset has not undergone a full pretrained training run here.
Its 7×3 layer allocation was checked using a small 24-block RoBERTa fixture.
Larger budgets, the supplied longer training presets, and nonuniform-prior
sequence selection still need task-specific experiment evaluation. The latter's
algorithmic validity and budget constraints are covered by tests.

See [implementation decisions](s3t-decisions.md) for the classifier-head policy,
LoRA scaling, optimization schedule, selection rules, and measurement definitions.
