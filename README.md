# Machine Unlearning with SISA
### Lucas Bourtoule, Varun Chandrasekaran, Christopher Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, Nicolas Papernot
This repository contains the core code used in the SISA experiments of our [Machine Unlearning](https://arxiv.org/abs/1912.03817) paper along with some example scripts.

An independent S3T implementation is also available. See [S3T commands and experiment setup](S3T.md)
and [implementation decisions](docs/s3t-decisions.md). It supports ViT on CIFAR-10/CIFAR-100,
RoBERTa on SST-2, and the shared [data splits](SPLITS.md) and [deletion workloads](DELETIONS.md).

Start with the [SISA Purchase study](studies/2026-01-09_1800_sisa_purchase/README.md) or the [study index](studies/README.md). Each study keeps its reports beside the associated launch scripts.

``sisa.py`` is the script that trains a given shard. It should be run as many times as the number of shards.

## SISA checkpoints and retained run data

SISA study launchers pass their canonical run record directory directly, such as
`studies/2026-10-26_2348_shared_comparison/runs/<tag>/sisa-cifar10/`. Native
commands accept that directory through `--container`; no root container links
are created.

Each SISA record directory contains `config.json`, `report.csv`, timings,
predictions, split/deletion records, and `checkpoint_storage.json`. Its `cache/`
link points to the separate `saved_models/` store. On this machine,
`saved_models/` points to `/mnt/csdrive/containers/`. Existing caches retain their
external targets; new study caches use matching study/run subdirectories there.
Reports are collected by each study's `reports/README.md` index.

To delete checkpoints while keeping run records:

```bash
bash clear_containers.sh studies/2026-10-26_2348_shared_comparison/runs/comparison_v1/sisa-cifar10
bash clear_containers.sh  # all SISA run checkpoints
```

Cleanup saves the checkpoint-size measurement before removing model files.
Predictions and reports remain available; generating new predictions or resuming
training requires checkpoints. See [timing boundaries](TIMING.md) and the
[study index](studies/README.md).

## Citing this work

If you use this repository for academic research, you are highly encouraged
(though not required) to cite our paper:

```
@inproceedings{bourtoule2021machine,
  title={Machine Unlearning},
  author={Lucas Bourtoule and Varun Chandrasekaran and Christopher Choquette-Choo and Hengrui Jia and Adelin Travers and Baiwu Zhang and David Lie and Nicolas Papernot},
  booktitle={Proceedings of the 42nd IEEE Symposium on Security and Privacy},
  year={2021}
}
```

## Shared comparison preset

For one SISA/S3T configuration per CIFAR-10, CIFAR-100, and SST-2 dataset with
the same training pools and 25 saved deletions as BlockUnlearn, see
[the shared comparison study](studies/2026-10-26_2348_shared_comparison/README.md). Preview without training:

```bash
bash studies/2026-10-26_2348_shared_comparison/run.sh --device 0 --tag comparison_v2 --dry-run
```

## Research studies

Specific comparisons, LR searches, and validation experiments live in
[studies/](studies/README.md), each with its own driver, settings, documentation,
and local `runs/` outputs. Shared training implementations and dataset manifests
remain reusable. New study runs capture their launch source and environment;
resuming training/replay requires matching source. Historical results retain
original provenance and are marked separately from current-code runs.

See [the migration map](studies/MIGRATION.md) for canonical locations. Old entry
points and directory aliases have been removed. Checkpoint cleanup remains separate from run-record cleanup.
