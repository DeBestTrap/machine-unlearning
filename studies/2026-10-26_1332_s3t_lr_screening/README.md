# S3T learning-rate screening

Saved results are collected in the [report index](reports/README.md), alongside
the study’s `run.sh` / `run.py` entry points.

This study screens BMS learning rates using a first-slice classifier head.
Each trial trains shard 0 / sequence 0 of the full five-shard partition. Selection
uses final-prefix validation accuracy; these pilot results are not full-ensemble
test results. General S3T training remains in the `s3t` package.

From any working directory, with paths relative to the repository here:

```bash
bash studies/2026-10-26_1332_s3t_lr_screening/run.sh --help
bash studies/2026-10-26_1332_s3t_lr_screening/run.sh run --device cuda:0 --datasets cifar10 sst2
bash studies/2026-10-26_1332_s3t_lr_screening/run.sh run --device cuda:0 --datasets cifar100
```

Omitting `--output` creates a fresh timestamped directory inside this study's
`runs/`. To choose a location or resume a matching new study run:

```bash
python -m studies.s3t_lr_screening.run run --device cuda:0 \
  --output studies/2026-10-26_1332_s3t_lr_screening/runs/my-screen
python -m studies.s3t_lr_screening.run status \
  --output studies/2026-10-26_1332_s3t_lr_screening/runs/my-screen
```

The queue writes a live `report.md`, `results.csv`, `status.json`, `queue.json`,
resolved `configs/`, trial model records, and worker logs. New searches preserve
their launch source/settings and reject resume after source or plan changes.

## Historical results

The original completed searches are retained under:

- `runs/s3t-lr-bms-first-slice-20261026/`: CIFAR-10 and SST-2.
- `runs/s3t-lr-bms-first-slice-cifar100-20261026/`: CIFAR-100.

These directories have historical-import markers and retain their original plans,
results and checkpoint/source snapshots at their canonical study paths.
Read their reports or use `status`; do not resume them with the current source.
Start a fresh study run to test updated code.

The fixed folder prefix comes from the later search's completion evidence:
its `status.json` records `2026-10-26T17:32:35.284652+00:00`, or 13:32 in
America/New_York. This is the historical study grouping time, not the migration
time or the date of every future run.
