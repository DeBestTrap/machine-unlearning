# Study migration

Experiment source and completed records live at canonical study locations.
The old source entry points, report shortcuts, root run paths, and container links
have been removed. The tables document where the files moved; they do not imply
that the old locations still work. Shared engine code and global manifests retain
their existing locations.

## Source entry points

| Previous location | Canonical location |
| --- | --- |
| `run_shared_comparison.py` | `studies/2026-10-26_2348_shared_comparison/run.py` |
| `comparison/quick25.json` | `studies/2026-10-26_2348_shared_comparison/configs/quick25.json` |
| `comparison/README.md` | `studies/2026-10-26_2348_shared_comparison/README.md` |
| `s3t/lr_search.py` | `studies/2026-10-26_1332_s3t_lr_screening/run.py` |
| `docs/s3t-validation.md` | `studies/2026-10-26_0948_s3t_validation/README.md` |

## Generated records

| Previous location | Canonical location |
| --- | --- |
| `runs/shared_comparison/` | `studies/2026-10-26_2348_shared_comparison/runs/` |
| `containers/comparison_v1_sisa_<dataset>/` | `studies/2026-10-26_2348_shared_comparison/runs/comparison_v1/sisa-<dataset>/` |
| `report-shared25-comparison_v1-<dataset>.csv` | `studies/2026-10-26_2348_shared_comparison/runs/comparison_v1/reports/sisa-<dataset>.csv` |
| `runs/s3t-lr-bms-first-slice-20261026/` | `studies/2026-10-26_1332_s3t_lr_screening/runs/s3t-lr-bms-first-slice-20261026/` |
| `runs/s3t-lr-bms-first-slice-cifar100-20261026/` | `studies/2026-10-26_1332_s3t_lr_screening/runs/s3t-lr-bms-first-slice-cifar100-20261026/` |
| `runs/s3t-validation/` | `studies/2026-10-26_0948_s3t_validation/runs/validation_20261026/` |

`python studies/migrate.py` previews local moves; add `--apply` to perform them.
The migration checks existing process locks, preserves regular-file inodes,
sizes, modification times, and text-record hashes, and rebases SISA's operational cache links
to the same external targets. It renames data without copying large checkpoints.
An ignored inventory under `studies/.migration/` records each verification.

Original record files, embedded paths, model hashes, and old source snapshots are
not rewritten. New `study_run.json` sidecars describe historical provenance as
unverified rather than attaching the migration-time implementation. Repeating
migration preserves canonical records and sidecars without recreating old paths.

Legacy SISA reports and launch scripts now have study ownership as shown below.
Original CSV contents remain unchanged. Where a report's exact producing script
or configuration is unknown, its study README says so explicitly.

## Legacy SISA reports and associated scripts

| Previous location | Canonical study location |
| --- | --- |
| `report-purchase.csv` | `studies/2026-01-09_1800_sisa_purchase/reports/historical/report-purchase.csv` |
| `report-vit.csv`, `r-vit.csv` | `studies/2026-01-09_1617_sisa_vit_cifar10/reports/historical/` |
| `report-roberta.csv` | `studies/2026-01-09_1605_sisa_roberta_sst2/reports/historical/report-roberta.csv` |
| `example-scripts/purchase-sharding/run*.sh` | `studies/2026-01-09_1800_sisa_purchase/run*.sh` |
| `example-scripts/vit-cifar10/run*.sh` | `studies/2026-01-09_1617_sisa_vit_cifar10/run*.sh` |
| `example-scripts/roberta-sst2/run*.sh` | `studies/2026-01-09_1605_sisa_roberta_sst2/run*.sh` |
| `filter_report_purchase.py` | `studies/2026-01-09_1800_sisa_purchase/summarize.py` |

Old source entry points and report paths are removed. Original shell script contents are archived
in each study's `configs/original-launchers/`; current launchers preserve their
available recipe grids while using fresh study run IDs, local report paths, and
explicit/inherited GPU selection. Shared orchestration stays in
`example-scripts/common/`. Historical CSV hashes, timestamps, row counts, and
metadata limits are recorded beside them in `provenance.json`.

All studies expose a `reports/README.md` index. Links under each study's
`reports/<run-name>/` point into its saved run outputs; refreshing an index does
not recalculate or overwrite historical metrics.

## Canonical paths only

SISA study runners pass `studies/<study>/runs/<tag>/<trial>/` directly to native
commands through `--container`; no separate root container aliases are needed.
Model cleanup discovers SISA records inside studies and retains their reports.
The `saved_models/` drive link, per-run checkpoint caches, and study report-index
links serve the current layout and remain in place.

Historical `plan.json`, source snapshots, and provenance files retain their
original recorded path strings as evidence; they are not rewritten to pretend
they were generated under the new layout.
