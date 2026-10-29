# Research studies

Each study owns its experiment driver, launchers, specific configuration,
documentation, and generated `runs/<run-id>/` records. Reusable SISA/S3T engines,
dataset loaders, shared split/deletion manifests, and general model presets stay
outside studies. This follows the organization used by BlockUnlearn.

| Study | Purpose | Entry point |
| --- | --- | --- |
| [SISA RoBERTa / SST-2](2026-01-09_1605_sisa_roberta_sst2/README.md) | Historical reports and available sharding/LR recipes | `2026-01-09_1605_sisa_roberta_sst2/run.sh` |
| [SISA ViT / CIFAR-10](2026-01-09_1617_sisa_vit_cifar10/README.md) | Historical reports and available sharding recipes | `2026-01-09_1617_sisa_vit_cifar10/run.sh` |
| [SISA Purchase](2026-01-09_1800_sisa_purchase/README.md) | Historical reports and available sharding/LR recipes | `2026-01-09_1800_sisa_purchase/run.sh` |
| [S3T validation](2026-10-26_0948_s3t_validation/README.md) | Full-pool implementation and replay verification | `2026-10-26_0948_s3t_validation/run.sh` |
| [S3T LR screening](2026-10-26_1332_s3t_lr_screening/README.md) | BMS/first-slice LR pilots on one ensemble member | `2026-10-26_1332_s3t_lr_screening/run.sh` |
| [Shared comparison](2026-10-26_2348_shared_comparison/README.md) | Compare methods on the same split and deletion workload | `2026-10-26_2348_shared_comparison/run.sh` |

Run shell entry points with `bash` from any working directory, using the training
environment. Use `--help` and the individual README before launching training.
The comparison and validation drivers support `--dry-run`; LR screening's
`status` command reads saved progress without launching workers.

Folders use fixed `YYYY-MM-DD_HHMM_<study>` prefixes in America/New_York.
Validation and LR screening use the latest historical completion evidence
documented in their READMEs. Shared comparison retains its matching BlockUnlearn
study folder name. New run timestamps belong in `runs/`, without renaming the
study. Stable imports are `studies.shared_comparison.run`,
`studies.s3t_lr_screening.run`, and `studies.s3t_validation.run`.

## Code changes and historical records

New launches record `study_run.json`: the exact plan/settings, source hashes,
Git commit and working-tree state, Python, and dependency versions. `launch_source/`
preserves the relevant code, model presets, and study configs, including
uncommitted source. Each engine also writes its normal resolved run configuration
and split/deletion records.

Resuming an LR study requires matching source and plan. Native SISA training
and S3T training/retraining/deletion replay also check the enclosing study record.
If source changed, use a fresh run or restore the captured implementation.
Dependency versions are recorded; source agreement alone does not guarantee an
identical environment or reproduce a historical GPU result.

Migrated records have `kind: historical_import` and unverified current-code
compatibility. Their existing engine snapshots and provenance are retained;
today's code is never presented as their original implementation. Saved metrics
and reports remain available. Migrated historical runs are excluded from training
and replay with the current code; use a fresh run for a new implementation.

SISA commands use the canonical study run directory directly through
`--container`. Its checkpoint cache still points to the same external
`saved_models/` store; new caches use study/run subdirectories. Existing S3T
run/checkpoint layouts are preserved. Deleting a study's entire `runs/` directory
deletes its records; model cleanup remains a separate operation.

Retired root `runs/`, container aliases, example launchers, and report shortcuts
have been removed. Use the study paths directly. See [the migration map](MIGRATION.md).

## Reports and their launchers

Every study has a `reports/README.md` index next to its `run.sh` / `run.py`.
Historical SISA CSVs reside under the owning study's `reports/historical/`;
associated original launch recipes are archived under `configs/original-launchers/`.
Current launchers create fresh `runs/<tag>/` records and publish links to their
CSVs/reports under `reports/<tag>/`, leaving the original generated files in place.
Report indexes and source are versioned; generated outputs and historical CSV
archives remain local artifacts.

New SISA/S3T executions export the [shared comparison contract](../reporting/README.md).
On successful completion the study also builds `runs/<tag>/reports/comparison/`:
per-step and per-run CSVs, plots, settings and source checksums. These reports
are published through `reports/<tag>/comparison/`. Change reporting settings and
refresh without training; source capture guards apply to training, not analysis.
