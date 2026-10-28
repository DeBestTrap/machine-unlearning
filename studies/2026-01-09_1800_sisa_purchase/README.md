# SISA Purchase experiments

Historical reports: [report-purchase.csv](reports/historical/report-purchase.csv). The [report index](reports/README.md) also explains
where new results appear. The CSV contents and modification times were preserved;
`reports/historical/provenance.json` records their original paths and hashes.

The recorded report has three rows: two initial evaluations and one deletion evaluation. Its recorded hyperparameters match the available single-run recipe, but the original seed and training-source revision are not recorded.

## Associated run scripts

```bash
bash studies/2026-01-09_1800_sisa_purchase/run.sh 2 1 --dry-run
bash studies/2026-01-09_1800_sisa_purchase/run_lrs.sh 2 1 --dry-run
```

`run.sh` launches one shard count; `run_lrs.sh` screens the original LR grid (1e-6, 3e-6, 1e-5).
All launchers accept `--help`, `--dry-run`, `--tag`, and `--device cpu|<GPU index>`.
Omit `--dry-run` to train. Without `--device`, they retain the current CUDA
visibility; the old RoBERTa launcher's fixed GPU selection is no longer forced.
Positional shard/request arguments work from any working directory.

Current recipes live in [configs/recipes.json](configs/recipes.json). The original
shell files are archived in `configs/original-launchers/`. Use these study launchers directly; the former experiment-specific
`example-scripts/` paths have been removed.
Reusable SISA orchestration remains in `example-scripts/common/`.

Each new launch creates `runs/<tag>/` with its source/settings snapshot, plan,
per-trial run records, and `reports/`. The study-level `reports/<tag>/` index links
to those generated CSVs. Historical reports are kept separate from fresh runs.
Use a new tag for changed code; model checkpoints remain separately stored.

The fixed folder prefix comes from the main report's saved modification time:
January 9, 2026 at 18:00 America/New_York. It records historical grouping, not a
verified original source revision. Its legacy timing cannot be treated as the
new `update_v1` measurement.

Inspect the historical report with `python studies/2026-01-09_1800_sisa_purchase/summarize.py`.
Pass `--csv <path>` to inspect a new report.
