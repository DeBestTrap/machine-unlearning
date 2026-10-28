# SISA RoBERTa / SST-2 experiments

Historical reports: [report-roberta.csv](reports/historical/report-roberta.csv). The [report index](reports/README.md) also explains
where new results appear. The CSV contents and modification times were preserved;
`reports/historical/provenance.json` records their original paths and hashes.

The historical report has 23 rows and 13 recorded hyperparameter combinations. These include settings outside the available LR sweep. The preserved launchers are the available SST-2 recipes, not proof of the exact commands that generated every row.

## Associated run scripts

```bash
bash studies/2026-01-09_1605_sisa_roberta_sst2/run.sh 2 1 --dry-run
bash studies/2026-01-09_1605_sisa_roberta_sst2/run_lrs.sh 2 0 --dry-run
bash studies/2026-01-09_1605_sisa_roberta_sst2/run_shards.sh 0 --dry-run
```

`run.sh` launches one shard count; `run_lrs.sh` sweeps 1e-3, 1e-4, and 1e-5 with 15 epochs and one slice; `run_shards.sh` sweeps 64, 128, and 256 shards.
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
January 9, 2026 at 16:05 America/New_York. It records historical grouping, not a
verified original source revision. Its legacy timing cannot be treated as the
new `update_v1` measurement.
