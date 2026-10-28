# SISA ViT / CIFAR-10 experiments

Historical reports: [report-vit.csv](reports/historical/report-vit.csv), [r-vit.csv](reports/historical/r-vit.csv). The [report index](reports/README.md) also explains
where new results appear. The CSV contents and modification times were preserved;
`reports/historical/provenance.json` records their original paths and hashes.

The main report contains 22 rows covering multiple shard counts, slice counts, and epoch budgets. The available default launcher uses four slices and two nominal epochs; it does not reproduce every recorded row. `r-vit.csv` is an earlier 11-row report without hyperparameters or independently verified dataset identity. It is retained here as a legacy ViT report, without inventing its missing configuration.

## Associated run scripts

```bash
bash studies/2026-01-09_1617_sisa_vit_cifar10/run.sh 2 1 --dry-run
bash studies/2026-01-09_1617_sisa_vit_cifar10/run_shards.sh 1 --dry-run
```

`run.sh` launches one shard count; `run_shards.sh` sweeps the original counts 1, 4, 16, 32, and 64.
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
January 9, 2026 at 16:17 America/New_York. It records historical grouping, not a
verified original source revision. Its legacy timing cannot be treated as the
new `update_v1` measurement.
