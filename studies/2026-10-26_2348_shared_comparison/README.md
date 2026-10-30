# Shared 25-deletion comparison

## ViT reporting preview

To run only CIFAR-10 across SISA, SU and BM sequentially on one GPU:

```bash
bash studies/2026-10-26_2348_shared_comparison/run_preview.sh \
  --datasets cifar10 --devices 0 --tag vit_preview_c10_new --dry-run
bash studies/2026-10-26_2348_shared_comparison/run_preview.sh \
  --datasets cifar10 --devices 0 --tag vit_preview_c10_new
```

Use a fresh tag for each execution. The default still runs both datasets on
separate GPUs. `--devices` assigns one logical GPU per selected dataset in order;
an inherited scheduler `CUDA_VISIBLE_DEVICES` mask is preserved for child jobs.
The BlockUnlearn study contains `run_vit_preview.sbatch` and Alpha submission
instructions for this single-dataset run. Both repositories must be deployed.

`run_preview.sh` launches SISA, SU and BM on CIFAR-10 and CIFAR-100 with one
epoch, one training seed, the full 47,500-example training pools, and the first
three single-example deletions of the verified workload. SISA uses five shards
and two slices; SU/BM retain their selected allocations. CIFAR-10 and CIFAR-100
run on separate GPUs, with methods processed serially within each dataset.
Settings are in [configs/vit_preview.json](configs/vit_preview.json).

```bash
bash studies/2026-10-26_2348_shared_comparison/run_preview.sh \
  --devices 0 1 --tag vit_preview_20261029 --dry-run
bash studies/2026-10-26_2348_shared_comparison/run_preview.sh \
  --devices 0 1 --tag vit_preview_20261029
```

The launcher requires CUDA and writes native SU/BM experiments in the sibling
BlockUnlearn checkout, retaining its configured checkpoint store. SISA records,
per-method logs, source snapshots, launch plans and combined reports live in this
study's `runs/<tag>/`. BlockUnlearn's corresponding study references the same
native records. Every successful method refreshes both accuracy and macro-F1
tables/figures. Start with `runs/<tag>/report.md` or
`reports/<tag>/comparison/summary.csv`. This is a functional preview with a small
deletion workload, not a convergence benchmark. Use a fresh tag for another run.

## Shared reports and plots

New runs automatically export the [shared comparison records](../../reporting/README.md)
and rebuild `runs/<tag>/reports/comparison/`, linked from `reports/<tag>/comparison/`.
The three saved files are `run.json`, `steps.jsonl`, and `evaluations.jsonl` per
method run. Generated `steps.csv` joins costs with final-state scores;
`summary.csv` summarizes each run and excludes initial training from deletion
statistics. Compressed classification predictions/labels and confusion matrices
stay with the records when checkpoints are removed.

Defaults live in [configs/report.json](configs/report.json). To regenerate reports
and figures with another metric/axis, without loading models or datasets:

```bash
python -m studies.shared_comparison.report --tag comparison_v2
python -m studies.shared_comparison.report --tag comparison_v2 \
  --metric macro_f1 --x p95_update_time_sec --y final_metric
```

Add BlockUnlearn runs exported with the same schema:

```bash
python -m studies.shared_comparison.report --tag comparison_v2 \
  --include ../BlockUnlearn/unlearning/experiments/MODEL/EXPERIMENT/RUN
```

Additional input paths are saved for subsequent refreshes. Use `--split val`,
`--no-plots` or `--strict` as needed. The report manifest and README identify
different evaluation pools, hardware, models and timing scopes. Existing
`comparison_v1` results retain their historical evidence; no new timer or F1 is
invented from an old accuracy scalar. Explicit legacy export commands are in the
shared reporting documentation.

Saved results are collected in the [report index](reports/README.md), alongside
the study’s `run.sh` / `run.py` entry points.

`configs/quick25.json` selects one configuration per method and dataset. The same file,
split manifests, and deletion workloads live in both repositories. The launcher
checks their recorded checksums before launching any commands. No dataset or
model download is needed for `--dry-run`.

Each workload contains the **first 25 single-example requests** from the existing
seed-123, 100-request workload, with a new checksum. All methods use the complete
shared training pool, the same validation holdout, and the same original-source
deletion IDs in the same order. Shards, training schedules, and model randomness
remain method-specific; this preset does not make their training costs equal.

| Dataset | FC-M, lambda=0.5 | FC-wide, input to output | Train / validation |
| --- | --- | --- | --- |
| CIFAR-10 | `1 <- 1 <- 50` | `19 <- 1 <- 2 <- 3` | 47,500 / 2,500 |
| CIFAR-100 | `1 <- 20 <- 1` | `18 <- 1 <- 1 <- 5` | 47,500 / 2,500 |
| SST-2 | `1 <- 1 <- 50` | `17 <- 1 <- 1 <- 6` | 63,982 / 3,367 |

The manifest's single `epochs: 1` setting controls every comparison run:
FC/FC-M receive `train_args.max_epochs=1`, SISA receives its nominal `--epochs 1`
budget, and S3T receives `epochs_per_slice=1`, overriding the longer schedules in
its base presets. These values also apply during deletion retraining.

Epoch counts have different scopes: BlockUnlearn trains each group for one
epoch, SISA distributes its nominal budget across cumulative slices using its
native integer schedule, and S3T trains each cumulative prefix for one epoch per
sequence. In particular, SISA can assign zero epochs to some slices at this low
budget. This matches the configured epoch count, not total example visits or
compute across methods.

SISA uses five shards and four slices, with batch size and learning rate recorded
per dataset. S3T retains five shards, six slices, and six sequences from its base
presets. It remains a full ensemble run; the launcher does not reuse LR pilots.

## BlockUnlearn

Run from `BlockUnlearn/unlearning/` in its Python environment:

```bash
python studies/2026-10-26_2348_shared_comparison/run.py --device 0 --tag comparison_v1 --dry-run
python studies/2026-10-26_2348_shared_comparison/run.py --device 0 --tag comparison_v1
```

This trains six new models: FC-M and FC for each of the three datasets, then
performs 25 deletions for each. Experiment names start with
`Shared25_comparison_v1_`; the original launch scripts and their selections remain
available. Those CIFAR/SST-2 selected configs now explicitly bind the shared
split and 25-request workload too. A request manifest determines the actual
deletion count, overriding `num_sequential`.

## machine-unlearning

Run from `machine-unlearning/` in its Python environment:

```bash
python studies/2026-10-26_2348_shared_comparison/run.py --device 0 --tag comparison_v1 --dry-run
python studies/2026-10-26_2348_shared_comparison/run.py --device 0 --tag comparison_v1
```

This runs SISA and S3T for the same three datasets. SISA uses
`DELETION_REQUEST_MANIFEST` with the existing common runner, retaining labels
0 through 25. S3T trains, replays the same workload, and audits the result.

S3T's default `--s3t-on-exhaustion continue` records all requests, marking scores
unavailable with `requires_retraining` when no sequence survives. To obtain a
predictor after exhaustion, add `--s3t-on-exhaustion retrain`; the extra training
cost and generations are recorded by S3T. `stop` is also available to end replay
at exhaustion. Unavailable scores should remain unavailable in comparisons.

## Smaller reruns and records

```bash
# BlockUnlearn: one dataset and one method
python studies/2026-10-26_2348_shared_comparison/run.py --datasets cifar100 --methods fcm --device 0 --tag fcm_c100_v1

# machine-unlearning: SISA only
python studies/2026-10-26_2348_shared_comparison/run.py --datasets cifar100 --methods sisa --device 0 --tag sisa_c100_v1
```

Omit `--tag` to generate a fresh UTC timestamp. Use a new tag for each comparison;
the launcher refuses to overwrite a launched plan or an existing baseline run.
It records the exact protocol and commands in
`studies/2026-10-26_2348_shared_comparison/runs/<tag>/plan.json` and subprocess exit codes in
`execution.jsonl`. BlockUnlearn keeps normal metrics under `experiments/`, SISA
under `studies/2026-10-26_2348_shared_comparison/runs/<tag>/sisa-<dataset>/`, and S3T under
`studies/2026-10-26_2348_shared_comparison/runs/<tag>/s3t-<dataset>/`. Native SISA commands use that record directory directly; combined SISA CSVs live
in the same study run’s `reports/` directory. Commands run serially on the selected
GPU and stop on failure. To resume interrupted training, use the underlying
method's saved run and native resume commands; this launcher starts fresh runs.

Changes to the shared manifest or workload files should be copied to both repos.
The existing split/deletion validators check source identity when data is loaded;
the launcher's dry run validates the saved protocol without loading raw data.

SQuAD's selected configs explicitly share its train/validation split, but SQuAD
is excluded from this comparison because shared deletion replay does not support
its overlapping token windows, and the local baselines do not support that task.
Proto-SISA is not installed in either repo. To add it later, use these exact split
indices and ordered source IDs through a compatible adapter.

## Launch source and historical results

Every new launch saves `study_run.json` and `launch_source/` with the protocol,
exact commands, source hashes, Git state, Python, and dependency versions. The
engines also retain their resolved configs and request/split records. Use a new
tag after implementation changes; native training/replay checks launch source
before mutating a study run.

`runs/comparison_v1/` contains the imported October 26–27 comparison records.
Its historical-import marker preserves the available original provenance without
asserting current-code compatibility. Keep its results as historical evidence;
use a fresh run for new code. Retired root output/report/container shortcuts have been removed. The study folder name matches BlockUnlearn’s fixed
`2026-10-26_2348_shared_comparison` identifier; local `execution.jsonl` records
completion at 03:27 America/New_York on October 27.
