# Shared comparison records, version 1

The `reporting/` package is identical in machine-unlearning and
BlockUnlearn/unlearning. It uses only the standard library to read records and
build tables. Classification capture needs NumPy; figures need Matplotlib.
Adapters belong to their engines (`comparison_export.py` here and
`lib/comparison_export.py` in BlockUnlearn), keeping engine imports out of report
regeneration. ProtoSISA can write the same contract without changing the builder.

## Saved measurements versus generated reports

Each method execution/repeat owns three source files:

- `run.json`: `schema_name=unlearning_comparison`, `schema_version=1`, unique
  `run_id`, canonical method/dataset/model names, resolved config and its SHA256,
  seed, split/workload checksums, requested step count, precision, execution mode,
  hardware, provenance and `measurement_origin` (`native` or `historical_export`).
- `steps.jsonl`: one initial training record (`step=0`) and one final outcome per
  deletion (`step=1...N`). Fields are `run_id`, `step`, `operation`,
  `request_sha256`, `deleted_count`, `deleted_total`, `state_id`, `status`,
  `update_time_sec`, `training_time_sec`, `timing_scope`, `method_details` and
  `schema_version`. Request hashes use ordered original-source IDs when known.
- `evaluations.jsonl`: one record per `(state_id, split)`. Fields include
  `run_id`, `step`, `state_id`, `split`, `evaluation_sha256`, `identity_status`,
  `examples`, `evaluation_time_sec`, `metrics`, `confusion_matrix`, prediction
  artifact path/checksum, `status`, `method_details` and `schema_version`.

Statuses are `ready`, `requires_retraining`, `failed`, or `incomplete`. Steps are
contiguous from zero; cumulative counts must agree. Missing values are JSON null,
never zero or NaN. `method_details` contains engine-specific diagnostics. General
metric values are finite numbers; classification accuracy and F1 use fractions.

Classification capture also retains compressed `evaluation_data/*.npz` files
containing ordered labels and aggregate predictions, plus scores when available.
Confusion matrices have true classes in rows and predicted classes in columns.
Macro F1 averages classes present in labels or predictions; undefined divisions
are zero. Balanced accuracy averages recall over classes present in labels.
Score semantics (logits, vote counts, probabilities) stay explicit in details;
vote counts are not calibrated probabilities. Per-member caches remain optional.
All these files live with records, outside disposable model checkpoint storage.

`evaluation_sha256` binds dataset, split, example count, ordered labels, validation
split checksum and raw source fingerprint when known. `verified_source` means a
raw source fingerprint is present. `labels_and_protocol` explicitly lacks that
proof; historical scalar records can have `unverified` identity. Comparison notes
surface missing identity, caps/different pools, different models/hardware/scopes.
Preprocessing and training budgets remain in config; equal epoch counts do not
imply equal compute. Reports never substitute validation scores for test scores.

## Regenerate tables and plots

From either repository's Python root:

```bash
python -m reporting refresh PATH_TO_RUN_OR_RUNS OTHER_REPO_RUNS \
  --output PATH_TO_STUDY/runs/TAG/reports/comparison

python -m reporting refresh PATH_TO_RUN_OR_RUNS OTHER_REPO_RUNS \
  --output PATH_TO_STUDY/runs/TAG/reports/comparison \
  --metric macro_f1 --x p95_update_time_sec --y final_metric
```

Inputs are native comparison directories or parent directories containing them.
No engine, original dataset, or checkpoint is loaded. One command rebuilds:

- `steps.csv`: one row per run/operation, joining metadata, costs and the chosen
  split/metric of the operation's **final state**.
- `summary.csv`: one row per run with initial/final/change/mean post-delete metric,
  initial training cost, request coverage, mean/median/p95 and total deletion cost.
- SVG/PNG figures: metric trajectory, per-request update cost, and configurable
  summary scatter axes. Missing scores produce gaps.
- `report_settings.json`, `report_manifest.json` and README: exact settings,
  input paths/checksums, generated figure names and comparison notes.

`--split val`, `--no-plots`, `--strict`, and `--settings report_settings.json` are
available. Strict mode rejects comparison notes before generating output.
Mean/median/p95 use measured completed operations and expose `measured_steps`.
Full-workload totals require every requested operation and timing. Final metric
requires the requested final step; an earlier available score is not substituted.
The mean post-delete metric uses available scores with `scored_steps` coverage.
Initial training never enters deletion statistics. Retraining after exhaustion
belongs to the triggering request; intermediate exhausted evaluations are retained
as diagnostics but do not add table rows or duplicate request costs.

Changing saved settings or CLI options takes effect on the next refresh; no file
watcher runs in the background. Accuracy, macro/weighted F1 and balanced accuracy
can be rebuilt from confusion matrices. Other metrics require the corresponding
retained measurements. Unknown/unavailable metrics produce a clear error.

## Engine integration and history

SISA's common runner captures evaluations, exports the three records and builds
its comparison reports. Study execution also builds a combined report. S3T
evaluation retains aggregate predictions; replay/export writes comparison records,
and its CLI refreshes reports. Its operational journals include planning, state
writes, required selection work, retraining and publication, excluding reporting
evaluations/trace I/O. Resumed work accumulates; a lost hard-kill interval remains
unmeasured rather than being silently reconstructed.

BlockUnlearn captures full validation/test classification predictions initially
and after each complete update, outside update timing and with RNG state restored.
SU uses joint warmup; BM uses weight/delta averaging. Its standard run summary
exports each repeat under `comparison/repeat-NNN/` and rebuilds local reports.
Historical percentage/scalar scores retain `native_score` and native units; their
evaluation split is preserved. Unlabelled trace scores are never guessed to be
test accuracy. QA/regression require their own evaluation adapter.

Existing results are not automatically rewritten or assigned new timings. To
explicitly export available saved measurements without models:

```bash
# machine-unlearning
python comparison_export.py s3t PATH_TO_S3T_RUN
python comparison_export.py sisa PATH_TO_SISA_RECORDS

# BlockUnlearn/unlearning
python -m lib.comparison_export PATH_TO_EXPERIMENT_RUN
```

Older SISA reports need retained predictions/labels evaluated once to add new
classification metrics; the export command alone cannot reconstruct those from
an accuracy number. Legacy timing stays legacy. A fresh measurement is required
for a newly defined interval. Export source files are distinct from native engine
logs/checkpoints and do not change historical producer provenance.

## Adding ProtoSISA

Construct a manifest with `method=protosisa`, record step zero and every request
with `step_record`, use `classification_evaluation`/`record_evaluation`, and publish
with `write_bundle`. Use the same portable split and ordered deletion workload.
Keep prototype counts and adaptation details under `method_details`. The shared
validator and report tests define the contract; future incompatible changes need
a new schema version in both repositories.
