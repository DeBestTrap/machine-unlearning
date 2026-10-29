# Comparable SISA and BlockUnlearn update timing

New measurements use `timing_scope=update_v1` and seconds throughout. Update time
starts with a resident model and a supplied deletion request and ends when the
updated state is ready for use. It includes identifying affected shards/blocks,
reading request/split metadata, restoring weights, preparing replay state, loading
training batches, device transfers, all required training, and checkpoint writes
needed for subsequent deletions. Initial model construction, Python process startup,
and preparing the deletion workload are setup costs outside this interval.

Evaluations used only to report accuracy are excluded. Validation or replacement
scores that the algorithm needs to decide whether/how to retrain are included.
Per-step timing/trace publication is excluded; incidental logging inside a training
call remains part of that call's elapsed time. CUDA is synchronized at operation
and epoch boundaries, without adding synchronization to every minibatch.

## SISA

`sisa.py --train` saves local `studies/<study>/runs/<run>/<trial>/times/shard-<i>:<label>.json`
records containing `update_time_sec`, `training_time_sec`, and a breakdown of the
slices actually trained or reused. Training time covers epoch loops, including
batch loading and transfers; update time additionally covers planning, checkpoint
restoration, and checkpoint writes. A cached slice contributes no historical
training time from the request that originally created it. Its lookup overhead
still belongs to the current update.

The common runner processes shards serially. Its CSV report sums **all affected
slices across all shards**, so `update_time_sec` is serial operation time, excluding
interpreter/model setup between shard invocations. It is not parallel ensemble
latency. `retraining_time` remains a compatibility alias for the same update total;
`training_time_sec` is the secondary sum of epoch-loop times. For example, slices
with 2, 3, and 4 seconds of training contribute 9 seconds before checkpoint and
other update work, rather than only the final slice's 4 seconds.

The initial model is label 0. Subsequent labels measure the work for that supplied
deletion state. Rerunning a completed label preserves its saved measurement.
Handled interruptions accumulate work across invocations; recovery starts after
the last completed checkpoint epoch. A hard kill can lose an unrecorded timing
interval, so that measurement is marked incomplete and cannot be exported as a
complete update. Use a new container to obtain a fresh measurement in that case.

`python time.py --container studies/<study>/runs/<run>/<trial> --label 1` prints the sum and shard mean of these
complete update timings. Predictions, timing JSON, and reports remain local after
model cleanup, allowing reports to be regenerated without checkpoints.

## BlockUnlearn

New trace events carry `update_time_sec` with the same `update_v1` scope. The step
wall interval includes replay preparation and other update work. Its diagnostic
evaluation and trace/metrics-publication intervals are subtracted. Evaluations
needed for a configured replacement threshold are retained. Run summaries export
`mean_update_time_sec`, `total_update_time_sec`, and `timing_scope`; comparison
tables prefer the explicitly scoped mean.

## Historical results

Legacy SISA `retraining_time` measured batch compute in the last slice per shard,
without an explicit CUDA synchronization boundary. It cannot reconstruct complete
update time after the fact. Appending a new report preserves old rows and marks
them `legacy_batch_last_slice`, leaving their new update-time columns empty.
Reports require every expected shard's complete timing record and never substitute
a missing shard with zero. The comparison parser groups timing scopes separately.

BlockUnlearn's older replacement/retraining component totals remain legacy
measurements. Historical traces are not automatically relabelled `update_v1`.
Record fresh runs under the same split/workload and hardware for comparisons using
the new definition.

## S3T and shared reports

New S3T runs record `comparison_timing/` journals with `update_v1` scope. Deletion
work includes request validation, state/partition reads, selection and authoritative
state writes. If exhaustion triggers retraining, member training and generation
preparation/publication are charged once to that request. Validation needed for
`selection=validation` is included; diagnostic validation for `longest_prefix` and
reporting evaluation/trace publication are excluded. The controller journal excludes
member intervals, which have their own journals, preventing double counting.
Handled interruptions accumulate; hard-kill intervals make a timing unavailable.

Shared records explicitly identify serial execution. SISA sums serial shard
operations; S3T sums serial member/controller operations. These measurements are
not parallel request latency. Generation/member costs are not repeatedly charged
to every evaluation row. Historical S3T planning-only time stays a diagnostic,
not a substitute for complete update time. See [the report contract](reporting/README.md).
