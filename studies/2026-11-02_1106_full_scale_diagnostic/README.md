# Full-scale ResNet baseline diagnostic

This study owns the SISA/ProtoSISA implementation worker and native records for
the joint, frozen full-training diagnostic coordinated by
`BlockUnlearn/unlearning/studies/2026-11-02_1106_full_scale_diagnostic/`.

`baseline_worker.py` is invoked by that coordinator for initial training and
then serialized per-shard replay. It trains on the verified 45,000-example
final-training pool and scores the same 1,000 held-out confirmation examples as
SU/BM. It never loads an official test dataset. Selected settings are frozen
before execution; full-training results must not guide configuration selection.

The two baselines share training when their shard/slice/epoch/LR settings match.
SISA scores average raw member logits. ProtoSISA uses the published SAFE weighting
and prototypes from the full retained training pool. Each original shard receives
one single-example deletion, in ascending shard order. Prototype repair is timed
separately and added to the member-repair time for ProtoSISA.

Generated records live in `runs/<run-name>/<job>/`, with model/optimizer caches
in the existing `saved_models/` store. Committed request journals retain exact
source IDs, timings and member paths. After scoring the full sweep, checkpoints
are removed while predictions, shard memberships, reporting bundles and timing
journals remain. Final summary/control records and frozen inputs are referenced
from the coordinating BlockUnlearn study rather than duplicated here.

See the coordinator's README for launch, progress, outputs and resume commands.
