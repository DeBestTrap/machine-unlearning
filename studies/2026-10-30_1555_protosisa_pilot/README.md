# ProtoSISA 10% CIFAR-10 pilot

The active pilot uses SAFE’s published mixture:
`lambda = exp(-mean_retained_shard_size / 100)` for independent shards, with
unscaled prototype cosine scores. The engine and shared preset use this rule
by default. Temperature scaling and its training/evaluation options have been
removed.

The [SAFE-weighting evaluation](runs/cifar10_10pct_v1/safe_weighting_v1/report.md)
recombines saved real-model outputs and obtains **92.06% official test accuracy
after deletion**, matching its SISA component. The resolved prototype weight is
0.00866035. This evaluates the mixture on the local independent full-finetuning
ProtoSISA proxy; SAFE’s InCA/shard-graph training is outside this implementation.

## Data and settings

The fixed training sample contains 4,750 examples, exactly 10% of the existing
47,500-example training pool, stratified to 475 examples per class with seed 42.
Saved `selection.json` files record original source IDs and logical training
indices. Sharding uses seed 1 and ten random 475-example shards. Task models
and prototypes use the same sample.

The existing 2,500-example validation holdout contains 1,000 scoring examples
and 1,500 confirmation examples, stratified independently with seed 42. Neither
pool enters training or deletion. The official 10,000-example test set was
originally evaluated after a frozen candidate’s deletion replay; the SAFE
comparison reuses those exact unscaled component outputs.

[configs/selected.json](configs/selected.json) records the current SAFE recipe:
three nominal epochs, one slice, AdamW at 0.0001, batch size 32, and seed 1.
This pilot does not establish full-training-pool, CIFAR-100, or SST-2 performance.

## Run

Use cached ViT weights and CIFAR-10 at the repository root’s
`cifar-10-batches-py/`. The original experiments used a local link to the main
checkout’s data cache. Use fresh group/trial names after changing source;
`capture_run` retains the exact source and settings.

```bash
python -u studies/2026-10-30_1555_protosisa_pilot/run.py \
  --group cifar10_safe_v1 --trial baseline_1ep --epochs 1 --device cuda:1
python -u studies/2026-10-30_1555_protosisa_pilot/run.py \
  --group cifar10_safe_v1 --trial three_epochs --epochs 3 --device cuda:1
python -u studies/2026-10-30_1555_protosisa_pilot/run.py \
  --group cifar10_safe_v1 --trial three_epochs --finalize --device cuda:1
python studies/2026-10-30_1555_protosisa_pilot/safe_weighting.py \
  --group cifar10_safe_v1 --output-name safe_weighting_v1
```

To recombine the existing real-model outputs with the current source, use a
fresh output name:

```bash
python studies/2026-10-30_1555_protosisa_pilot/safe_weighting.py \
  --output-name safe_weighting_v2
```

Training prints per-shard loss, accuracy, example counts, and epoch duration.
Finalization deletes the first fixed selected training ID, retrains the affected
shard, checks that 4,749 prototype contributors remain, and scores the holdout
and test pools. Current finalization records use `finalization_safe/`.
Model checkpoints remain under `saved_models/`, linked from the study records.

## Historical experiments

The original fixed and calibrated runs remain under `runs/cifar10_10pct_v1/`,
including their original source snapshots, scores, reports, and configuration.
The calibrated variant reached 93.24% test accuracy; it is historical and is not
the SAFE-weighted result. Its removed scaling behavior can be reproduced from
its archived launch source. `summarize.py` only reads and labels those historical
records; it does not provide score scaling or training controls.
