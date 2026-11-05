# Cost-matched SISA / ProtoSISA development search

Search shard count, slice count, nominal epochs and LR on a fixed development
pool. The initial recipe is ResNet-50 on CIFAR-10. Both baselines share each
trained shard ensemble: SISA averages raw logits; ProtoSISA blends these with
frozen pretrained feature prototypes using SAFE's published weighting.
Prototypes are constructed exclusively from development fitting examples.

## Inputs and protocol

The recipe copies signed 10% manifests from BlockUnlearn, preserving its exact
3,000 fitting / 1,000 scoring / 1,000 confirmation source indices. The remaining
45,000 examples do not enter fitting or scoring. Only the original training
source is opened; official test data is never loaded. Loading verifies the raw
source fingerprint and manifest checksums/partitions.

Task models and the separate frozen encoder use ImageNet V2 ResNet-50. Task
heads use Xavier initialization. Task training freezes stem parameters and tunes
layer1 through layer4 and fc, matching SU/BM. Batch norm running statistics update
during task training, also matching that study. Input transforms match its 224px
ImageNet-normalized CIFAR pipeline. Optimizer: AdamW with its default weight decay;
batch size: 32; full precision. The recipe lives in `configs/resnet50_cifar10.json`.

- 100 ensemble configurations; search seed 1729; training seed 1.
- Shards `[2,3,4,5,6,8,10,12,16,20,24,32]`, slices `[1,2,3,4,5,8]`, epochs 1–10.
  Fractional schedules are excluded by `sisa_schedule.epochs_per_slice`.
- LR sampled uniformly in log space between `1e-5` and `1e-3`.
- Expected replay ceilings 0.10, 0.25, 0.50 against **one fixed full-data epoch**.
  Candidate epochs are charged in the numerator. Costs use the existing
  equal-shard/equal-slice model, excluding prototype maintenance, size remainders
  and removed examples.
- 34/33/33 candidates in disjoint cost strata `(0,.1]`, `(.1,.25]`, `(.25,.5]`.
  Valid structural tuples are shuffled per stratum and visited without repetition
  until exhausted, with a new independent LR per trial.
- Freeze top three candidates per method/budget, ties broken by lower cost, then
  lower LR, then candidate ID. Freshly train their union with seeds 1, 2, 3 on
  the same fitting data and evaluate only on confirmation. No adaptive feedback.
- Select highest mean confirmation accuracy per method/budget from those three
  frozen candidates; report sample SD and actual expected cost.

The 100 search configurations contain 1,880 shard-model fits and yield 200 method
scores. Confirmation adds at most 54 ensemble configurations. This is a proxy
search: rankings from 3,000 fitting examples need not transfer to 45,000.
ProtoSISA's published weighting depends on examples per shard, so its resolved
weight also changes with final training size. The formula remains fixed rather
than treating the development-resolved weight as a tuned constant.

## Prepare and launch

Prepare the plan and validate data without training:

```bash
python studies/2026-11-02_0213_baseline_search/run.py prepare \
  --config studies/2026-11-02_0213_baseline_search/configs/resnet50_cifar10.json \
  --name resnet50_cifar10_dev10_v1
```

Execute the complete search and confirmation:

```bash
python studies/2026-11-02_0213_baseline_search/run.py execute \
  --run studies/2026-11-02_0213_baseline_search/runs/resnet50_cifar10_dev10_v1 \
  --devices cuda:0 cuda:1
```

Absolute script paths work from any directory. Use persistent tmux or a detached
launcher. The coordinator checks GPU power limits are at most 250W and both
record/model disks have at least 20 GiB free. Concurrent GPU workloads are allowed;
durations are diagnostic, not paper timing measurements.

## Outputs, cleanup and resume

Generated files belong to `runs/<name>/`: signed plan, copied manifests, launch
source, frozen-feature caches, per-trial logs, shared-schema method reports,
memberships and predictions. Checkpoints temporarily use `saved_models/`.
Once a shard's predictions/checksum are durable, only its model/optimizer files
are removed. This bounds live checkpoint storage to one member per worker.
Predictions, memberships and timing journals remain.

`progress.json` identifies coordinator/worker PIDs and active logs.
`search_results.csv` stores screening scores; `finalists.json` freezes finalists;
`confirmation_results.csv` and `selected_by_budget.csv` contain confirmation and
chosen defaults. Accuracy is a fraction in `[0,1]`. Each trial retains
`predictions.npz` and reporting bundles for both methods. Its signed result is
published after both reports have been saved.

Create `runs/<name>/STOP` to finish active configurations and stop dispatching
new ones. Archive/remove STOP and repeat execute to resume. Completed trials
and durable shard-prediction markers are skipped. Interrupted members restart
from their last complete slice with optimizer state and deterministic RNG.
Power loss can mark timing incomplete; timing does not enter search selection.
Resume requires the captured launch source and exact saved plan/data identities.

Refresh tables without training:

```bash
python studies/2026-11-02_0213_baseline_search/run.py summarize \
  --run studies/2026-11-02_0213_baseline_search/runs/resnet50_cifar10_dev10_v1
```

## ViT-B/16 on CIFAR-100

The reusable `configs/vit_b16_cifar100.json` recipe uses the same search counts,
cost ceilings, whole-epoch scheduler and 60/20/20 development protocol. Its
`model_args.layers_to_finetune` explicitly selects the last four encoder layers,
final LayerNorm and classifier head, matching the completed SU/BM ViT study.
`model_args.task_head_initialization=xavier_uniform` matches its classifier
initialization. The study driver applies these options to both SISA and ProtoSISA
task models; the prototype encoder remains the separate frozen pretrained ViT.
Recipes omitting these options keep their historical model-construction behavior.

Prepare with this config and a new name such as `vit_b16_cifar100_cost_dev10_v1`,
then use the same execute command with that run's absolute path. Original-source
manifests are shared exactly with the SU/BM CIFAR-100 development searches.
Only 3,000 development fitting examples train models; 1,000 scoring examples guide
selection and the frozen finalists are confirmed on the remaining 1,000 examples.
