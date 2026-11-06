# S3T development allocation search

This study prepares and runs separate S3T searches for ViT-B/16/CIFAR-100 and
ResNet-50/CIFAR-10. It uses the same **fixed 10% original training-source
development pools** as the current BlockUnlearn studies. The exact seed-42
source-index parent and 60/20/20 development manifests are copied under
`configs/manifests/` so a clone on another server does not depend on a second
checkout. Each candidate trains on 3,000 examples and scores on 1,000; its
frozen finalists train again on the same 3,000 and score on the other 1,000.
The final 45,000 training examples are outside this search. The loader creates
only the original CIFAR training dataset; an S3T development run has no official
test dataset. It does not issue deletion requests during configuration search.

The search varies shard count, slice count, sequence budget, complete epochs per
slice, and learning rate. Every trial fits and scores the **complete S3T
ensemble**, using logits aggregation and the longest retained prefix. There are
36 deterministic, balanced screening configurations per model (12 in each of
three shard-count bands), followed by the top two per band, frozen before
confirmation and repeated at seeds 1/2/3: 18 confirmation fits per model. The
per-trial limits of at most 32 members and 30 one-epoch-equivalent training
passes bound the search. `search_results.csv` and `selected_by_band.csv` retain
actual training example presentations, model count, and accuracy. S3T's
sequence switching has no direct equivalent to the SU/BM expected-retraining
cost ceiling; the search therefore reports its training work and shard count
without imposing those ceilings. Full-data deletion and wall-clock comparisons
remain separate experiments.

The ViT implementation uses the existing S3T query/value LoRA adapters in all
12 encoder blocks, grouped evenly from output to input. The **ResNet-50 path is
an explicit S3T adaptation**: bottleneck `conv2` and `conv3` receive independent
low-rank convolution branches; its four residual stages form the slices and
the task head follows the same first-slice policy. The ImageNet V2 backbone
and Xavier task head match the existing ResNet comparison. Batch-normalization
running statistics stay fixed, including at each resumed epoch, so saved
prefixes can be reconstructed from the frozen backbone plus adapter/head state.

## Commands that train

From the repository root, run **one command per model**. Each command prepares
its immutable plan if absent, then launches or resumes training on the listed
device. The CIFAR training archive is downloaded by torchvision if absent; use
`--data-root /path/to/data` if it should live elsewhere. Install this repo's
S3T dependencies on the server first.

```bash
bash studies/2026-11-05_2252_s3t_allocation_search/run.sh run \
  --config configs/vit_cifar100.json --name vit_cifar100_dev10_v1 --devices cuda:0

bash studies/2026-11-05_2252_s3t_allocation_search/run.sh run \
  --config configs/resnet50_cifar10.json --name resnet50_cifar10_dev10_v1 --devices cuda:0
```

Use `--devices cuda:0 cuda:1` to split one complete search across two GPUs, or
assign different GPUs to the two commands if running them concurrently. The
optional `maximum_gpu_power_watts` setting checks existing power limits on a
server; the bundled configs leave it unset.

## Preparation, outputs, and resume

`prepare` accepts the same `--config`, `--name`, and `--data-root` arguments and
validates the source fingerprints without training. `run` auto-prepares if the
named run does not exist. If preparation was interrupted before `plan.json` was
written, the same command preserves the incomplete folder with an
`.incomplete-<timestamp>` suffix and prepares the named run again. Each run
lives under `runs/<name>/`, with a signed
`plan.json`, copied manifests, exact launch-source snapshot, native trial
records, `progress.json`, individual `worker.log` files, and CSV reports.
Completed candidate model weights are discarded after validation scoring;
their config, timing, predictions, metrics, and a retention receipt remain.
An interrupted member resumes from its checkpoint. Frozen finalists are in
`finalists.json`, bound to a digest of all screening scores.

To pause after active trials, create `STOP` in the run directory. Remove it and
resume with:

```bash
bash studies/2026-11-05_2252_s3t_allocation_search/run.sh resume \
  --run studies/2026-11-05_2252_s3t_allocation_search/runs/vit_cifar100_dev10_v1 \
  --devices cuda:0
```

`status --run PATH` reads the latest progress without training. The folder
prefix records preparation on 2026-11-05 22:52 America/New_York; no search in
this study had completed when it was created. Later executions stay under
`runs/`.
