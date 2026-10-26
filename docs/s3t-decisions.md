# S3T implementation decisions

Implementation: `s3t/`. Specification reference:
[ICLR 2025 paper, arXiv v3](https://arxiv.org/html/2406.16257v3).
The [public release](https://github.com/brcsomnath/S3T) was inspected as context;
this implementation does not depend on that repository or its training scripts.

## Dependency isolation and classifier heads

Transformer blocks are assigned from the output end toward the input. At stage
`j`, only the new group's query/value LoRA parameters and the permitted head
parameters train. Data is the union of the first `j+1` slices in that sequence.
Earlier adapter groups stay active and frozen; later groups are disabled. The
pretrained backbone, all its biases, embeddings, and normalization parameters
remain frozen. Adapter weights are never merged into the base weights.

The paper does not provide a complete classifier-head policy for its
single-checkpoint deletion description. The public classification scripts train
the head at multiple stages. Keeping the last such head after disabling later
adapters would retain a dependency on later data.

**Optional policy: `head_policy="snapshot"`.** Train the head at every stage,
initialize it from the preceding stage's head, and preserve its state after each
stage. Serving prefix `k` activates only adapter groups `<k` and restores head
snapshot `k-1`. All adapter groups and head snapshots live in one sequence
checkpoint; the full backbone is stored only once per run. This adds head-state
storage, which is included in reported checkpoint size. It is an explicit
implementation choice, not a claim that the authors used this policy.

**Default decision: `head_policy="first_slice"`.** Train the head only at stage zero
and freeze it afterward. Surviving prefixes then use that same head. This keeps
the head dependent only on the first slice, at a potential accuracy cost. This
became the default at the user's request; earlier validation runs explicitly
recorded `snapshot` in their saved configurations.

A prefix of length zero is never included in an ensemble. If deletion touches
the earliest slice in a sequence, that sequence has no usable trained prefix.
Later deletions can only shorten the prefix. Inactive trained weights remain in
research artifacts but are never read by an active forward branch.

The guarantee being implemented is absence of fine-tuning dependencies in the
active predictor, conditional on the fixed backbone, partition, and randomness.
It discards entire affected suffixes. It does not assert equality with ordinary
full fine-tuning on every remaining example, nor does it unlearn pretraining data.

## Backbone and parameter choices

- Local comparison configs use torchvision ViT-B/16 ImageNet-1K weights and
  RoBERTa-base to match the existing repo's backbone choices. Separate HF ViT
  ImageNet-21K and RoBERTa-large configs are provided. These presets must not be
  mixed when attributing accuracy differences to unlearning methods.
- LoRA targets the query and value projections only. The torchvision fused-QKV
  projection is evaluated with explicit scaled-dot-product attention so that
  its inference fast path cannot bypass adapters.
- `scaling="lora"` uses `alpha/r`; `"rslora"` uses `alpha/sqrt(r)`. Conventional
  scaling is the default. The public release uses rsLoRA and has some settings
  that differ from the paper's hyperparameter table; both choices are explicit.
- Adapters start with Kaiming-initialized A and zero B. Trainable head weights
  use the same initialization families as the existing local backbones.
- `layers_per_slice` is explicit in supplied configs. If omitted, it is the
  integer floor of depth/slices. Any leftover bottom blocks stay frozen; invalid
  layouts are rejected. The RoBERTa-large preset uses 21 of 24 blocks (7×3).
- Full precision is the default. Optional `precision="bf16"` uses autocasting;
  it changes numerical behavior and is recorded in outputs.

## Optimization, randomness, and saved states

`epochs_per_slice` is the number of passes over each cumulative prefix. It is
not a full-dataset-equivalent budget and does not use SISA's epoch redistribution.
Each stage gets a fresh AdamW optimizer by default, learning rate as configured,
weight decay 0.01, and gradient clipping at norm 1. No early stopping or
test-driven checkpoint selection is performed. Constant learning rate is the
default; linear/cosine schedules and warmup are optional and restart per stage.

Training stores the final epoch of each stage. Earlier adapter parameters never
enter a later optimizer, so weight decay and stale momentum cannot update them.
Within an interrupted stage, optimizer and scheduler state are restored. Every
epoch deterministically seeds its shuffle, worker transforms, and dropout from
the experiment seed and generation/shard/sequence/stage/epoch indices. Different
members can therefore be scheduled in any order. Exact reproducibility across
different hardware or package versions is not assumed.

Initialization stores one shared pretrained base plus the initial head. Each
sequence receives independently seeded adapter initialization. New retained-data
generations reuse that same base and initial head and get new deterministic
partition/optimization seeds. All run settings, versions, code hashes, and a
source snapshot are preserved. The source snapshot is a record; running `--run`
uses the currently checked-out implementation, so deliberate code changes should
normally use a new run directory.

## Sequence selection and inference

- `cyclic` implements whole-sequence rotations for budgets up to the slice
  count. For larger budgets, it expands rotations of successively shorter
  suffixes, round-robin across parents. The paper explicitly leaves some
  corner-case budget heuristics unspecified; this deterministic ordering is our
  choice. Exactly B distinct sequences are generated, without enumerating L!.
- `bms` uses maximum-weight bipartite matching for B≤L and the paper's printed
  weighted score in Eq. (24). That expression is treated as a selection score;
  it is not presented as a separately validated expectation formula.
- For B>L with nonuniform priors, conditional sampling uses weight `1-p` for
  each remaining slice. If rejection sampling stalls near full budget or with
  degenerate priors, lexicographic completion fills the remaining distinct
  sequences. The sampling weight and completion rule are documented choices.
- `random` samples distinct permutations with an independent RNG and is useful
  as an ablation. Sequence lists are persisted, not regenerated during deletion.
- Default `selection="longest_prefix"` chooses the longest surviving prefix
  within each shard, with lowest sequence ID breaking ties. This follows the
  paper's worked example and does not consult test metrics.
- Optional `selection="validation"` compares the saved validation accuracy for
  each candidate's **current surviving prefix**. A score from a subsequently
  deleted stage cannot influence that decision. Ties prefer the longer prefix,
  then the lower sequence ID.
- Each active shard contributes one model. Logits are averaged with equal shard
  weights by default; probability averaging and majority voting are explicit
  alternatives. Exhausted shards are omitted. All-exhausted ensembles return
  no accuracy and require rebuilding.

## Data and experiment protocol

The shared train/validation manifests and original source IDs are reused.
Sharding and slicing happen only after the training pool is selected. Random
partitioning keeps every example, including division remainders. Validation and
test preprocessing are deterministic. CIFAR uses its official test pool; SST-2
uses its labeled official development pool for final evaluation, consistent with
the existing local protocol. SST-2 training tokens are the repo's prepared
RoBERTa arrays (length 128), with both raw-source provenance and array hashes.

Deletion workloads use the shared saved batches, not freshly sampled requests.
Retained-data rebuilding starts a separate generation and excludes all original
source IDs deleted so far. The default replay policy stops at exhaustion so
deletion-capacity curves have a clear endpoint. Optional automatic rebuilding is
explicit in the command and recorded with separate generation costs.

## Tests that defend these choices

- Compare every earlier prefix's predictions with predictions after later
  training and suffix deactivation, exactly on small ViT and RoBERTa models.
- Check that backbone parameters and earlier adapter tensors stay bitwise fixed
  across subsequent AdamW stages, including nonzero weight decay.
- Verify that head restoration is necessary and works, and that the first-slice
  head alternative remains frozen.
- Compare resumed training with uninterrupted training tensor for tensor.
- Exhaustively check small sequence budgets and deletion subsets, including
  exhaustion, source-ID membership, and shard/slice remainder coverage.
- Replay portable workloads across process restarts, reject incompatible
  configs/workloads/caps, and rebuild using only retained examples.

These are implementation checks, not evidence that the paper's reported accuracy
or latency has been reproduced. See the run artifacts and validation report for
the experiments actually executed in this checkout.
