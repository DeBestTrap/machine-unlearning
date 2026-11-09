# ViT/CIFAR-100 full-data, 100-request comparison: companion worker

This folder provides SISA/ProtoSISA and S3T execution for the corresponding
BlockUnlearn `unlearning/studies/2026-11-09_0417_vit_fixed20_full100/` study.
The BlockUnlearn study owns the frozen settings, exact 45,000/5,000 train/dev
manifest, common seed-123 workload, Slurm launcher and full documentation.
Deploy both folders; submit the BlockUnlearn `run_full100.sbatch` with explicit
checkout paths. The job requests three GPUs and assigns one to each seed.

`worker.py` is invoked by the owner coordinator. It trains one shared 20-shard
SISA ensemble for SISA and ProtoSISA, preserving their development-selected
last-four-block ViT scope and Xavier classifier initialization. It replays all
100 distinct source IDs, updating the affected shard and the native frozen
feature prototypes, and evaluates the same official test set after each request.
Prediction caching reuses unchanged checkpoints and immutable test features;
model updates and prototype maintenance retain their original implementation.

For S3T, it runs the native complete ensemble and native
`on_exhaustion=retrain` lifecycle, including test accuracy after every request.
Initial training is excluded from reported maintenance time, and publication
timers are used without also counting their nested generation timers.

Generated records are under `runs/RUN_NAME/records/`. SISA model checkpoints
remain in the engine's configured `saved_models/` store. S3T uses its native
record directory. Successful jobs discard models unless the owner was prepared
with `--keep-models`; interrupted jobs retain their checkpoints for resume.
Every saved result belongs to the owner's signed plan and captured sources.

For purpose, exact inputs/configurations, submission, output locations and
resume limitations, read the owner study's README. No training starts by
copying these files or running the following offline checks:

```bash
python -m unittest studies.vit_fixed20_full100.test_worker
```

The two repositories use the same stable study slug `vit_fixed20_full100` in
their respective Python package namespaces. The folder prefix records study
creation, not a completed full-data result.
