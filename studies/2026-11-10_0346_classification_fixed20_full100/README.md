# Classification full-data / 100-request companion workers

Companion for BlockUnlearn's
`unlearning/studies/2026-11-10_0346_classification_fixed20_full100/`.
ResNet-50/CIFAR-10 and RoBERTa-base/SST-2 support SISA, ProtoSISA, and S3T;
BlockUnlearn owns SU, BM/GM, FT, plan preparation, and Slurm submission.
The folder prefix records preparation time in America/New_York.

Deploy this folder and matching Block source before preparing server plans.
See the Block study README for exact preparation, submission, status, and
resume commands. No direct companion preparation is needed: workers consume
the frozen Block plan, source-index manifest, and deletion workload. Settings
must have development confirmation before full-data plan preparation succeeds.

All methods use the same final 90% training pool and 100 sequential source-ID
deletions with three training seeds. SISA/ProtoSISA share models only when their
confirmed hyperparameters match. Native ensembles use SAFE prototype weighting.
SST-2 uses the exact prepared length-128 token arrays from its development
search; final accuracy is on the labeled official GLUE validation split (872
examples). CIFAR-10 final evaluation uses the test set (10,000 examples).

Generated native records live under `runs/<run>/records/`; common ensemble
models use the engine's configured `saved_models/` store. Successful jobs
remove their models by default while preserving scores, logs, source identity,
timing journals, and deletion receipts. Interrupted jobs retain resumable
artifacts. Do not delete an unfinished run's model cache.

`portable_lock.py` installs readable lock descriptors in this worker process
for NFS compatibility; it does not edit shared engines or other studies.

From this checkout, offline checks (including real tiny-model 100-request
replay/resume):

```bash
python -m unittest discover \
  -s studies/2026-11-10_0346_classification_fixed20_full100 -p 'test_*.py'
```
