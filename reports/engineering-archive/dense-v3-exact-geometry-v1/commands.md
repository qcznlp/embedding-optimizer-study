# Executed CPU-only verification commands

All commands used the isolated checkout and absolute source paths, with
`CUDA_VISIBLE_DEVICES=` and `PYTHONDONTWRITEBYTECODE=1`. Numerical audits additionally used
`OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. No GPU worker was launched.
The command paths below are retained evidence namespaces; use new empty namespaces for a future
rerun. Never overwrite a receipt or recreate the frozen protocol over existing evidence.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /root/embedding-optimizer-story-refactor

python -B scripts/prepare_dense_v3_exact_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --output configs/dense_primary_v3_exact_geometry_protocol.json

python -B -m pytest -q -p no:cacheprovider tests/test_primary_v3_exact_geometry.py \
  --junitxml=reports/engineering-archive/dense-v3-exact-geometry-v1/focused-tests.xml
python -B -m pytest -q -p no:cacheprovider \
  --junitxml=reports/engineering-archive/dense-v3-exact-geometry-v1/full-tests.xml

python -B scripts/audit_dense_v3_exact_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_primary_v3_exact_geometry_protocol.json \
  --workdir /tmp/dense-v3-exact-geometry.CBVst9

python -B scripts/audit_dense_v3_exact_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_primary_v3_exact_geometry_protocol.json \
  --workdir /tmp/dense-v3-exact-replay.7jjL3Y \
  --replay /tmp/dense-v3-exact-geometry.CBVst9/result.json \
  --replay-sha256 113815a561f65d4d9461f54308715f5212fdc3890f7dafb396629682eb0cd100

python -B scripts/audit_bridge_null_feature.py \
  --repository /root/embedding-optimizer-story-refactor \
  --output reports/engineering-archive/dense-v3-exact-geometry-v1/bridge-null-feature.json

python -B scripts/validate_dense_v3_exact_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-exact-geometry.CBVst9/result.json \
  --audit-sha256 113815a561f65d4d9461f54308715f5212fdc3890f7dafb396629682eb0cd100 \
  --replay /tmp/dense-v3-exact-replay.7jjL3Y/result.json \
  --replay-sha256 910b4de66f0fcf2012f5d38d1794640b139936a957694df0517c8caa6fd291ac \
  --output reports/engineering-archive/dense-v3-exact-geometry-v1/validation.json
```

Focused testing passed 45 cases; the full suite passed 2,051, with no failures/errors/skips.
The initial actual audit and fresh replay ended successfully at 21:15 and 21:25 UTC respectively.
The final evidence receipt authenticates the null-feature counterexample but explicitly does not
accept the unchanged bridge as correct or claim that its issue has been repaired.

The artifact-only progress refresh runs from the live checkout with the same isolated source path:

```bash
cd /root/embedding-optimizer-study
python -B -m embed_optim.corrected_progress \
  --output /root/embedding-optimizer-story-refactor/CURRENT_PROGRESS.json
```
