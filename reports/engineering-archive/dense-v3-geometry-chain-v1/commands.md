# CPU-only reproduction and actual invocation record

Run from `/root/embedding-optimizer-story-refactor`. Use absolute `PYTHONPATH` entries because
test subprocesses change working directories. All shown output paths already exist and are retained;
**supply new explicit output paths for any future invocation**, including JUnit/validation receipts.
These commands neither release primary training nor authorize publication or controller transitions.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/prepare_dense_v3_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --output configs/dense_primary_v3_geometry_protocol_v2.json

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_v3_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-geometry-retry.Pysj3k

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/replay_dense_v3_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --rehearsal /tmp/dense-v3-geometry-retry.Pysj3k/result.json \
  --rehearsal-sha256 0bd4d9290fd672229ee2d3a5cd4c891854c3e951fc47286bf8e8424190c5c665 \
  --workdir /tmp/dense-v3-geometry-replay.jd0MyI

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  pytest -q --junitxml=reports/engineering-archive/dense-v3-geometry-chain-v1/full-tests-v2.xml

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/validate_dense_v3_geometry.py \
  --repository /root/embedding-optimizer-story-refactor \
  --rehearsal-sha256 0bd4d9290fd672229ee2d3a5cd4c891854c3e951fc47286bf8e8424190c5c665 \
  --replay-sha256 526b8e9187d29ff8113bbfa3a7bfc65a3c7708fb752b88f45ec73e3504250d90 \
  --output reports/engineering-archive/dense-v3-geometry-chain-v1/validation.json
```

The actual primary `produce` and `inspect` CLI argument lists, exit codes and missing-primary
refusals are retained in `rehearsal-v2.json`. They require all twelve complete full-horizon v3 runs;
do not substitute old held checkpoints or the diagnostic models. Draft-parent release is not a
status-only edit and still needs reviewed source/runtime/parent integration and owner direction.

The original failed audit ran under protocol `9f11cbcd...` into
`/tmp/dense-v3-geometry.75bnhK`. Its then-executed source and the reduction-localization script are
preserved under `attempt-1/source/`. Those scripts depended on the original source closure; do not
run them with the active v2 modules and claim it reproduces the original execution. Neither old
source nor old expected hashes are refreshed. The active v2 module copies are under `source-v2/`.
