# Actual CPU-only commands

Run from `/root/embedding-optimizer-story-refactor`. All output paths shown already exist and are
retained. Supply **new explicit paths** for future attempts; do not overwrite prior receipts.
Use absolute source paths, including for pytest subprocesses that change working directory.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/prepare_dense_geometry_robustness.py \
  --repository /root/embedding-optimizer-story-refactor \
  --output configs/dense_geometry_robustness_protocol.json

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_geometry_robustness.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_geometry_robustness_protocol.json \
  --workdir /tmp/dense-geometry-robustness.DNQw2T

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_geometry_robustness.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_geometry_robustness_protocol.json \
  --workdir /tmp/dense-geometry-robustness-replay.E7a4yz \
  --replay /tmp/dense-geometry-robustness.DNQw2T/result.json \
  --replay-sha256 ca916d9169d2398bc20e2db1d1393b162671850315930d975d313d1c2d49960c

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_probe_coupling.py \
  --repository /root/embedding-optimizer-story-refactor \
  --parent /tmp/dense-geometry-robustness.DNQw2T/result.json \
  --output /tmp/dense-probe-coupling.tth6XL/result.json

env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_probe_coupling.py \
  --repository /root/embedding-optimizer-story-refactor \
  --parent /tmp/dense-geometry-robustness.DNQw2T/result.json \
  --output /tmp/dense-probe-coupling.tth6XL/replay.json \
  --replay /tmp/dense-probe-coupling.tth6XL/result.json \
  --replay-sha256 e182110128a7e36204ddea9bbfd2919f80754ca1872d434294839b49e5aefe48

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  pytest -q --junitxml=reports/engineering-archive/dense-geometry-robustness-v1/full-tests-final.xml

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/validate_dense_geometry_robustness.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit-sha256 ca916d9169d2398bc20e2db1d1393b162671850315930d975d313d1c2d49960c \
  --replay-sha256 c048e330a1edb1db2ed155b2c2425b5b31f46541426a201c3b9f203acd1bad8c \
  --coupling-sha256 e182110128a7e36204ddea9bbfd2919f80754ca1872d434294839b49e5aefe48 \
  --coupling-replay-sha256 45db3e8563492e89c6f22ba969fd71a4ea3e2566885a4a921c98a56ca08c93f3 \
  --output reports/engineering-archive/dense-geometry-robustness-v1/validation.json
```

The shared-probe follow-up's fixed plan is embedded in its source and copied into both receipts.
It was declared after the parent approximation observations, before the crossed-seed calculation;
it is expressly post-hoc diagnostic work and does not amend the primary protocol. The first failed
unit-test XML is retained separately from its revised retry. No formal training, inference outcome
or publication release is authorized by these commands.
