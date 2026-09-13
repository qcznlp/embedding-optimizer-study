# CPU-only reproduction commands

Run from `/root/embedding-optimizer-story-refactor`. Always use absolute `PYTHONPATH` entries:
paper-build subprocesses change their working directory. None of these commands authorizes formal
training, deployment, publication or an old-controller transition. Existing outputs are immutable;
the producer/replay/validator require new explicit output paths on every later invocation.

The preparation invocation was:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/prepare_dense_weight_entries.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --output configs/dense_weight_entry_measurement_amendment.json
```

The actual reference copy was made by `cp --dereference --update=none --parents`, from the pinned
HF snapshot, for precisely the eleven files in `common_identity.model.files`. The successful audit
verifies all copy/original-blob digests against that already frozen manifest; there was no download
or mutation of the snapshot. Original work paths below are retained and cannot be reused as fresh
output directories.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/audit_dense_weight_entries.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --reference /tmp/dense-weight-entry-reference.Um8TxW \
  --workdir /tmp/dense-weight-entry-census-retry.DgbmCc

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/replay_dense_weight_entries.py \
  --repository /root/embedding-optimizer-story-refactor \
  --rehearsal /tmp/dense-weight-entry-census-retry.DgbmCc/result.json \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-weight-entry-replay.a9BFoN

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  pytest -q \
  --junitxml=reports/engineering-archive/dense-v3-geometry-v1/full-tests-final-absolute-path.xml

env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B scripts/validate_dense_weight_entries.py \
  --repository /root/embedding-optimizer-story-refactor \
  --output reports/engineering-archive/dense-v3-geometry-v1/validation.json
```

Tests exercise actual file/array readers and separately declared synthetic counterexamples.
The full suite's layout PDFs contain synthetic scores and are not experiment results. The actual
missing-primary CLI command, exit code and refusal are preserved inside `rehearsal.json`.
Do not run formal geometry on old held checkpoints or pass diagnostics off as the twelve-run matrix.
