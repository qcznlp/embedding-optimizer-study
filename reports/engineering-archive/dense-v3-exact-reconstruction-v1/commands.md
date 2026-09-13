# Bounded CPU-only verification

Use `/usr/bin/python -B`, never the live `.venv`. Set:

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

Both complete cold invocations are terminal/pass; these are retained commands, not pending work:

```bash
/usr/bin/python -B scripts/audit_dense_v3_exact_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-exact-reconstruction-cold.LVS9GV
/usr/bin/python -B scripts/audit_dense_v3_exact_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-exact-reconstruction-replay.9Gx3RL \
  --replay /tmp/dense-v3-exact-reconstruction-cold.LVS9GV/result.json \
  --replay-sha256 80d0727dc4422b90255027dba4d2c28e433d40667561d438154898bc6a21717a
```

Each workdir was created empty using mktemp. The independent replay uses a distinct
directory and the externally computed digest of the completed first result.

The focused retry and whole-development commands are:

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_exact_reconstruction.py \
  --junitxml=/tmp/dense-v3-exact-reconstruction-tests.SR979t/focused-final.xml
/usr/bin/python -B -m pytest -q \
  --junitxml=/tmp/dense-v3-exact-reconstruction-tests.SR979t/full-first.xml
```

Their status is tracked in RUNNING.md. Never overwrite those retained XML files.

The completed real-input check and independent full-fixture oracle are:

```bash
/usr/bin/python -B reports/engineering-archive/dense-v3-exact-reconstruction-v1/real_spectrum_readback.py \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-exact-reconstruction-v1/real-spectrum-readback.json
/usr/bin/python -B reports/engineering-archive/dense-v3-exact-reconstruction-v1/full_fixture_oracle.py \
  --archive-root /tmp/dense-v3-exact-reconstruction-cold.LVS9GV/fixture/archive \
  --expected-manifest-sha256 ceefc39c128d541032b73ed6bfbd2773fd5f39f6a81e6be6be68da1079695660 \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-exact-reconstruction-v1/full-oracle-first.json
```

These outputs are already complete. Reproducing a check requires a new output path,
not overwriting an accepted receipt. The synthetic oracle is neither model admission
nor retrieval evidence. No command here authorizes training, deployment or release.

Final combined preservation acceptance (do not overwrite a generated receipt):

```bash
/usr/bin/python -B reports/engineering-archive/dense-v3-exact-reconstruction-v1/validate.py \
  --first /tmp/dense-v3-exact-reconstruction-cold.LVS9GV/result.json \
  --first-sha256 80d0727dc4422b90255027dba4d2c28e433d40667561d438154898bc6a21717a \
  --replay /tmp/dense-v3-exact-reconstruction-replay.9Gx3RL/result.json \
  --replay-sha256 893368c2182c01bd189637df0c02ab1c1c9a489f9e953352e14185cd11cbbc05 \
  --full-xml /tmp/dense-v3-exact-reconstruction-tests.SR979t/full-first.xml \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-exact-reconstruction-v1/validation.json
```
