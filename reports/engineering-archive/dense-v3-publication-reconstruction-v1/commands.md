# Bounded CPU-only development checks

Use the pinned `/usr/bin/python`, not the live `.venv`. Set CUDA_VISIBLE_DEVICES
to the empty string; use PYTHONDONTWRITEBYTECODE=1, OMP_NUM_THREADS=1,
OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1, all three HF/HF_DATASETS/TRANSFORMERS
offline flags to 1, WANDB_MODE=disabled, and absolute development src/repository
PYTHONPATH. These commands do not authorize training, deployment or publication.

Focused test (98303, terminal/pass):

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_publication_reconstruction.py \
  --junitxml=/tmp/dense-v3-publication-transport-tests.cPEgNC/focused.xml
```

First complete numerical invocation (40446, see RUNNING.md before any polling):

```bash
/usr/bin/python -B scripts/audit_dense_v3_publication_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-publication-cold.PvOmLl
```

The work directory must be newly created and empty. Do not rerun this command
against the retained directory or start another job merely because an observation
timed out. A later independent replay requires a new directory and the completed
first result's externally obtained SHA-256; it retains the same source closure.

Independent replay 64182 is terminal/pass:

```bash
/usr/bin/python -B scripts/audit_dense_v3_publication_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-publication-replay.kVna8I \
  --replay /tmp/dense-v3-publication-cold.PvOmLl/result.json \
  --replay-sha256 abf6d68f5eab6d457edec92d25448e7ed4d0a7137db0402ec4eea0a6d36e601c
```

The prepared acceptance command rechecks both terminal results and preservation;
it neither repeats numerics nor authorizes publication:

```bash
/usr/bin/python -B reports/engineering-archive/dense-v3-publication-reconstruction-v1/validate.py \
  --first /tmp/dense-v3-publication-cold.PvOmLl/result.json \
  --first-sha256 abf6d68f5eab6d457edec92d25448e7ed4d0a7137db0402ec4eea0a6d36e601c \
  --replay /tmp/dense-v3-publication-replay.kVna8I/result.json \
  --replay-sha256 2e51be2d5623bf78e90661a3fa0064aad8429912e943458dc7b479914a56ed0a \
  --full-xml /tmp/dense-v3-publication-reconstruction-full-tests.3kdRv5/full.xml \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-publication-reconstruction-v1/validation.json
```

Do not overwrite a completed acceptance or rerun completed sessions as missing work.
