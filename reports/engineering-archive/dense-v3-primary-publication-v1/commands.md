# Reproduce the bounded preparation checks

These commands neither train nor publish. Use this development checkout and the
pinned CPU-only interpreter. Do not run another copy of the live regression.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

The 63 focused tests use full synthetic table populations, schema controls and
real missing-primary refusals. Always choose a new XML path:

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_publication.py \
  --junitxml=/absolute/new-location/focused.xml
```

The existing canonical draft is immutable. To independently prepare another
draft without reading primary outcomes, use a new output file:

```bash
/usr/bin/python -B -m scripts.prepare_dense_v3_publication \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --output /absolute/new-location/protocol.json
```

`probe_complete.py` uses the complete, externally anchored joint reconstruction
on this host. It does not read original producer paths as a fallback, admit
checkpoints, or provide portable publication. Give it a new empty directory.
Its optional previous-result comparison is specifically against the pre-label
six-output generation, not an arbitrary current result:

```bash
/usr/bin/python -B reports/engineering-archive/dense-v3-primary-publication-v1/probe_complete.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --workdir /absolute/new-empty-directory \
  --previous /tmp/dense-v3-primary-publication-retry.qfy7oX/result.json \
  --previous-sha256 ca672b515fa520aa132f5bab0faf826029e2684e2690426b11cb09c536fa2cde
```

`render_full_layout.py` consumes a successful current generation plus its
externally recorded result SHA, copies the complete paper into a new directory,
adds an explicit synthetic banner and uses the existing synthetic factorial
layout fixture. It compiles only that copy. Its successful 11-page sample has
the main endpoint on page 7; the remaining pages are references/appendix.
The manuscript, strict publication gate and real experiment remain incomplete.

Exact sources, first failures and command handles are retained in source-first/,
exact-csv-attempt-1/, layout-attempt-1/, source-current/ and RUNNING.md.
