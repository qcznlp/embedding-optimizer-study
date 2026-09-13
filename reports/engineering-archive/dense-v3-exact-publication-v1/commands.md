# Retained CPU-only commands

Use `/usr/bin/python -B`, not the live `.venv`, with:

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

The following commands are retained records; their outputs already exist. Do not
overwrite them or rerun a completed check as missing work. A deliberate reproduction
must use new empty workdirs created by mktemp and new output paths.

```bash
/usr/bin/python -B -m pytest -q \
  tests/test_primary_v3_exact_publication.py tests/test_primary_v3_paper_layout.py \
  --junitxml=/tmp/dense-v3-exact-publication-tests.5ncZ0V/focused-final.xml
/usr/bin/python -B reports/engineering-archive/dense-v3-exact-publication-v1/full_generation.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-exact-publication-retry.lioRRV
/usr/bin/python -B reports/engineering-archive/dense-v3-exact-publication-v1/render_full_layout.py \
  --repository /root/embedding-optimizer-story-refactor \
  --workdir /tmp/dense-v3-exact-publication-layout-retry.E2mqpj \
  --generation /tmp/dense-v3-exact-publication-retry.lioRRV/result.json \
  --generation-sha256 2363e68ee559dfa7235eb395a45f7c78d2883beaa5a174c979b0703fbb962e63
```

The whole-development command has finished as 7103, exit 0; its complete XML is
preserved in this archive. See RUNNING.md for the distinct reported/serialized counts:

```bash
/usr/bin/python -B -m pytest -q \
  --junitxml=/tmp/dense-v3-exact-publication-tests.5ncZ0V/full-first.xml
```

Actual complete-input authoring is exposed through:

```bash
/usr/bin/python -B -m embed_optim.primary_v3_exact_publication --help
```

Its build/inspect modes require complete real primary inputs, both original and exact
result roots, pinned source contracts and an external vector-manifest hash. There is
no synthetic-admission switch. Missing primary runs fail before later input/output
access. The output must be a new ordinary preparation directory outside the paper.
The draft protocol is not a release or permission to train. Do not invoke historical
training/controller commands from earlier archives.
