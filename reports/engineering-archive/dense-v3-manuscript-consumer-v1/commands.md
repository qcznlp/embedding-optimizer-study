# Exact CPU-only document checks

These commands are retained records, not instructions to overwrite their outputs.
Use `/usr/bin/python -B`, not the old live virtual environment, with:

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

Completed focused and full-document calls:

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_manuscript.py \
  --junitxml=/tmp/dense-v3-manuscript-tests.gfHZZN/focused-first.xml
/usr/bin/python -B -m pytest -q \
  tests/test_primary_v3_manuscript.py tests/test_primary_v3_manuscript_build.py \
  --junitxml=/tmp/dense-v3-manuscript-tests.gfHZZN/focused-build-first.xml
/usr/bin/python -B reports/engineering-archive/dense-v3-manuscript-consumer-v1/complete_document.py \
  --repository /root/embedding-optimizer-story-refactor \
  --workdir /tmp/dense-v3-complete-document.I0a0Bj \
  --generation /tmp/dense-v3-exact-publication-retry.lioRRV/result.json \
  --generation-sha256 2363e68ee559dfa7235eb395a45f7c78d2883beaa5a174c979b0703fbb962e63
```

The whole-development command finished as 5462, exit 0; its XML is preserved:

```bash
/usr/bin/python -B -m pytest -q \
  --junitxml=/tmp/dense-v3-manuscript-tests.gfHZZN/full-first.xml
```

For intentional future reproduction choose a new `mktemp -d` output root and a
new XML path. Do not restart any completed or currently live call merely because
an old status note says it was pending. The fixture uses synthetic presentation
values; no command here runs formal training, installs manuscript results,
transitions a controller or publishes artifacts.
