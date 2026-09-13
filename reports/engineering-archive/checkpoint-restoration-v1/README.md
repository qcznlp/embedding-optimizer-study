# Independent checkpoint download rehearsal

Engineering only, executed on September 5, 2026. No numerical experiment source or live controller
was modified. The fresh directory remains `/tmp/dense-checkpoint-restore.Og1kZs/download`; it is a
temporary local copy, not the durability source. The original immutable HF revision remains public.

## Actual download

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 \
hf download qcz/embedding-optimizer-study-checkpoints \
  --revision e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae \
  --include 'corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/checkpoint-3126/**' \
  --include 'corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/run_config.json' \
  --local-dir /tmp/dense-checkpoint-restore.Og1kZs/download \
  --force-download --max-workers 2
```

The first attempt also supplied a `--cache-dir` and was rejected by the installed HF CLI before
downloading: local directory and cache directory are mutually exclusive in that CLI. The supported
command above exited 0. It retrieves only the exact public stage/configuration and uploads nothing.

## Actual offline diagnostic

From `/root/embedding-optimizer-story-refactor`:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src \
python -m scripts.audit_checkpoint_download \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /root/embedding-optimizer-story-refactor/reports/experiment-integrity/huggingface-digest-audit-ten-run.json \
  --audit-sha256 9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6 \
  --run-id padded-normuon-3e-4 --step 3126 \
  --download-root /tmp/dense-checkpoint-restore.Og1kZs/download \
  --output /root/embedding-optimizer-story-refactor/reports/experiment-integrity/checkpoint-download-normuon-3126.json
```

This succeeded at 03:34 UTC. Use a new output receipt for any rerun; the command refuses to replace
the original. It verifies 18 files / 1,351,464,827 bytes, two short encodings, and exact CPU
optimizer/scheduler continuation of all 134 parameters across steps 3126 → 3127 → 3128. The original
payloads remain unchanged. It does not restore data-loader/rank RNG, run GPU training or score BEIR.
No original live checkpoint is read, but the source/configuration checkout is still used: this is
not a network-isolated clean operating-system image or a second physical machine.

## Checks and boundaries

- Initial lint identified only import ordering and an unused test import; both were fixed.
- `checkpoint-download-focused-tests.xml`: 31 tests pass.
- `checkpoint-download-full-tests.xml`: 1,150 tests pass with no exclusions.
- `checkpoint-download-documentation-tests.xml`: final documentation and download guards.
- The first W&B observation used an unsupported timestamp property; the same exact run handles
  were re-polled using supported state/step fields. Neither read-only diagnostic error indicates a
  training fault. The successful 03:39 observation is retained separately.

All receipts above are under `reports/experiment-integrity/`. The restoration guide and new script
remain isolated pending the existing source-publication approval. No HF deletion, runtime migration,
source commit/push, keeper inspection or training intervention occurred. This work supplies no paper
optimizer result and belongs outside the manuscript.
