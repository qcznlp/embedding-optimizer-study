# Original-outcome CPU reconstruction commands

These are diagnostic commands, not training, primary admission or publication commands. Existing
output paths below are preserved evidence; a new attempt must use a fresh directory and XML path.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
```

The host's verified interpreter is `/usr/bin/python`, not the live `.venv`. Reconstruction checks
Python and the recorded package/CUDA-build versions even though the calculation is CPU-only.

## Targeted and complete regressions

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_outcome_reconstruction.py \
  --junitxml /tmp/dense-v3-outcome-reconstruction-tests.QX7X3H/focused-retry.xml
/usr/bin/python -B -m pytest -q \
  --junitxml /tmp/dense-v3-outcome-reconstruction-tests.QX7X3H/full-retry.xml
```

The focused retry is terminal/pass for 62 cases, and the full retry is terminal/pass for 2,481
cases, with no failures, errors or skips. Use [the execution record](RUNNING.md). Initial focused/full failures and their exact source and inputs are
retained in [attempt-1](attempt-1/README.md); they must not be silently replaced by retry files.

The unit fixture synthesizes its own 4,096 row identities and all scores and does not need old
private data directories. The independent audit instead retains the previously verified actual
validation row identities from the immutable first raw-scoring fixture. Both simulate checkpoint
admission, scores, timing and hardware; neither creates model-derived findings.

## First complete cold audit

```bash
/usr/bin/python -B scripts/audit_dense_v3_outcome_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --workdir /tmp/dense-v3-outcome-reconstruction-cold.GsLOPE
```

Owned session 68209 is terminal/pass. Its complete receipt is 9,738,347 bytes, SHA-256
`3c6f3d86bfc84907ab780570ca95df0f21e7196343959aa6fd965c25604757dc`.
The fixture receipt is 2,061 bytes, SHA-256
`7c8179c8586eee421e1130fa190a2767e75c76f538fe6aa5366b0bb476ec4138`.
It anchors local payload manifest
`02cb5c5f977bcbcdd49938be32c9c5c9faaf1a85887e445653b4051a1277d681`.

The child removes the forbidden old editable-install search entry, then keeps the old-location
and network access guard active for all package imports and numerical work. All 71 imported
project modules come from the archived source role. The positive reconstruction replays every
raw validation record and BEIR task, reconstructs all ten tables and preserves the original
provenance strings. It checks the original statistical kernel; its independent numerical
reference checks remain in the previously accepted outcome parent, not a newly claimed oracle.

Each of 21 controls runs in its own fresh child with all affected envelopes rehashed and a new
explicit **test** anchor. The aggregate control changes the primary mean by one floating-point
unit; fresh numerical output remains, but no successful reconstruction receipt is written.
The eight-way FP32-to-scalar validation tolerance remains unchanged and is not a selection tolerance.

## Independent second-directory replay

```bash
/usr/bin/python -B scripts/audit_dense_v3_outcome_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --workdir /tmp/dense-v3-outcome-reconstruction-replay.2PrtN4 \
  --replay /tmp/dense-v3-outcome-reconstruction-cold.GsLOPE/result.json \
  --replay-sha256 3c6f3d86bfc84907ab780570ca95df0f21e7196343959aa6fd965c25604757dc
```

Owned session 40528 is terminal exit 0. Its 9,013,817-byte receipt has SHA-256
`ae0d73c8079be783077a660dedfc562a1e16eaede0e2139a4e00ea7ebf858295`.
The replay verifies the complete first
audit's retained files, copies its payload to another directory, and compares the fresh child
receipt exactly before repeating every semantic refusal. This is same-host relocation, not
execution on another physical machine, and does not rerun any accepted vector reconstruction.

## Actual portable reader

For a future externally authenticated archive and matching Python environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/path/to/archive/source/src \
  /path/to/verified/python -B -m embed_optim.primary_v3_outcome_reconstruction \
  --archive-root /path/to/archive \
  --expected-manifest-sha256 EXTERNALLY_TRUSTED_ARCHIVE_SHA256 \
  --output /path/to/new-outcome-reconstruction
```

Remove old project editable-install search paths if present; never disable source-byte checks or
allow producer-path fallback. An anchor calculated from untrusted local content is not proof of
checkpoint-backed authoring. This consumer recomputes statistics, not weights, text, encoding,
retrieval, timing, geometry, functional inference or final publication. Its primary/science flags
remain false. Actual missing primary authoring is still rejected before any output is created.

## Bounded acceptance after all terminal checks

The validator also rechecks the immutable vector parent, failed attempts, retained inputs,
current source copies, unchanged live numerical files and the original main ledger. It cannot
accept a missing/failed full regression or turn this fixture into primary evidence:

```bash
/usr/bin/python -B scripts/validate_dense_v3_outcome_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --fixture /tmp/dense-v3-outcome-reconstruction-cold.GsLOPE/fixture.json \
  --fixture-sha256 7c8179c8586eee421e1130fa190a2767e75c76f538fe6aa5366b0bb476ec4138 \
  --audit /tmp/dense-v3-outcome-reconstruction-cold.GsLOPE/result.json \
  --audit-sha256 3c6f3d86bfc84907ab780570ca95df0f21e7196343959aa6fd965c25604757dc \
  --replay /tmp/dense-v3-outcome-reconstruction-replay.2PrtN4/result.json \
  --replay-sha256 ae0d73c8079be783077a660dedfc562a1e16eaede0e2139a4e00ea7ebf858295 \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-outcome-reconstruction-v1/validation.json
```

Never overwrite an existing receipt. The accepted flags remain limited to synthetic original-
outcome reconstruction; geometry/inference, complete primary publication and formal release are
separate pending surfaces.
