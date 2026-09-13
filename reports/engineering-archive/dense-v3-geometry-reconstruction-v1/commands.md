# Exact CPU-only verification commands

These commands describe retained attempts, not overwrite destinations. For another audit,
use a new empty directory created with `mktemp -d`; never reuse a retained producer or output.
On this host the pinned interpreter is `/usr/bin/python`, not the live checkout's `.venv`.

All calls used:

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
```

The geometry kernel explicitly selects its frozen four CPU threads. The parent invokes a
fresh child with only the relocated source in PYTHONPATH, removes the old editable-install
search entry and blocks network and old producer/source/model/data paths. Its CLI is
`python -B -m embed_optim.primary_v3_geometry_reconstruction` with an explicit local archive,
external manifest SHA-256 and new output directory. It reads no model checkpoint weights.

Retained tests:

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_geometry_reconstruction.py \
  --junitxml /tmp/dense-v3-geometry-reconstruction-tests.aGzFAJ/focused-retry.xml
/usr/bin/python -B -m pytest -q \
  --junitxml /tmp/dense-v3-geometry-reconstruction-tests.aGzFAJ/full.xml
```

First complete cold audit:

```bash
/usr/bin/python -B scripts/audit_dense_v3_geometry_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --workdir /tmp/dense-v3-geometry-reconstruction-cold.6zffw0
```

Independent replay uses the same script with a new empty `--workdir`, plus `--replay` pointing
to the completed first `result.json` and `--replay-sha256` obtained from its externally retained
receipt. It verifies the original audit's source/artifact bindings before copying the payload.
No running, partial or unanchored receipt may be used as a replay parent.

The complete fixture preserves native axes and all 88 matrices × 60 checkpoints. Its stored
weights/spectra/entry-count metadata is hypothetical and its bases are synthetic one-hot
coordinates. The independent oracle checks those coordinate bases by exact set intersections
and scalar/rational aggregates, not by calling production geometry summaries. Every semantic
control operates on a new `rehashed-*` synthetic copy and updates all affected envelopes to an
explicit new test anchor. This is not permission to rehash or repair actual primary evidence.

For terminal state and exact receipt hashes follow [RUNNING.md](RUNNING.md). Same-host relocation
is not physical cross-host proof. Passing these checks neither releases a training source nor
verifies a scientific optimizer result, primary publication or manuscript installation.
