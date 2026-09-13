# Restore all continuation probes and locate their checkpoints

All **638 files / 586,119,383 bytes** were uploaded, anonymously downloaded and
independently checksum-verified on 2026-09-12. A copied-source CPU replay then
reproduced every saved score and all **5,490 metric values** across sixty
continuation checkpoints plus the pretrained reference exactly.

- [Public immutable data snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/cff3f190e169548931fbd33eadcf1279439798e1/corrected-dense-correctness-v3/continuation-probes-v1/bcb092fe16e11150abc678a6b1e977497afbc6bf977b9dd9085cebff37004081).
- Repository: `qcz/embedding-optimizer-study-analysis-artifacts` (dataset).
- Revision: `cff3f190e169548931fbd33eadcf1279439798e1`.
- Manifest SHA-256: `bcb092fe16e11150abc678a6b1e977497afbc6bf977b9dd9085cebff37004081`.
- [Actual transport and replay evidence](../reports/engineering-archive/dense-v3-factorial-probe-durability-v1/README.md).

## Download and verify

The standalone [restoration script](../scripts/restore_factorial_probes.py) uses
`huggingface_hub` for anonymous downloading and only standard Python for file
verification. It imports no model/GPU package and never deserializes pickle files.

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/restore_factorial_probes.py \
  download --path /data/dense-continuation-probes-NEW
```

The destination must not exist; failed attempts are preserved. The script prints
`snapshot_root`, the directory containing `artifact_manifest.json`. Use that
directory, not its download parent, for subsequent commands:

```bash
python scripts/restore_factorial_probes.py verify --path /path/to/snapshot_root
python scripts/restore_factorial_probes.py list-checkpoints --path /path/to/snapshot_root
```

The second command validates and lists all sixty existing model checkpoint links,
including immutable revisions and exact remote prefixes. The included
`checkpoints/<run>/artifact_manifest.json` binds all eighteen files per checkpoint;
`verified.json` records its original anonymous HF metadata verification. Models
remain in `qcz/embedding-optimizer-study-checkpoints`, not in this probe dataset.
This command does not redownload models or establish GPU-resume equivalence.
The original primary matrix has its separate [checkpoint guide](checkpoint-restoration.md).

## Data layout

Each `factorial-v3-<state>-<operator>-seed<seed>/checkpoint-<step>/` contains:

| File | Preserved measurement |
| --- | --- |
| `raw/vectors.npz` | Original normalized FP32 query/document vectors and ordered sample IDs/task groups |
| `raw/manifest.json` | Original encoder, model identity and native vector provenance |
| `vectors-fp16.npz` | Exact FP16 vector inputs used by the frozen metric scorer |
| `scores.npz` | Original FP32 cosine scores, shape `[224, 8]` |
| `metrics.json` | Six original overall and per-task metrics |
| `worker.completed.json` | Original completed-worker source and output bindings |

Query and document vectors have shapes `[224, 768]` and `[224, 8, 768]`. The fixed
eight candidates have the relevant document at index zero. Each of fourteen task
groups contributes sixteen queries. Steps are **79, 157, 235, 313, 391**, covering
all four state/operator cells under orders **314159, 271828, 161803**. `pretrained/`
provides the authenticated reused reference in original FP32 and scoring FP16 form.
Open NPZ files with `numpy.load(..., allow_pickle=False)`.

## Recompute scores and metrics from copied sources

The [closed numerical entry](../reports/engineering-archive/dense-v3-factorial-probe-durability-v1/actual/numerical-replay/replay.py)
and its adjacent `source/` directory extract only the two unchanged pure scoring
and summary functions. Copy the entire `numerical-replay` directory and restore
the immutable data snapshot; original producer/model directories are not used.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 python /path/to/numerical-replay/replay.py \
  --snapshot /path/to/snapshot_root --output /data/probe-replay-NEW.json
```

The replay checks every FP32-to-FP16 projection, saved score and original metric
without a tolerance adjustment. It does not re-encode text or replace the
checkpoint-backed native readers. The actual tested Torch/NumPy versions are in
the [replay receipt](../reports/engineering-archive/dense-v3-factorial-probe-durability-v1/actual/numerical-replay/actual-replay.json).
Exact replay was demonstrated on this host using downloaded data and copied
source; identical floating-point execution on every other CPU is not asserted.

No raw example text or executable source was uploaded in the data snapshot.
The scripts are currently in the local worktree; the current GitHub source release
is still pending. Absolute paths inside original receipts are provenance only.
These are eight-candidate diagnostic metrics, not full-corpus BEIR results,
factorial-effect inference or a completed scientific paper.
