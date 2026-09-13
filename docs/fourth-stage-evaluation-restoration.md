# Restore all twelve corrected fourth-stage BEIR outcomes

The complete step-3126 snapshot contains all twelve optimizer/rate configurations,
with fourteen full-corpus task scores per checkpoint. Six original pool-B states
overlap the preceding [partial fourth-stage snapshot](fourth-stage-pool-b-evaluation-restoration.md);
only six original pool-A states / 84 scores are newly backed up. No previous
snapshot is replaced, and overlaps are not additional experiments.

## Immutable recovery target

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc`.
- Prefix:
  `corrected-dense-correctness-v3/complete-fourth-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c`.
- Manifest: `artifact_manifest.json`, 178,694 bytes, SHA-256
  `21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c`.
- Complete snapshot: **557 files / 1,959,719 bytes**, including 216 native result/
  metadata/acceptance files, 336 original worker records, two CSVs and provenance.

The manifest's external digest must come from the trusted handoff; do not trust a
replacement manifest simply because it lists its own hashes. Use the exact revision
and prefix above, not the moving default branch or the mixed-history repository root.

## Download without a token

Use `huggingface_hub` (the actual transport used version 1.28.0) and a new destination:

```python
from huggingface_hub import snapshot_download

prefix = (
    "corrected-dense-correctness-v3/complete-fourth-stage-evaluations/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc",
    allow_patterns=[prefix + "/**"],
    local_dir="/absolute/path/to/new-fourth-stage-download",
    token=False,
    max_workers=2,
)
```

The actual evidence uses per-file anonymous downloads of exactly these 557 paths,
followed by full inventory/hash checks. The convenience snippet above was not a
separate second transport experiment. Client cache metadata is outside the supplied
snapshot prefix and should not be included as a result payload.

Actual full anonymous recovery completed **2026-09-12 11:23:59 UTC**. The copied
standalone reader returned zero at **11:25:49 UTC** and reproduced every declared
score, successful worker record and exact mean. This was same-host recovery into
a fresh directory, not a physical second-host model experiment.

## Independently reconstruct the native scores

Copy the local standalone
[verify_recovered.py](../reports/engineering-archive/dense-v3-complete-fourth-stage-artifact-backup-v1/source/verify_recovered.py)
along with this guide. Its SHA-256 is
`a0050dd292e68df684812c2789f3588a27574228b25e2a6a3421727b52179cda`.
The reader uses only the Python standard library and the supplied snapshot:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/new-fourth-stage-download/corrected-dense-correctness-v3/complete-fourth-stage-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c \
  --manifest-sha256 21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c \
  --output /absolute/path/to/new-fourth-stage-reconstruction.json
```

It checks all file hashes, reads all 168 native primary scores, authenticates the
168 original successful workers and reconstructs twelve exact macro means and
180 CSV rows. Original absolute producer paths are interpreted only as provenance;
the reader never opens them, loads checkpoint tensors or executes a model.

The source/guide remain local WIP, not a published GitHub source release. Data-only
HF publication does not bypass that boundary. See the
[local evidence archive](../reports/engineering-archive/dense-v3-complete-fourth-stage-artifact-backup-v1/README.md)
for the actual transport/reconstruction receipts and the separate, completed
[full-trajectory numerical readout](../reports/dense-v3-complete-trajectories-v1/README.md).
The complete-trajectory tables/figures have not yet been added to this score-only
snapshot. Functional and crossed-continuation analysis remain separate requirements.
