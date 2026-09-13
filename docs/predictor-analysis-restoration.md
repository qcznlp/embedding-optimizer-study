# Restore weight-to-retrieval predictions and sensitivity

This immutable data snapshot preserves both completed sixty-state analyses,
including all original and exploratory comparisons, exact predictions, plans,
figures and independent verification receipts. It does not contain source code.

## Exact target

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `42689be00e1644aaf24721ecc239ec1c4d425a2a`.
- Prefix: `corrected-dense-correctness-v3/weight-retrieval-predictions/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d`.
- Manifest: `artifact_manifest.json`, 11,238 bytes, SHA-256
  `52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d`.
- Complete snapshot: **47 files / 12,301,350 bytes**.

Use the external digest from this trusted handoff, not a digest supplied solely
by an untrusted downloaded manifest. Do not use the moving default branch or
download the mixed historical repository root.

## Preserve the two different analysis roles

`locked/` contains nine original and five separate exact-measurement features:
840 predictions, 56 folds and fourteen pooled comparisons. Its existing numerical
definitions are preserved; the included execution plan was written after retrieval
outcomes were available and is not a new preregistration.

`sensitivity/` is explicitly **post-result exploratory analysis**, not a protocol
amendment. All fourteen features appear under four recipe baselines and two
displacement-conditioned comparators: 5,040 predictions, 336 folds, 84 comparisons.
Only the final second figure display version is included; renaming its directory
does not change any file bytes. Its predictor gains depend on the comparator.
Neither analysis establishes significance, mediation or retrieval-useful dimensions.

Original archive verification receipts list larger local archives. They do not
assert that every referenced source file is included here. Historical absolute
paths are provenance, not executable instructions or reader dependencies.

## Download anonymously

The actual transport used `huggingface_hub` 1.28.0 and downloaded every file
individually at the immutable revision. This equivalent convenience snippet is
not a separate transport trial; use a new destination:

```python
from huggingface_hub import snapshot_download

prefix = (
    "corrected-dense-correctness-v3/weight-retrieval-predictions/"
    "4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/"
    "52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d"
)
snapshot_download(
    repo_id="qcz/embedding-optimizer-study-analysis-artifacts",
    repo_type="dataset",
    revision="42689be00e1644aaf24721ecc239ec1c4d425a2a",
    allow_patterns=[prefix + "/**"],
    local_dir="/absolute/path/to/new-prediction-download",
    token=False,
    max_workers=2,
)
```

## Independently reconstruct the recovered prediction errors

Copy the local [standalone reader](../reports/engineering-archive/dense-v3-prediction-artifact-backup-v1/source/verify_recovered.py)
and this guide off-host separately. Reader SHA-256:
`cda3c8859f1797044dfe172704ece602c063047edf91ec44490edd3e266806ff`.
It requires only Python's standard library, no project installation or model:

```bash
python -I -B /absolute/path/to/verify_recovered.py \
  --snapshot /absolute/path/to/new-prediction-download/corrected-dense-correctness-v3/weight-retrieval-predictions/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d \
  --manifest-sha256 52930a02e24ec8e0d622aa6dddcd6b379dd41f8a3ae7bd33b3263fa58fcd7b3d \
  --output /absolute/path/to/new-prediction-reconstruction.json
```

All files were actually downloaded anonymously at **2026-09-12 13:09:04 UTC**.
A byte-identical isolated reader returned zero at **13:09:25 UTC**, checking:

- 47 files and their original selected-archive bindings;
- 6,875 CSV rows against typed JSON and sixty joined checkpoint outcomes;
- exact rational errors from all 840 + 5,040 predictions, 392 folds and 98 pooled
  comparisons, including eight redundant self-addition folds;
- all 840 original-baseline predictions unchanged in the exploratory analysis;
- all 98 plotted comparisons and the empty all-four-baseline support list.

It refuses incomplete, extra, changed or symlinked payloads. It does not refit
OLS, rerun bootstrap sampling, evaluate a model, import producer source, access
the network or read historical absolute input paths. It numerically reconstructs
errors from stored predictions; the original independent full-matrix OLS receipts
are preserved separately, not newly executed during recovery.

Actual recovery occurred on the same physical host, not a second-host experiment.
For upstream data, see [weight analysis](weight-analysis-restoration.md),
[complete retrieval and raw scores](retrieval-trajectories-restoration.md) and
[model checkpoints](checkpoint-restoration.md). Public source release, functional
analysis, crossed continuation and the full paper remain incomplete.
The [backup archive](../reports/engineering-archive/dense-v3-prediction-artifact-backup-v1/README.md)
preserves actual calls, failed checks and unchanged-source evidence.
