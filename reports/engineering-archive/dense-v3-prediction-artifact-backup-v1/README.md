# Complete prediction-artifact backup and recovered-error reconstruction

The two latest actual scientific analysis reports now have one immutable
**data-only** HF snapshot. No source was published. Revision
`42689be00e1644aaf24721ecc239ec1c4d425a2a` adds only
`corrected-dense-correctness-v3/weight-retrieval-predictions/` beneath the existing
corrected namespace. All twenty other root entries, fourteen prior corrected
subtrees and the exact root attributes remain unchanged. No old remote file was
deleted or overwritten.

## Actual completion

- Upload returned zero; immutable commit created **2026-09-12 13:08:08 UTC**.
- Complete anonymous download returned zero at **13:09:04 UTC**:
  **47 files / 12,301,350 bytes**, every size/hash matched.
- A copied stdlib reader ran with isolated Python from a new directory and
  returned zero at **13:09:25 UTC**. Only recovered snapshot files were read.
- All 840 original + 5,040 exploratory predictions reproduce 392 exact fold
  error comparisons, 98 pooled comparisons and all 98 plotted values. Every
  original B0 prediction is retained. All 6,875 CSV rows match the typed tables.
- Eighteen bounded local tests pass, including missing/extra/corrupt inputs,
  malformed predictions/decisions, source/credential-bearing JSON, unsafe SVG,
  wrong upload modes, overwrites and old-subtree preservation controls. They
  are transport/reader tests, not additional scientific experiments.

The [restoration guide](../../../docs/predictor-analysis-restoration.md) gives the
immutable prefix, external manifest digest and portable reader command.
[snapshot/](snapshot/) contains the exact remotely stored bytes. Actual receipts
are in [actual/](actual/), command/terminal identities in [commands.json](commands.json).
Source versions are retained in [source/](source/); they remain local only.

## Scientific boundaries

`locked/` preserves the previously fixed nine-feature and separate five-feature
numerical analyses, not a newly preregistered study. `sensitivity/` remains
explicitly post-result exploratory analysis. Neither baseline selection nor
predictive flags are promoted into significance, mechanism or useful dimensions.
No OLS refit, new inference, model, GPU worker, old coordinator, manuscript change
or source release occurred in this backup. Numerical error reconstruction is a
check of the same observations, not a new experiment. No physical second-host
experiment is claimed.

## Preserved initial checks

Before staging, the copied transport rejected the final heatmap SVG's embedded
PNG images. The old transport is unchanged. This new adapter now accepts only
inline PNG image resources after base64, PNG chunk and checksum validation;
scripts, event handlers and external resources are still refused. The exact
original SVG bytes are retained. Initial source is `source/backup-initial.py`.

The new recovery reader initially compared differences of separately rounded
RMSE values, causing a check failure. A second display check used that same wrong
rounding order for plot deltas. The final independent reader reconstructs the
difference in Decimal precision before converting to binary64, and maps plots
from that difference, matching the preserved original numerical definitions.
Both earlier source versions and their failed tool exits remain recorded.
No original prediction, table, protocol or figure was modified to pass a check.

[verification.json](verification.json) seals this bounded backup evidence,
rehashes both original archives, all 112 primary source copies and eleven
protected inputs. The full NAACL/reproducibility objective is not complete;
functional recovery and formal crossed continuation remain pending.
