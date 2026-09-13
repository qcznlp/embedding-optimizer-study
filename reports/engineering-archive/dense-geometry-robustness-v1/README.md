# Exact geometry controls and random-probe coupling

This is **engineering/measurement evidence on short diagnostic trajectories**, not a primary
optimizer result or manuscript material. It preserves the complete v3 geometry acceptance
`053866a2...` and does not alter any frozen primary feature, training code, model, ledger or paper.

## What was tested

The fixed diagnostic protocol `e950a43d...` inherits all twelve previous controls: four named
attention/MLP matrices in block zero, all three diagnostic optimizers, checkpoint 3, cumulative
displacement from the same pretrained base. These are not a random sample of all layers and do
not cover the twelve full-horizon primary recipes. The prior observed spectral error motivated
this work; it is not a blind confirmatory scientific test.

The exact controls use the identical FP32 displacement, losslessly promoted to FP64. Both NumPy
and PyTorch compute a full SVD and reconstruct the original matrix. They agree on the complete
singular spectrum and full-spectrum metrics. Across four ranks per matrix, the largest observed
normalized projector discrepancy between the two references is approximately `2.99e-13`.

For each matrix, all thirteen declared sketches are evaluated: ranks 8/16/32/64 with each of
three RNG seeds and two power iterations, plus rank 16 with eight power iterations at the original
seed. Original rank-16/seed/power-2 bases match the saved v3 bases exactly. The audit contains:

- 12 exact spectra and 48 independent rank-specific projector controls;
- 156 approximate-subspace accuracy measurements;
- 144 same-matrix, different-seed comparisons;
- 156 cross-optimizer pair comparisons, with exact counterparts.

Full-spectrum stable rank and singular-value entropy are explicitly distinct from truncated-sketch
metrics. Projector overlap is `tr(PQ)/r`, equivalently mean squared principal-angle cosine. A value
of one means identical subspaces; it is not an embedding cosine or retrieval score. FP64 QR only
normalizes the comparison representation of approximately orthonormal saved bases; it preserves
their column spaces and never changes the original artifacts. We retain boundary gaps and energy
capture instead of inferring unique directions from supported numerical rank alone.

## What the actual diagnostic controls show

Ranges below span the four inherited matrices and both left/right spaces, using the original
rank-16, two-power-iteration sketch. They are descriptive, not confidence intervals.

| Diagnostic optimizer | Overlap with exact top-16 subspace | Relative top-16 energy shortfall | Same-matrix, different-seed overlap |
|---|---:|---:|---:|
| AdamW | 0.898–0.945 | 0.003–0.015 | 0.815–0.920 |
| Muon | 0.133–0.169 | 0.122–0.147 | 0.061–0.091 |
| NorMuon | 0.131–0.266 | 0.123–0.177 | 0.066–0.142 |

The original stable-rank error remains about +9–12% for the selected Muon/NorMuon matrices, versus
less than `8e-7` absolute relative error for AdamW. Eight power iterations improve exact-projector
agreement to 0.997–1.000 for AdamW, 0.382–0.426 for Muon, and 0.374–0.620 for NorMuon. **No blanket
approximation-accuracy pass is declared.** Numerical implementation agreement is not adequate
accuracy for a mechanism claim. All individual observations are retained in [details.json](details.json).

### Additional shared-probe diagnostic

After observing the projector error, a separately source-fixed, explicitly post-hoc follow-up
evaluated **every 3×3 seed pairing** for each optimizer pair and inherited matrix: 108 rows, with
three same-seed and six different-seed observations in each of twelve cells. No primary protocol
or feature was rewritten. All 36 same-seed values and twelve exact controls reproduce the parent
audit. A new process recomputes every pairing and summary.

The complete Muon/NorMuon comparison is:

| Block-zero matrix | Exact top-16 overlap | Same-seed sketch mean | Different-seed sketch mean |
|---|---:|---:|---:|
| attention Wo | 0.809 | 0.909 | 0.077 |
| attention Wqkv | 0.299 | 0.663 | 0.063 |
| MLP Wi | 0.300 | 0.630 | 0.068 |
| MLP Wo | 0.788 | 0.920 | 0.074 |

Thus sharing a random probe materially affects the approximate similarity measurement in these
controls. Switching to different seeds is **not** an accuracy repair: those estimates also differ
substantially from the exact answer. Do not interpret either result as evidence about retrieval
quality, useful embedding dimensions, a primary optimizer ranking, or full-model geometry.
All other optimizer pairs are also retained in [coupling-audit.json](coupling-audit.json).

## Verification, preserved failures and provenance

The real audit `ca916d91...` and fresh replay `c048e330...` independently recompute all measurements
and all 348 saved spectrum/basis arrays from the retained checkpoint inputs. The replay also
rejects four altered/rehashed numerical artifacts: exact metrics, approximate metrics, pair metrics
and basis arrays. The separate probe-pairing audit `e1821101...` and replay `45db3e85...` agree.

The first pre-protocol unit run is retained: 41 cases, 40 passes and one failure from requiring
exact real-arithmetic equality after decimal scaling. A one-ULP intermediate floating-point
roundoff was incorrectly treated as a mathematical mismatch. The numerical kernel was not changed
to fit that test. The revised test uses an exact rational reference for the actual binary inputs,
with an explicit one-ULP rounding allowance, and adds three power-of-two controls requiring exact
equality. Both the failed test source/XML and the amended source remain archived. This test
expectation revision happened before protocol preparation or real-matrix execution; no empirical
acceptance tolerance was changed after observing results. A preceding development lint typo was
fixed before these tests and never executed on model inputs.

The final focused checks are 44 geometry plus seven coupling cases. The pre-follow-up whole suite
passes 1,999 cases; the final **2,006 cases pass with zero failures/errors/skips**, as recorded in
[full-tests-final.xml](full-tests-final.xml).
The separate numerical identity candidate's eleven old source-contract failures remain unwaived;
tests of this isolated checkout do not certify that different source tree.

The final [validation receipt](validation.json) binds preserved parent evidence, source copies,
exact input/output bytes, both fresh replays, test receipts and unchanged numerical cores,
manuscript and stopped dispatch chain. Complete diagnostic numerical verification is **not**
primary experiment acceptance. All model updates and GPU-worker counts in these audits are zero.

Retain these existing directories; do not overwrite them to rerun a command:

- `/tmp/dense-geometry-robustness.DNQw2T` — exact audit, full raw details and saved arrays;
- `/tmp/dense-geometry-robustness-replay.E7a4yz` — fresh recomputation and altered-file controls;
- `/tmp/dense-probe-coupling.tth6XL` — all crossed seed pairings and fresh replay.

The earlier geometry/reference/natural-diagnostic directories remain required parents. No formal
run, controller transition, commit/push, network upload, HF deletion or protected-helper action
occurred. The old twelve runs / sixty checkpoints remain on scientific hold. Engineering incidents
belong in no manuscript section. See [commands.md](commands.md) for exact invocation boundaries.

## Next action

The bounded exact-spectrum/projector and shared-probe diagnosis is complete; do not repeat it as
missing work or replace it with more same-kernel agreement tests. Prepare an explicit prospective
exact/error-controlled geometry consumer for valid primary checkpoints, retaining the old
approximate columns under their correct names. Do not silently change the nine frozen bridge
features or claim that merely selecting different RNG seeds makes inaccurate sketches exact.
Then integrate the v3 retrieval bridge, functional dimension/publication consumers, separately
routed factorial and reviewed assembled-source/runtime/release-parent transition. Functional
dimension utility still requires embedding-level interventions and retrieval evidence, not just
weight rank or projector differences. No primary scientific conclusion has been produced here.
