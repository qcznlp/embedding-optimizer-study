# Complete exact-geometry consumer

This milestone prepares a separate, source-bound exact measurement layer. It does **not** replace
the frozen approximate geometry or nine-feature retrieval bridge, release formal training, accept
the old held matrix, or supply a manuscript finding. Parent acceptance is `03165391...`; the new
draft protocol is `b1335115...`. The previous approximation/probe-coupling diagnosis is complete and
is not repeated as the target of this work.

## Complete primary interface

`embed_optim.primary_v3_exact_geometry` first admits all twelve complete v3 primary runs and their
sixty scheduled checkpoints. Before that admission, it neither reads model tensors nor creates an
output namespace. It then covers all 88 hidden matrices, every retained stage and both displacement
kinds. Auxiliary embeddings, norms and biases remain outside this explicitly declared population.
The output cardinalities are:

| Output | Complete primary rows |
|---|---:|
| checkpoint exact geometry | 60 |
| run-pair exact subspace overlap | 660 |
| optimizer-pair exact subspace summary | 60 |
| per-matrix/stage/displacement exact geometry | 10,560 |

Stage-one saved segments and all cumulative displacements use the immutable pretrained base.
Later segments use the preceding retained checkpoint. These are saved displacements over intervals,
not per-step optimizer updates or path length. Computation uses the same FP32 saved-weight
differences as the parent, promoted losslessly to FP64. Complete NumPy SVD replaces randomized
sampling in this new layer only; every singular value is retained in the raw artifacts.

Stable rank uses full squared spectral energy divided by the leading singular value squared.
Full-spectrum entropy normalizes all singular values, not only a top-64 sketch. Neither metric
counts embedding dimensions that are useful for retrieval. Their normalized versions divide each
matrix's rank by its smaller dimension before parameter weighting.

## Defined and undefined measurements

The retained subspace rank is `min(16, smaller matrix dimension)`. Numerical signal rank uses
FP64 epsilon × larger matrix dimension × leading singular value. A supported truncated subspace
is considered unresolved at the boundary if `(sigma_r - sigma_(r+1))/sigma_1 <= 2e-8`.
This is a declared numerical-resolution convention, not proof of physically stable or useful
directions. Full-dimension subspaces have no external cutoff boundary.

Every matrix retains its complete raw spectrum and status. Exactly zero displacement has undefined
stable rank/entropy and no retained basis. Nonzero matrices with insufficient signal rank or an
unresolved boundary still have full-spectrum scalar metrics, but no arbitrary retained basis is
exported as a resolved principal subspace.

For exact rank/entropy aggregation, let `n_i` be the number of parameters in matrix `i`, and let
`M = sum(n_i for matrices with nonzero displacement)`. The reported mean is
`sum(n_i * metric_i) / M` over those matrices. A zero denominator produces an undefined mean.
The `exact_*_nonzero_parameters`, `exact_*_nonzero_parameter_fraction` and status-mass fields
are **matrix-population parameter mass for these averages**, not counts/fractions of individual
entries that changed. Individual changed entries remain the separate validated census. The
`*_parameter_weighted_nonzero` suffix also distinguishes these exact means from old zero-imputed
approximate-rank aggregates; do not silently identify the two estimands.

Pair comparisons retain every matrix in their coverage accounting. Matrices with a zero
displacement on either side are excluded under the original zero-subspace convention, with their
mass exposed. Unsupported/unresolved **nonzero** mass is separately reported. If any such unsafe
mass exists, the complete nonzero-population overlap is undefined; a resolved-only conditional
mean has its own explicit label. Optimizer summaries average every rate pair equally only when
all complete pair means are defined. They never silently average just the available pairs.

The fast exact overlap computes the squared Frobenius norm of the cross-Gram matrix divided by
retained rank, then averages the left/right spaces. It avoids forming ambient-size projectors.
Independent explicit-projector controls verify this identity. Within a single matrix/stage, one
SVD can serve both anchors only when the FP32 displacement arrays are exactly equal; a norm or
approximate similarity is never used as a cache key.

## Verification scope

The complete twelve-run × five-stage × 88-matrix positive fixture is synthetic, with tiny matrices
and simulated admission. It verifies orchestration and table cardinalities, not real primary
training. All **45 focused tests and 2,051 full-suite tests pass**, with no failures/errors/skips.
Tests cover every undefined state, denominator behavior, exact value-cache reuse, all-rate summary
rules, fast/explicit projector equivalence, and refusal of altered/rehashed numerical artifacts,
partial outputs, changed protocols and absent primary inputs.

The actual audit is limited to the three existing short natural-diagnostic trajectories and their
nine retained checkpoints, but covers **all 88 hidden matrices** and both displacement kinds, not
the twelve-matrix subset used by its parent. Its producer/reader compares every raw metric, status,
complete spectrum and retained basis to fresh checkpoint computation. A separate PyTorch full SVD
is the independent reference for nonzero spectra/projectors; direct norms and NumPy-weighted
aggregation also cross-check scalar summaries. A later fresh process repeats the complete reader
and table production and tests two actual rehashed corruptions. Actual outcomes and immutable
receipts are recorded by [validation.json](validation.json); positive numerical checks do not
constitute primary optimizer or functional-dimension findings.

The initial actual audit (`113815a5...`, 2026-09-06 21:15 UTC) passes all 1,584 matrix/kind records:
528 exact zeros and 1,056 nonzero complete spectra, with all nonzero retained subspaces resolved
under the declared convention. The maximum independent singular-value discrepancy normalized by
the leading value is about `3.94e-15`; maximum normalized projector discrepancy is `1.71e-12`.
Eight altered protocols and both actual absent-primary CLI actions reject. The resulting diagnostic
tables contain 9 checkpoint rows, 18 run-pair rows, 18 optimizer-pair rows and 1,584 matrix rows.

The fresh CPU replay (`910b4de6...`, 2026-09-06 21:25 UTC) reproduces all nine checkpoint readers,
spectra, bases and four tables. Both actual altered/rehashed artifacts (one zero-displacement scalar
and one nonzero basis) are refused. Preserve `/tmp/dense-v3-exact-geometry.CBVst9` and
`/tmp/dense-v3-exact-replay.7jjL3Y`; [commands.md](commands.md) records the executed commands.

For transparency, the following descriptive comparison joins **all three optimizer pairs and both
displacement kinds** at the final diagnostic stage. It averages all 88 hidden matrices, not the
previous four-matrix subset. Values come from the unchanged
[approximate table](../dense-v3-geometry-chain-v1/run_pair_subspace_overlap.csv) and new
[exact table](run_pair_exact_subspace_overlap.csv), joined by both run IDs, stage and anchor kind.

| Diagnostic pair | Displacement | Original approximate overlap | Exact overlap |
|---|---|---:|---:|
| AdamW–Muon | saved segment | 0.01214 | 0.00186 |
| AdamW–NorMuon | saved segment | 0.01247 | 0.00795 |
| Muon–NorMuon | saved segment | 0.65204 | 0.32895 |
| AdamW–Muon | cumulative | 0.02681 | 0.01728 |
| AdamW–NorMuon | cumulative | 0.02484 | 0.02173 |
| Muon–NorMuon | cumulative | 0.67426 | 0.41840 |

The approximation difference therefore also appears in these whole-hidden-population summaries.
This is a diagnostic measurement comparison, not a primary optimizer ranking or a new explanation
of retrieval quality. It does not authorize selecting whichever measurement favors a hypothesis.

All candidate/old numerical sources and manuscript bytes are preserved. The separate training
identity candidate still has eleven unwaived old source-contract failures; the isolated checkout's
green suite does not certify that different tree. No formal run, GPU worker, controller transition,
commit/push, HF mutation or protected-helper action is part of this milestone. Every implementation
incident remains engineering provenance and enters no manuscript section.

## New statistical correctness issue: a baseline-redundant feature

A separate source-fixed [read-only counterexample](bridge-null-feature.json) (`5c8d8e3f...`) probes
the unchanged bridge (`347331a3...`). It is not a bridge correction or an accepted new analysis.
Both fixed synthetic cases use all sixty run/stage rows and put the candidate feature exactly in
the baseline design's column space. Baseline and augmented designs both have rank eight. Explicit
closed-form coefficients reproduce the feature exactly; the second case also reproduces its
outcome exactly using binary-representable fractions. There is no new identified predictor.

The initial stage-valued case does **not** pass the old usefulness rule, but the old function
reports finite residual Pearson/Spearman associations despite mathematically zero feature residual.
In the separately fixed bounded case, every feature is `stage/8`, and the outcome is
`0.5 + stage/64 + muon_indicator/32 - normuon_indicator/64`. The old pooled baseline RMSE is
`4.1860953867932023e-16`; the augmented value is `2.749530325049722e-16`. The roughly
`1.44e-16` reduction plus three nominally improved folds triggers `predictively_useful=true`.
The residual Pearson and Spearman values are about `0.09767` and `0.07117`, although the feature
has no residual direction. All nine feature slots deliberately repeat the same column: these
are nine repeated flags for **one numerical pathology**, not nine independent discoveries.

No bridge source, solver, feature definition, support threshold or prior protocol was changed.
Before reusing the bridge on real primary outcomes, declare and independently test the numerical
identifiability/zero-residual policy. Preserve the old failure and include null, near-null and real
signal controls; do not tune a threshold to the observed retrieval outcome or this chosen example.

## Next integration

Exact-geometry preparation and its actual complete diagnostic replay are no longer missing.
First resolve the confirmed bridge null-feature boundary under an explicit amendment, then
integrate the v3 retrieval bridge and separately declared exact-measurement sensitivity, functional
dimension/publication consumers, routed factorial and reviewed assembled-source/runtime/release-parent
handoff. [Bridge interface notes](bridge-interface-notes.md) record the inspected integration points;
they are not an executed or frozen statistical amendment.
Original features, failed attempts, raw models and old ledgers remain preserved. Formal primary
training, the 840-cell retrieval grid and paper findings are still separate, uncompleted goals.
