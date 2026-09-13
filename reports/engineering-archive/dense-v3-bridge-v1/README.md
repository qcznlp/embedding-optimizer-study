# V3 bridge numerical correction and complete input adapter

This milestone prepares the missing v3 geometry-to-retrieval bridge and corrects the previously
confirmed null-feature boundary under a **separate draft amendment**. It does not execute a formal
primary analysis or clear the held twelve-run matrix. All positive panels here are synthetic.
The old source (`347331a3...`), old protocol (`88fb013b...`), two original counterexamples and
parent exact-geometry acceptance (`63076c3f...`) remain unchanged. New protocol: `540c3c30...`.

## What changed, and what did not

The same nine feature values, eight-column optimizer/stage/log-LR baseline, four 45/15 leave-dose
folds, one-feature-at-a-time unregularized OLS and strict pooled-plus-three-fold support rule remain.
All twelve rates and all five retained stages must be present. No score selects a row or feature.
The [result-independent numerical plan](numerical-amendment-plan.md) records the changes before the
new solver was run; no error epsilon is tuned to an observed outcome.

Every input binary64 value is represented as an exact rational number. An exact baseline solve
and Frisch–Waugh–Lovell decomposition calculate the augmented OLS prediction. Because the baseline
contains an intercept, an invertible training-only affine feature transformation changes neither
the real-arithmetic fitted subspace nor its predictions. Its exact mean/variance and display scale
are retained; the fitting algebra introduces no rounded standardization step.

Support compares exact rational MSEs, which have the same ordering as RMSEs. Decimal square roots
and displayed RMSE differences cannot create a positive decision from a mathematical tie. This
does **not** turn tiny genuine differences into practical significance, statistical confidence or
causal evidence. Magnitudes and exact error records remain visible.

| Added feature condition | Held-out comparison | Residual association |
|---|---|---|
| Exactly in baseline span, same relation holds on held-out rows | Exactly the baseline prediction; zero gain | Undefined for zero feature residual |
| Exactly in training baseline span, relation fails on held-out rows | Unidentified; no arbitrary extrapolation | Separately checked on full panel |
| Nonzero but below declared numerical rank resolution | Undefined, not rounded into an exact zero | Undefined with resolution reason |
| Resolved additional direction | Exact OLS prediction and exact error comparison | Defined only if both residual directions are resolved |

Resolution uses the documented NumPy default least-squares cutoff on the training-standardized
augmented design: FP64 epsilon × larger design dimension × leading singular value. Exact arithmetic
does not remove near-collinearity or measurement sensitivity; this cutoff is a declared numerical
convention, not a physical or statistical identifiability theorem. Every singular spectrum/cutoff
is retained. One undefined fold makes the full pooled augmented comparison and usefulness flag
undefined. Available-only pooling is prohibited; all folds and rows remain in the outputs.

Residualization is exact, zero residuals have undefined correlation, and average-rank ties are
computed on the exact residual values. Both preserved null cases now have exactly zero pooled and
fold improvement, false usefulness flags, and undefined Pearson/Spearman residual associations.
The old bounded case's nine repeated false-positive flags are still **one duplicated-feature
pathology**, not nine independent findings. The old function itself is not modified or deployed.

## Complete primary interface

`embed_optim.primary_v3_bridge` admits all complete v3 primary runs before opening score bundles or
creating outputs. It obtains freshly recomputed validation/outcome evidence through the unchanged
v3 outcome consumer, then recomputes all raw geometry and its tables through the unchanged geometry
consumer. It validates the actual bundles, joins all 60 rows with all 660 run-pair comparisons and
rechecks raw/table/source bindings after the join. Both actual CLI actions refuse missing primary
runs before output. No diagnostic checkpoint is relabelled as a primary run.

The original 60 input rows, 36 fold rows, nine summaries and nine association rows are retained.
A 540-row held-out prediction table additionally exposes every row's baseline/augmented prediction
and exact rational values. Numerical diagnostics and all identities are preserved in the evidence
bundle. Its fresh reader recomputes expected bytes instead of trusting refreshed file hashes.

## Verification and preserved failed attempt

All **54 final focused / 2,105 full-suite tests pass**, with zero failures/errors/skips. Earlier
28-case and 52-case positive receipts are also retained. Tests include null, constant, real signal,
training-only and near-collinearity boundaries, exact ties, lossless rescaling/sign changes, row
permutation, independent full-system solves, malformed panels, complete adapter wiring and altered
or rehashed output rejection. The wiring positive explicitly simulates upstream model admission;
it is not an actual primary evidence positive. The separate numerical training candidate's eleven
old source-contract failures remain unwaived; this checkout's green suite cannot clear them.

The first independent audit exits 1 in its **reference-only** rank helper: SymPy Boolean atoms do
not support `int()` conversion in this runtime. Source, unchanged protocol, partial fixture outputs
and [failure receipt](attempt-1/failure.json) are retained. Conditional counting plus three exact
rank/tie controls correct that reference helper only. No analysis source, protocol or tolerance
changes between failed and successful audit attempts.

The successful [rehearsal](rehearsal.json) (`3fbfd13a...`, 22:04 UTC) independently solves all 36
full augmented systems using SymPy, not the production rational FWL implementation. Every one of
**540 held-out predictions**, 36 fold MSE comparisons and nine pooled decisions matches exactly.
All eighteen Pearson/Spearman values also match independent full-system residual calculations.
The [fresh process replay](fresh-replay.json) (`c4316000...`) repeats all complete tables, exact
values, numerical diagnostics and both corrected counterexamples. Each process rejects four
altered/rehashed bundles and eight altered protocols; both actual absent-primary CLI actions refuse.

Preserve these CPU-only namespaces:

- `/tmp/dense-bridge-numerics.DE6nMV`: all test receipts;
- `/tmp/dense-v3-bridge-audit.6xH3YX`: failed reference audit and partial outputs;
- `/tmp/dense-v3-bridge-retry.kWkgar`: successful independent audit and complete synthetic bundle;
- `/tmp/dense-v3-bridge-replay.osgmcN`: fresh replay and independently rejected corruptions.

[commands.md](commands.md) provides reproduction commands. [validation.json](validation.json)
authenticates the final archive, parent preservation and current source/ledger/manuscript boundaries.
No formal run, GPU worker, controller transition, commit/push, HF mutation or protected-helper action
occurred. No part of this engineering provenance belongs in any manuscript section.

## Next work

The original-feature v3 bridge and its numerical correction are no longer missing preparation.
Add the separately declared exact-measurement sensitivity, retaining the old approximate feature
family and its limitations. Full-spectrum entropy and nonzero-matrix denominators are different
estimands and must keep explicit names/coverage. Integrate functional-dimension/export/publication
consumers and their analogous numerical boundaries, then routed factorial and reviewed assembled-
source/runtime/release-parent handoff. Full valid primary training, the 840-cell BEIR grid, actual
mechanism findings and a releasable NAACL paper remain uncompleted.
