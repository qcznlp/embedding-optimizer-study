# Result-independent bridge numerical amendment plan

Prepared after the two preserved synthetic counterexamples and before running the new solver.
Parent exact-geometry acceptance: `63076c3f...`; unchanged bridge: `347331a3...`.
No accepted v3 primary run, validation selection, retrieval grid or bridge outcome exists.

The nine features, eight-column baseline, 45/15 leave-dose folds, training-only feature transform,
unregularized OLS, pooled 60-row error and strict pooled-plus-three-fold support rule are retained.
This is an explicit numerical/undefined-domain amendment, not an old protocol hash refresh.

## Exact arithmetic for the fixed input numbers

Interpret each finite binary64 input as its exact rational value (`Fraction.from_float`), without
decimal rounding or denominator approximation. Solve the eight-column baseline using exact rational
linear algebra. The Frisch–Waugh–Lovell decomposition adds each feature using its exact baseline
residual. With an intercept, an invertible training-only affine feature transformation leaves these
unregularized predictions unchanged in real arithmetic. Record the original training mean/scale;
perform the algebra on unscaled rational inputs so the transform introduces no new fit roundoff.

Compare exact rational mean squared errors. Their ordering is identical to RMSE ordering, including
exact ties; square roots and decimal conversion are display-only. No RMSE epsilon is fitted to the
counterexample or retrieval data. An arbitrarily small exact difference is not practical or
statistical significance; retain its magnitude and the original limited descriptive claim boundary.

Exact feature residual zero identifies baseline redundancy without an arbitrary residual tolerance.
If the training baseline relation also holds on every held-out row, predictions equal the baseline
exactly, with zero improvement. Otherwise the augmented held-out prediction is unidentified: retain
the fold and mark its augmented prediction/error undefined, rather than inventing an extrapolation.

## Numerical input-resolution boundary

Exact arithmetic does not make nearly collinear measurements reliable. Separately inspect the
training-standardized augmented design with NumPy's documented default least-squares singular-value
cutoff: `eps64 * max(rows, columns) * leading_singular_value`. A nonzero exact added direction below
that resolution is labelled unresolved, not rounded into an exact zero or reported as supported.
Baseline full rank is required in every fold. Preserve the singular spectrum, cutoff and status.
This is a numerical convention, not a statistical confidence interval or a physical identifiability
proof. It is never adapted to an optimizer or outcome. Constant-feature folds use the exact span
rule above and need no division by zero.

Residual associations use exact baseline residuals and exact average-rank ties. A zero residual has
undefined correlation. A nonzero but numerically unresolved feature or outcome residual is also
reported undefined with a distinct reason. Preserve every feature/fold: any undefined held-out
augmented comparison makes the complete pooled comparison and usefulness flag undefined. Do not
average only the available folds or equate undefined with no effect.

## Independent acceptance

Retain both original null cases without relabelling their old outcomes. Test constant/redundant,
training-only collinearity, near-collinearity, genuine additional signal, exact ties, input scaling,
row permutations and invalid panels. Independently solve full augmented systems, not just the same
FWL routine, and cross-check well-conditioned predictions against a separate numerical solver.
The complete v3 adapter must admit all real primary/validation/BEIR/geometry parents before output,
recompute its input bundles and refuse missing, altered, rehashed or mislabelled evidence.

Authoritative API references checked during preparation:

- [Python rational arithmetic](https://docs.python.org/3/library/fractions.html): constructing from
  a float preserves that float's exact value; no `limit_denominator` is used here.
- [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html):
  the default rank cutoff is machine precision times the larger matrix dimension.
- [SciPy Pearson correlation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html):
  constant inputs have undefined correlation and near-constant inputs can suffer numerical error.

These sources document APIs; the mathematical design and new implementation remain our responsibility.
All implementation provenance stays outside the manuscript. This plan does not authorize deployment.
