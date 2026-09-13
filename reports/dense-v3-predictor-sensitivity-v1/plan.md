# Post-result predictor sensitivity: recipe effects and conditional information

This is an explicitly **post-result exploratory analysis**, planned after the
locked 9+5 feature results were observed. It does not amend or replace those
protocols, outputs, predictive flags or endpoint conclusions. No new model,
retrieval, functional intervention, seed or causal branch is run.

Questions:

1. Does a displacement feature improve held-dose prediction when the comparator
   can represent nonlinear rate effects and optimizer-specific learning curves?
2. Does each other geometric measurement add predictive information after
   cumulative displacement is already in the model? Entropy is not privileged;
   retain every one of the fourteen existing feature columns.

Fixed diagnostic baselines, in this order:

- B0: the original eight columns, unchanged.
- B1: B0 plus a raw-learning-rate column for each optimizer (11 columns). Raw
  rate is nonlinear in the original log-rate regressor. Scale by fixed units
  1e-5 for AdamW and 1e-3 for Muon/NorMuon; this changes units, not model span.
- B2: fifteen optimizer-by-stage indicators, plus one centered log-rate and one
  raw-rate slope for each optimizer (21 columns). This permits different
  optimizer learning curves and different nonlinear rate responses.
- B3: B2 plus raw rate times nominal cumulative learning-rate schedule mass for
  each optimizer (24 columns). The nominal schedule has 3907 steps, 391 warmup
  steps, linear warmup/decay, step-zero multiplier zero, and mass normalized to
  one at the final checkpoint. It is a recipe-only covariate, not a measured
  update norm, optimizer path, decay intervention or matched-compute experiment.

Run all fourteen one-feature additions for B0–B3 (56 comparisons). Additionally,
for B0 and B3 only, include cumulative displacement in the comparator, then add
each of the same fourteen features (28 comparisons). Its self-addition is a
declared redundant control, not a new hypothesis. This makes **84 complete
comparisons, 336 folds and 5040 feature-specific held-out prediction rows**.
No baseline or feature is selected for reporting on the basis of its result.

Use the original four leave-dose-index-out folds (45 training / 15 test rows),
all twelve runs and five stages. Keep the original outcome and feature binary64
values. New recipe covariates are deterministic binary64 transforms of recipe
metadata and known stage. Exact rational OLS is evaluated at those values;
no ridge, model fitting by the test score, arbitrary pseudoinverse, column
dropping or clipping of predicted nDCG is introduced.

The original eight-column FWL implementation is not modified. A new standalone
generic-width exact implementation handles these additional baselines. Training
feature standardization is affine and eliminated algebraically in an intercept
span. For numerical resolution use eps64 * max(n,p) * largest singular value;
retain diagnostics. Exact redundant added columns predict equivalently only
when the relation extends to every held-out row. A nonzero unresolved direction
or unidentified extension is undefined, never zero-imputed. Any undefined fold
propagates to an undefined pooled augmented comparison.

For comparability retain lower pooled exact MSE plus at least three improved
folds as a **descriptive diagnostic flag**, not significance or confirmatory
support. Verify B0 against every original/exact result and use a separate full
augmented-matrix rational solve for the actual new fits. Preserve all outputs,
negative results, source versions and any failed attempts, then replay from
copied source/data. Record baseline-only performance as well as incremental
comparisons: lower feature gains may reflect a better comparator, and an
overparameterized comparator can itself extrapolate poorly.

The richest comparator has 24 parameters from 45 training rows. Four rates do
not establish its generalization; endpoints extrapolate from only three rates.
New covariates and conditional comparisons were chosen after seeing the first
results, so even stability across these choices is not independent confirmation.
No current-generation statistic alone identifies a causal or functional mechanism.

Only new local CPU analysis files and a dated handoff are in scope. Original
training, numerical sources, protocols, analysis archives, manuscript, remote
state, protected helpers and failed/stopped coordinators remain unchanged.
