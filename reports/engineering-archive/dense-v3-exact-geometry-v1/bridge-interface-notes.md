# Inspected bridge integration points — not an executed amendment

The main agent read the complete original analysis/bridge protocols and executable bridge during
the exact-geometry audit. No bridge source, feature, outcome, fitting rule or existing protocol was
changed. This note records useful integration details rather than treating an unexecuted plan as
completed work.

**Subsequent read-only check found a blocking correctness issue.** The preserved
[null-feature counterexample](bridge-null-feature.json) (`5c8d8e3f...`) gives a feature exactly in
the baseline span. In its binary-exact bounded case, the unchanged bridge labels a roughly
`1.44e-16` RMSE rounding difference predictively useful; residual associations are also finite
when the theoretical feature residual is zero. The initial floating-outcome case remains separately
recorded and does not trigger the usefulness flag. Do not blindly reuse `evaluate_bridge_features`:
first declare and verify an explicit numerical/identifiability amendment. No repair has been made.

`corrected_retrieval_bridge.py` (`347331a3...`) has reusable pure functions:

- `assemble_bridge_rows` requires the full 60-row geometry/outcome panel and all 660 run pairs;
- `evaluate_bridge_features` fits all nine original features individually under the frozen baseline,
  four leave-dose-index-out folds, training-only feature standardization, OLS and support rule;
- `_adamw_overlap_index` requires every finite pair and averages all four AdamW rates, or the other
  three rates for an AdamW focal run. Undefined pair values are rejected, not imputed.

The v3 `RecipeView` already exposes the exact fields this pure assembler needs. The bridge's old
CLI cannot read the v3 bundles: it expects old logical manifest keys, scopes and parent hashes.
Add a new adapter with complete input admission/recomputation; do not relabel old manifests or feed
diagnostic data through a simulated primary identity.

The v3 outcome path is `collect_evidence -> inspect_bundle`; its run-stage scores derive from the
complete admitted validation and 840-cell grid. Both approximate and exact geometry have their own
all-twelve-run producer/readers and distinct table names. Recheck raw output bindings after joins.
No BEIR or validation score should decide which geometry row, rate, checkpoint or feature survives.

A prospective exact-measurement sensitivity must be explicit before its valid-primary outcomes are
observed. Retain the original nine-feature analysis unchanged. Exact stable-rank fractions and
exact AdamW-overlap counterparts can be distinctly named additions. **Full-spectrum entropy is a
different estimand from top-64 renormalized entropy**, not simply a more accurate computation of
the old feature. Nonzero-matrix weighting also differs when zero matrices occur; retain denominator
coverage and refuse undefined complete features rather than selecting convenient rows.

If mirroring a feature family, predeclare all counterparts and report them all; do not switch to
whichever approximation or exact feature predicts retrieval better. Original per-fold and pooled
scores, associations and support thresholds remain fixed. Any new family or output cardinality
requires a separate source-bound measurement/statistical amendment, not a refresh of old hashes.

This bridge can assess prediction beyond optimizer, stage and within-optimizer log learning rate
in a one-seed fixed grid. It cannot identify mediation or prove a universal optimizer mechanism.
Functional dimension utility still requires embedding-level interventions and retrieval evidence;
weight spectrum, subspace overlap and predictive association alone do not establish it.
