# Remaining numerical closure after vector features

These are inspected interfaces and constraints, not implemented/accepted reconstruction.
The vector reader must not be advertised as completing the following independent requirements.

## Original outcomes

The existing authoring selector already retains the exact declared BEIR task JSON, sibling
`model_meta.json` / `run_settings.jsonl`, and per-job `primary_admission.json` under the `beir` role.
`primary_v3_io.evaluation_plan` defines `cache_key = digest({protocol, checkpoint})`; its
`results_root` is host-bound. Preserve that original string as evidence, but address archived
content by the validated cache key and relative paths, without opening the old location.

`PrimaryV3Contract.checkpoint` is a v3 override, not the base contract method: it wraps the
sealed stage row with `scope = dense_primary_correctness_v3`, the v3 protocol hash and run ID.
The whole-run `checkpoints` list contains the unwrapped sealed rows. Reconstruct the exact v3
wrapper before deriving the BEIR cache key; neither a bare stage row nor the older base scope
has the same identity. Validation deliberately uses the unwrapped final stage in its own plan.

Use `primary_completion.task_score` on each local task file to reproduce its pinned revision,
split/subset/runtime/model checks and nDCG. Preserve the intentionally unused auxiliary-NaN policy.
Require every 12 × 5 × 14 cell exactly once, and compare full admitted run metadata to the
vector admission's same run population before combining representation and retrieval results.
Do not infer a relative path from filename or digest alone when multiple jobs can share files.

The `validation` role retains four files per selected job. Reconstruct each path-neutral plan
from the actual validation contract, same final checkpoint, frozen settings and archived
`metadata.validation_identity`. Match its `content` to the v3 validation dataset inventory and
the complete 4,096 ordered row identities. This authenticates prior data identities, not raw text.
Then call `primary_v3_validation_io.inspect_saved`: it recomputes every saved candidate-score
record and summary. Rebuild all twelve `run_metrics` and apply the unchanged `select_recipes`.
No BEIR score may enter selection; do not use stored `selected` values as a substitute.

`primary_v3_outcomes.outcome_tables` recomputes all original inferential tables from the full
score grid and freshly reconstructed selection. `system_rows` consumes originally admitted run
timing/payload metadata; it does not remeasure runtime or verify absent checkpoint bytes.
Compare all original outcome tables exactly, preserving original provenance fields separately
from the new local file addresses.

## Original geometry and bridges

The `geometry` role includes all per-stage raw metric records and retained bases, not just
aggregate CSVs. Use each frozen stage descriptor plus the original checked run/reference metadata,
recompute `primary_geometry_kernels.checkpoint_row` and entry denominators, then reconstruct
`primary_v3_geometry.tables_from_verified_runs` from the local bases. Authenticate and validate
the recorded subspace-health population separately. Its projected numerical rank, singular-value
boundary gap and retained Frobenius energy are computed from displacement matrices by
`audited_top_bases`; bases alone cannot reproduce those measurements. A portable aggregate
reader must not label their retained values as freshly measured. Repeating the spectrum and
weight-to-metric computation needs the model archive, not just these analysis-role files.

Reconstruct all original nine-feature `primary_v3_bridge.bridge_tables` from those geometry
rows and newly reconstructed original outcome stage scores. Preserve exact identifiability,
all doses/stages and undefined coverage. Do not replace the original features with only the
new functional predictors or the separate exact-spectrum sensitivity family.

Finally, call the unchanged functional `dimension_inference.summarize` and renderer with the
raw-reconstructed feature tables and reconstructed original bridge rows. Compare all nine
simultaneous contrasts, 27 rotation contrasts, four named predictors, 68 plotted points and
rendered evidence. This final consumer must retain the frozen support/null/undefined boundaries.

Only after the entire closure is independently replayed may it become a candidate for the
separate real-primary publication and release gates. All current parents remain drafts and
all actual primary scientific results remain missing.
