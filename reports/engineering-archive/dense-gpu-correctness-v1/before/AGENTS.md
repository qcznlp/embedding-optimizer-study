# Agent handoff instructions

## Full-model NorMuon check — 2026-09-05 18:35 UTC

The new [full-model reference probe](reports/engineering-archive/normuon-full-reference-v1/README.md)
narrows the epsilon finding: **352/352 hidden-matrix update comparisons are bitwise equal** to
the authenticated official functions on one real DenseOn checkpoint and the fixed first two
materialized training rows. All 88 hidden matrices are covered with saved/fresh state and loss
multipliers 1/0.25. Actual input lengths are 11–611 tokens; no truncation or model update occurs.
Both scale conditions activate clipping, so this is not evidence that quarter-scaling is harmless.

The small-matrix 44/96 nonconformance result remains valid, but **its impact on actual primary
training is not established**. All measured BF16 denominators in this full-model probe are equal;
do not generalize that to all gradients. The diagnostic reads a digest-authenticated independent
download, preserves every source payload and all model/optimizer state, and uses CPU only.

The duplicate-normalization failure and scientific hold remain decisive. No GPU handoff approval,
pause, deployment, primary retraining or HF withdrawal has occurred. Existing evaluation ownership
and the invalidity of the old zero-step migration remain unchanged. This is engineering provenance
only, not manuscript content or an optimizer-quality result.

## Code verification update — 2026-09-05 18:06 UTC

**The code is not cleared for scientific use.** Training has now finished **12/12 runs**, with
**60/60 resumable and backed-up checkpoints**. Main has completed its first four post-training
steps and is evaluating BEIR; factorial still waits for main. The older running/10-run counts
below are superseded by this update and the fresh artifact-only snapshot.

The [new code-correctness audit](reports/engineering-archive/dense-code-correctness-v1/README.md)
retains the decisive duplicate-normalization failure and adds an independently authenticated
NorMuon reference check: 44/96 small-matrix update cases fail conformance because local code
uses a clamped norm while upstream adds epsilon. The actual primary-training impact is unknown.
The existing isolated normalization candidate has not been deployed.

Twenty-four four-rank CPU checkpoint continuations (both Trainer variants, three optimizers,
dropout 0/0.1, two restart points) restore initial weights, optimizer, scheduler, consumed rows
and per-rank RNG exactly, using a disclosed CPU-only deserialization mapping adapter. Subsequent
calculation is not bitwise identical; one live NorMuon case has a row-second-moment difference
above the prior gradient-audit tolerance. The full 1,281-test regression suite and 25 new guard
tests pass, but neither overrides these failed/incomplete numerical acceptance checks.

The owner was asked whether to pause automatic evaluation for GPU correctness checks; no reply
or pause has occurred. No production source, checkpoint or controller was modified. Keep the
scientific hold; do not launch GPU checks into the active evaluator. The old seven-file main
migration requires an inactive **zero-step** ledger: those old preconditions no longer hold.
It must be revalidated for the completed-step/evaluation state after any authorized handoff.
Source publication, runtime deployment, retraining and HF withdrawal remain unapproved.


## Repair candidate verified on CPU — 2026-09-05 07:10 UTC

The isolated `SingleNormalizationTrainerCandidate` now passes the actual four-rank CPU audit
on BOTH the tiny fixture and a digest-authenticated full DenseOn checkpoint: all three optimizers,
three steps each, global groups 128/128/32 and all 134 full-model parameter gradients before/after
clipping. Trainer keeps accumulation 4/4/1; Accelerator's second divisor is one. Actual scheduler
cadence and cross-rank final weights also pass. A deliberately wrong constant tail divisor fails
only on the partial group. Read `reports/engineering-archive/dense-normalization-candidate-v1/README.md`
and its receipts. The full-model source checkpoint is unchanged; diagnostic optimizers start fresh.

This does NOT fix any running job or certify GPU/BF16/max-length/resume/scientific outcomes.
The five core training files remain byte-identical to live main; the candidate is not imported by
the launcher. No owner approval, pause, deployment, re-training, source publication or HF withdrawal
has occurred. Keep the scientific hold below and independent backup in effect; existing controllers
are still running and do not enforce that documentation hold. The latest artifact snapshot is
07:05 UTC: 10/12 finished and 54/60 preserved, steps 2208/2095. Exact W&B handles at 07:07 UTC
remain running at 2220/2110. Do not launch a duplicate controller or touch `gpu.py`.

## Critical new finding — 2026-09-05 06:10 UTC

The actual four-rank CPU Trainer audit **fails gradient normalization**: raw gradients are
0.25000001 times the independent global-batch reference because Trainer and Accelerator both divide
by four. An explicitly compensated CPU-only control passes all six checks across AdamW/Muon/NorMuon;
no production code was fixed. Read `reports/engineering-archive/dense-ddp-accumulation-v1/README.md`
before acting. Earlier payload/CPU-loss/unit-test passes do not certify this previously untested path.
Do not promote the current matrix into paper findings or initiate new evaluation/factorial work
until this issue is resolved. A pause of the final two training jobs and automatic post-processing
has been requested from the owner, but has NOT been authorized or performed. Existing controllers
do not enforce this documentation notice. Do not infer pause/deployment/retraining/deletion authority
from an automatic goal continuation; keep independent checkpoint backup running. Training still
executes on the unchanged source; no checkpoint is discarded. The previous seven-file post-processing
repair does not address this new issue. Nothing about it belongs in the manuscript.

Execution counts at 06:10 UTC are 10/12 completed runs and 54/60 resumable/covered checkpoints.
Both exact NorMuon handles remain running (1740/1630); artifact steps are 1744/1634 of 3907.
These are execution/preservation counts, not a verdict that the experiment is scientifically valid.

## Goal and authoritative read order

The active goal is to determine whether AdamW, Muon, or NorMuon gives better optimization and
retrieval outcomes when adapting DenseOn on the same deterministic 500k-query, seven-negative
training set. The current phase is the 12-run independently padded corrective replication. LateOn
is no longer active work.

Read these sources in order before acting:

1. `PROJECT_STATUS.md` — canonical human-readable state, conclusions, blockers, and next steps.
2. `CURRENT_PROGRESS.json` — latest committed machine-readable training snapshot.
3. `README.md` — public study overview and reproducibility entry point.
4. `configs/dense_scope_amendment.json` — active Dense-only scope.
5. The frozen protocol relevant to the task. For the current phase, read
   `configs/dense_no_packing_execution_protocol.json`, its preflight parent, and
   `configs/dense_no_packing_evaluation_protocol.json`. Weight-space and retrieval-bridge work
   must additionally follow `configs/dense_no_packing_analysis_protocol.json` and the source-bound
   bridge implementation in `configs/dense_no_packing_bridge_implementation_protocol_v2.json`;
   corrected outcome aggregation must follow `configs/dense_no_packing_outcome_protocol.json`, and
   historical/corrected comparisons must follow
   `configs/dense_no_packing_sensitivity_implementation_protocol.json`. Final paper rendering
   must follow `configs/dense_no_packing_publication_protocol.json`. Functional use of embedding
   dimensions follows `configs/dense_dimension_utilization_protocol.json` and
   `docs/dimension-utilization.md`; native-coordinate claims must retain the shared-rotation
   boundary.
6. For the prospective state-by-operator mechanism follow-up, read both
   `configs/dense_no_packing_state_operator_factorial_protocol.json` and
   `configs/dense_no_packing_state_operator_factorial_implementation_protocol.json`, plus its
   publication and completion locks, the result-blind abstract-compliance migration and the
   narrower `configs/dense_no_packing_state_operator_claim_wording_amendment.json`, then follow
   `docs/state-operator-factorial.md`. Do not execute it from changed source bytes or before the main
   corrected completion ledger is fully complete.
7. `docs/dense-no-packing-retrain.md` — exact corrected-matrix operational commands.

If `logs/dense-no-packing-v1/recovery-supervisor-state.json` exists, read it after
`CURRENT_PROGRESS.json`. It is the atomic control-plane state for the recovery defined in
`configs/dense_no_packing_control_plane_recovery.json`; do not launch a competing matrix while its
phase is `waiting_for_adopted_training` or `matrix_running`.

The legacy recovery state has no code-level heartbeat while its blocking matrix child runs. The
artifact-only next-pair guard is frozen in
`configs/dense_no_packing_matrix_handoff_guard.json` and publishes
`logs/dense-no-packing-handoff-guard/state.json`. It waits for both current AdamW runs to become
deeply complete, then gives the existing matrix five minutes to create a declared Muon successor
log or output. It yields if any successor artifact appears and invokes the unchanged recovery
supervisor only after the full absence grace and a final race check. Do not launch a second guard
while its lease is held, and never replace its artifact gates with process inspection.

If `logs/dense-no-packing-finalization/pipeline-ledger.json` exists, it is the atomic handoff state
for incremental corrected checkpoint backup and the post-training evaluation/analysis/publication
chain. Do not launch a competing corrected finalizer while its controller lease is held. Resume it
only with `python -m embed_optim.corrected_completion_pipeline --resume`; the exact operational
source/protocol/command contract must still match.

The owner-directed paper-only amendment caused the pre-amendment controller to fail closed at
2026-09-04 00:07 UTC. Its sole authorized contract transition is frozen in
`configs/dense_no_packing_completion_contract_migration.json`. If the ledger still has the exact
source contract named there and the controller lease is free, run
`python -m embed_optim.completion_contract_migration` once, then resume the controller normally.
The migration archives the original ledger byte-for-byte and verifies that the matrix, execution
protocol, controller, arguments, and command order did not change. Never generalize this path to
accept arbitrary contract drift.

The first resumed audit exposed a narrower operational bug: audit-only full-run verification erased
the stored upload commit identity. The hardening transition is frozen separately in
`configs/dense_no_packing_backup_provenance_migration.json`; it preserves the original upload
commit during audits and adds `corrected_checkpoint_backup.py` to the controller's own contract.
If the live ledger still has that protocol's exact source hash and the controller lease is free,
run `python -m embed_optim.completion_backup_contract_migration`, then resume normally. This second
one-time transition must not be substituted for any future source change.

A later result-blind manuscript topology preflight found that placing every corrected table in the
main paper pushed its endpoint to page 9. The exact non-scientific transition is frozen in
`configs/dense_no_packing_publication_layout_migration.json`: the all-rate retrieval answer remains
in the main narrative, while the complete geometry-bridge and execution-sensitivity tables are
invoked after the appendix boundary. After this change reaches the experiment checkout and the
controller releases its lease, migrate with
`python -m embed_optim.completion_contract_migration --protocol
configs/dense_no_packing_publication_layout_migration.json`, then resume the ordinary controller.
Do not use that migration for any other source drift.

A second result-blind narrative audit found that the frozen corrected renderer would update its
results section but not the abstract or main Conclusion. The exact publication-only transition is
frozen in `configs/dense_no_packing_publication_narrative_migration.json`: the same primary
all-rate finding is rendered into the abstract and main Conclusion, while the full historical
packed-selector claim moves to the appendix. After this change reaches the experiment checkout and
the controller releases its lease, migrate with
`/usr/bin/python3 -m embed_optim.completion_contract_migration --protocol
configs/dense_no_packing_publication_narrative_migration.json`, then resume the ordinary controller
with `/usr/bin/python3`. This migration was executed at 03:12 UTC on 2026-09-04; its distributable
receipt is `reports/dense-no-packing/publication-narrative-migration.json`. Do not rerun it or use it
for any other source drift.

If `logs/dense-no-packing-sealed-backup/state.json` exists, it is the independent per-checkpoint
durability state. Its supervisor only reads training artifacts and uses CPU/network resources. Do
not launch a duplicate while `logs/dense-no-packing-sealed-backup/supervisor.lease` is held. A
covered checkpoint means a hash-audited remote backup exists; it never means the run or study is
scientifically complete.

On the experiment host, `CURRENT_PROGRESS.json` may lag the logs. Refresh it only through the
artifact-only command below; it does not inspect system processes:

```bash
python -m embed_optim.corrected_progress --output CURRENT_PROGRESS.json
```

## Handoff discipline

At 05:30 UTC on September 5, the primary matrix has ten complete runs, 52 resumable/covered
checkpoints and ten complete automatic whole-run backups. NorMuon 1e-4 and 3e-4 are finished;
1e-3 and 3e-3 have started, with latest artifact steps 1413 and 1303. Both exact W&B handles are
freshly confirmed running (1410/1300 at 05:30 UTC). No run remains unstarted. Each final run now has
an independently sealed/uploaded checkpoint 782; their uploader receipts verify 34 files total.
The matrix performs its own pool handoff. Do not launch a competing matrix. The new independent
ten-run audit verifies all 50 completed stages and 1,010 uploaded files at original immutable commits.

Raw geometry now covers all ten completed runs and 50 stages. The unchanged frozen CPU command
materialized both completed NorMuon runs under the existing partial-analysis runbook. The separate
`primary-geometry-ten-run-archive.json` verifies all 60 geometry files, 4,400 matrix rows and 102
model/metadata inputs. The additions-only HF commit `489c6060...` adds 12 missing NorMuon files
and preserves all 48 earlier geometry files. Anonymous checks verified the ten underlying NorMuon
weights at their original public commits; no old file/card/history was modified. This is not
permission to bypass the separate erroneous-result deletion rejection. Raw geometry does not
establish retrieval, functional-dimension, full subspace or factorial findings.

The current `PROJECT_STATUS.md` now concentrates the goal, actual evidence, approval boundaries
and continuation order. Its exact 830-line predecessor is preserved under `reports/handoff-history/`.
Older milestone counts and superseded proposals below are historical receipts, not the current
handoff. No safety or approval rule was relaxed by this documentation refactor.

The new `scripts/audit_checkpoint_download.py` verifies a mandatory trusted HF-audit hash and every
downloaded file before opening model/optimizer payloads. Its real NorMuon 3e-4 checkpoint-3126
receipt validates 18 independently downloaded files / 1,351,464,827 bytes, offline CPU encoding,
and exact optimizer/scheduler continuation of all 134 parameters after serialization. It reads
no live checkpoint and writes only a new diagnostic receipt. This is fresh-directory restoration
on the same physical host, not four-GPU resume, data-loader/rank-RNG replay or scientific evidence.
Follow `docs/checkpoint-restoration.md`, including the isolated-source availability boundary.
That restoration milestone passed 1,150 tests; older source-bound receipts remain unchanged.
Do not reintroduce whole-HF-repository default downloads that mix historical and current artifacts.

The new scientific claim audit under `reports/paper-review/factorial-claims-v1/` reproduces three
synthetic counterexamples to the current continuation renderer's prose. Positive interaction does
not establish beneficial co-adaptation; two positive average main effects need not be additive;
one supported effect alongside an inconclusive other effect does not establish dominance.
The original candidate and its 34-test receipt remain immutable. The narrower interpretation is
NOW integrated in this isolated checkout; current validation is under
`reports/paper-review/factorial-claims-v2/`. The exact fourth publication-only layer reconstructs
the previous three layers against archived predecessor bytes, preserves main target `4380a363...`
and all 47 commands, and passes all five consumer loaders. Its isolated/host projections are
`41fdf48c...` / `7aafd7af...`, not the old three-layer values below. The final suite passes 1,202
tests, including 41 claim/figure checks and a complete synthetic-paper build. The aligned figure
has explicit box-overflow checks; the protective development generator contains no invented scores.
The real pending draft ends main text on page 6; strict scientific release still fails on missing
primary/dimension publications and pending includes. No runtime/source deployment occurred, and no
primary or factorial model finding was created. The original scientific/implementation protocols,
old receipts and all runtime approval boundaries remain unchanged.

The latest prepared main dependency is the seven-file recovery target `4380a363...`, not the prior
six-file `af646ecf...`. The exact transition and boundaries are documented in
`reports/engineering-archive/main-resume-v1/README.md`. Its separate, source-bound migration script
has read-only default behavior, requires an inactive waiting zero-step ledger and a free existing
lease to apply, and preserves the original ledger and all backups. Twenty-five temporary-copy
checks pass; no real migration, source deployment or controller takeover occurred. Do not use the
old JSON-only publication migration for this Python-controller change.

The preceding three-layer successor added
`configs/dense_no_packing_state_operator_main_recovery_amendment.json`. This third layer changes
only the required main hash, an amendment binding and timestamp. The prior protocol, auditor and
tests are archived under `reports/engineering-archive/successor-resume-v1/before/`. All three layers,
five actual consumer loaders and all 47 unchanged commands validate. The local/host projections
are `ca0c7f5b...` / `6749baa1...`, not live `6605090d...`; 81 focused and 1,124 full tests pass.
The actual dry run reports main incomplete; no scientific output is created. Run the projector as
`python -m scripts.audit_successor_contract --repository <isolated-root>` from the checkout, with
its explicit `src` in `PYTHONPATH`. Older two-layer receipts and target hashes below remain history,
not the current prepared handoff. Do not merge the isolated branch into a running main finalizer.

The owner-requested experiment integrity audit lives under `reports/experiment-integrity/`.
`primary-audit-ten-run.json` freshly validates the complete stored data linkage and ten primary
runs/50 checkpoints; `huggingface-digest-audit-ten-run.json` independently checks all 1,010 uploaded
files at the original immutable commits. The earlier eight-run receipts remain unchanged. All
explicitly have `scientific_completion=false`. Do not promote
them into whole-study completion or a retrieval result. Rerunnable read-only commands are in
`scripts/audit_primary_experiment.py` and `scripts/audit_hf_run_digests.py`; use the audited checkout's
absolute `src` path in `PYTHONPATH`. Exact source bytes for the original audit are archived with its
receipt. The new dimension checks require the actual fixed probe identities and exact matrix cells,
not only matching row counts.

The later `checkpoint-cpu-resume.json` exercises three real primary checkpoint-2345 states with
`scripts/audit_checkpoint_optimizer_resume.py`. All 134 parameters and optimizer states per model
reload and continue exactly after in-memory serialization under fixed synthetic gradients; the
learning-rate schedule also matches. This is a two-thread CPU engineering check, not a four-GPU
training resume, data-loader/rank-RNG restoration, CPU/GPU equivalence or scientific optimizer
finding. Do not use it as a weight-space mechanism result or put it in the manuscript. The original
checkpoint files and all live controllers are unchanged; the exact offline command is in the audit
README. Its 15 focused tests supplement the prior 1,040-test whole-project suite.

The subsequent `dense-loss-contract-real.json` adds actual CPU forward/backward and MTEB-wrapper
checks for the same three checkpoints. It verifies production collator/trainer extraction and
query/document token prefixes, eight explicit candidates, the loss and all 134 trainable gradient
tensors against a separate float64 row-wise reference, and encoding/cosine agreement under forced
small length-budget chunks. All checkpoint inventories remain unchanged. It uses two short
synthetic queries, FP32 and evaluation mode with autograd: do not promote it to training-mode
dropout, distributed accumulation, BF16/FlashAttention, maximum-length execution or scientific
optimizer evidence. The new suite passes 12 focused checks and all 1,081 full-project tests without
exclusions. The run command and diagnostic-only first-attempt failure are documented in the audit
README. Neither numerical training code nor live controllers were changed.

The primary dimension export entry point is `python -m embed_optim.primary_dimension_probe` under
`configs/dense_primary_dimension_export_protocol.json`. Its dry run is read-only. Execution requires
all 12 primary runs/60 stages to be deeply complete, then takes one cooperative GPU lease. Do not
substitute historical probe exports. The corrected analyzer now requires its exact
`primary_exports.json` handoff; the publisher re-computes all inference and exact LaTeX during audit.
Operational commands are documented in `docs/dimension-utilization.md`. The source-bound
`configs/dense_primary_dimension_handoff_protocol.json` inserts eleven primary/dimension steps before
factorial calibration, preserving all 36 original steps (47 total). The controller's `--dry-run`
takes no lease and executes no work. This augmented contract has not been deployed; archive the
old still-zero-step ledger under the declared migration before restarting only that controller.
The full-source authoring audit creates a portable dimension-publication closure; the final paper
gate then recomputes statistics and exact LaTeX from it. This portable check does not rerun encoding
or coordinate ablations, and it does not certify training. Real primary execution, remote analysis
upload and the runtime transition remain required. The source-bound `dimension_archive` command
stages only the declared portable closure plus all 61 raw vector arrays, atomically uploads to a
content-addressed HF dataset prefix, and audits the returned immutable commit. Its `--audit-only`
retains the original verified commit; its `--verify-download` requires the trusted receipt hash.
Actual primary archival has not happened. Do not imply that tests are scientific results.

For reconstruction on another host, `python -B -m embed_optim.dimension_replay` accepts a verified
download root, the trusted receipt SHA-256 and a new output directory outside that archive. Use
`PYTHONDONTWRITEBYTECODE=1` when importing the archived source. It reuses the numerical kernel but
keeps scope `reconstruction_dense_dimension_utilization`; never promote its output into the primary
authoring scope. Its source/version checks, exact row identities and frozen numeric tolerances must
not be relaxed to pass a mismatch. The original 768D historical tables/arrays match a complete
post-refactor recomputation exactly, and the pre-refactor source is archived by SHA-256. That is a
method-equivalence check, not a primary optimizer result. Actual primary archival and replay remain
pending until the real matrix and analysis are complete.

Keep `PROJECT_STATUS.md` current whenever a meaningful run, failure, release gate, backup, or
scientific interpretation changes. At the same boundary, refresh `CURRENT_PROGRESS.json`, update
GitHub issue #41, and commit and push the documentation/evidence change once its checks pass. Do
not commit a new snapshot for every training step; the JSON is a durable handoff receipt, while the
command above is the live view. Never put claims based only on a running or partially written
checkpoint into the paper.

The active corrective phase is governed by `configs/dense_no_packing_execution_protocol.json`;
its engineering parent is `configs/dense_no_packing_preflight_protocol.json`, and corrected
checkpoint reloads are governed by `configs/dense_no_packing_evaluation_protocol.json`. The
weight-space operationalization and retrieval bridge are governed by
`configs/dense_no_packing_analysis_protocol.json`; the later executable bridge source binding is
governed by `configs/dense_no_packing_bridge_implementation_protocol_v2.json`. The v1 bridge
implementation lock is a superseded, never-executed receipt and must not be used. Validation
selection, max-T inference, and retrieval dynamics are governed by
`configs/dense_no_packing_outcome_protocol.json`; execution-path sensitivity is governed by
`configs/dense_no_packing_sensitivity_implementation_protocol.json`. Generate the complete
corrected paper tables only through `python -m embed_optim.corrected_publication`, which
verifies all four upstream manifests and the source-bound publication protocol. Do not hand edit
its generated paper include. Do not launch or interpret corrected runs from
an uncommitted matrix.

Use `python -m embed_optim.corrected_wandb_audit --allow-partial` for a read-only audit of active
corrected source runs. Omit the partial flag only after 12/12 training completion. This check may
verify W&B identity, configuration, and state, but it must not mutate source histories or supply a
scientific result.

The owner's last-run/idle-pool evaluation instruction requires a single-owner operational handoff.
The current main controller launches validation on all eight GPUs at 12/12 completion; its evaluator
does not share a GPU lease with a manual BEIR subset. Do not start an early subset while that
controller is still active. Follow the coordination boundary in `docs/dense-no-packing-retrain.md`;
the last training job's completion must not race an eight-GPU finalizer against early workers.
BEIR `--dry-run` writes metadata despite launching no evaluator: readiness checks use a new
engineering-only result directory, never the primary result root.

The active paper scope is DenseOn only. LateOn files are historical provenance and must not be
promoted into primary inference or used to justify new computation.

The completion controller uploads a run after all five scheduled checkpoints are deeply complete.
If a machine-shutdown risk requires earlier durability, use
`python -m embed_optim.incremental_checkpoint_backup` only on an already sealed scheduled
checkpoint. Its receipt must report `scientific_completion=false`, verify the local payload is
stable, and compare Hugging Face LFS SHA-256 or Git-blob SHA-1 digests after upload. This operation
preserves a resumable state; it must never promote a partial run into a completed result.
For unattended coverage use `python -m embed_optim.sealed_checkpoint_supervisor`; it reuses the
same sealed-checkpoint uploader, fails closed on invalid receipts or a changed source contract,
and yields the final checkpoint to an active whole-run backup before applying its own fallback.

The state-by-operator follow-up has its own source-bound handoff at
`python -m embed_optim.state_operator_factorial_completion --resume`. Its ledger is under
`logs/state-operator-factorial/completion/`. It must wait for the exact completed main corrected
ledger, then runs the frozen calibration, six training waves, checkpoint backup, probe/BEIR
evaluation, summary, paper-only renderer, and release gates. Do not launch any factorial command in
parallel with that controller while its lease is held.

The result-blind abstract-compliance transition is recorded in
`configs/dense_no_packing_state_operator_abstract_compliance_migration.json`. It preserves every
scientific choice while enforcing the ACLPUB 200-word abstract maximum after both result macros
expand. A pre-transition factorial ledger with zero executed steps must be archived byte-for-byte
before resuming the controller against the refreshed source contract; never overwrite that ledger
or treat its waiting state as scientific output. The completed host transition is recorded in
`reports/state-operator-factorial/abstract-compliance-runtime-migration.json`; the active contract
starts with `6605090d` and remains at zero steps until the exact main corrected ledger completes.

The owner-directed scientific-story rewrite is prepared on branch
`narrative/weight-space-spine`. It removes all implementation-debugging material and invalid
exploratory outcomes from the manuscript, makes the primary 12-run matrix the only optimizer
evidence, and organizes the paper around reached weight states, dimension utility, out-of-dose
retrieval prediction, and the crossed reset factorial. Its exact zero-step factorial transition is
recorded in `configs/dense_no_packing_state_operator_weight_space_narrative_migration.json`.
Do not merge this branch into the live experiment checkout while the main corrected completion
controller is active. Before merge, refresh every source binding and test the migration; after the
main controller completes, archive the still-zero-step factorial ledger and restart only that
controller under the new contract.

Publication preflight has now found three malformed table-header newlines in the live main renderer;
the two scientific tables retained by this rewrite are fixed here. The read-only proof is
`reports/experiment-integrity/live-publication-syntax-diagnosis-v2.json`. Do not assume the current
`4152531e...` main contract can finish its PDF release unchanged. A separate two-file syntax-only
patch and a proposed exact migration are archived beside that receipt; they are drafts, not deployed
or runtime-transition-tested. A later nine-test offline rehearsal is recorded in
`reports/experiment-integrity/live-publication-header-rehearsal.json`: it tests the exact patch and
temporary-copy migration, preserves existing backup records, and verifies the successor dependency
must change. It does not acquire a real controller lease or change any live source or ledger.
Resolve that narrow publication transition and the factorial's exact
main-contract dependency before deployment. Do not merge the entire narrative branch into a running
main controller to address this defect. Training, retrieval and numerical analysis do not need to
change for the table-header repair.

A subsequent real evaluation dry-run/consumer audit identifies another live finalization blocker:
`evaluation_source_provenance.py` rejects the corrected evaluator's exact ten-source mapping, so
`corrected_outcome_summary -> aggregate.collect_evaluations` cannot consume it. The isolated helper
now accepts that exact topology while retaining full reachable-Git byte/hash checks. See
`reports/experiment-integrity/evaluation-handoff-validation.json`: real 40-checkpoint metadata
passes the repaired reader, 65 focused tests and all 1,052 full-suite tests pass, but no BEIR scoring
or complete outcome summary was run. The helper was not named in the existing frozen source lists;
bind it explicitly in the controlled transition before deployment. Do not use the previous
header-only proposal as if it also resolves this consumer defect, or silently change live source
just because the current controller contract omits that transitive dependency. No live migration
or controller takeover has occurred. These are engineering-only findings, never manuscript content.

The combined repair is now archived under `reports/engineering-archive/main-handoff-v1/` with its
exact six-file patch, original sources/ledger and candidate files. Forty focused checks pass,
including real protocol-loader checks and a temporary-copy migration preserving all eight backups.
The main target is `af646ecf...`, not the older header-only `6d991052...`. The isolated successor's
follow-up dependency amendment now requires that combined target and refreshes the narrative,
dimension and factorial-publication dependencies. A read-only patch applicability check passes on live main,
but nothing was applied and no real lease was acquired. The preparer cannot deploy or signal jobs.
Do not relabel the old header-only migration or successor receipt as this new combined transition.

The active isolated dependency is now
`configs/dense_no_packing_state_operator_handoff_repairs_amendment.json`. Its seven byte-identical
predecessor protocols are archived under `reports/engineering-archive/successor-handoff-v1/`.
The first header-only amendment and its old receipts remain historical, not the current dependency.
`scripts/audit_successor_contract.py` reconstructs both preparation layers, exercises all five actual
consumer protocol loaders, projects exactly four checkout-local path arguments and preserves all
47 prepared commands, including the original 36 factorial commands and `/usr/bin/python3` arguments.
See `reports/experiment-integrity/combined-successor-contract-projection.json`: isolated contract
`a64b2658...` is not projected host contract `37a4485c...`, and neither is live `6605090d...`.
All 1,071 full-suite tests and the 56-test focused selection pass; the first focused attempt exposed a stale factorial-publication
amendment binding and is retained as a failed receipt. No scientific or numerical source changed.
Recompute the contract on the actual host after an approved deployment. Leave old main/factorial
controllers untouched until the narrow main transition, complete main gate and controlled zero-step
successor handoff are verified; this preflight takes no lease and executes no migration.

HF withdrawal is owner-requested but execution is currently awaiting exact-scope approval after a
safety-review rejection. See `reports/hf-obsolete-cleanup/README.md`: the reviewed plan has 9,629
files / 366,252,912,201 logical bytes. No HF artifact or history has been deleted, and neither has
any local evidence. Do not retry or split the rejected deletion to bypass confirmation. Current
replication backups, shared data and separately retained LateOn files are protected. Historic commit
removal additionally requires preserving and re-auditing current payloads and migrating any pinned
backup dependencies; ordinary HEAD deletion is not historical erasure.
The later `history-dependencies.json` proves shared content objects and pins the then-existing 49
primary backup receipts / 39 commits. Some shared objects are current RNG/scheduler payloads and a
retained untrained baseline export; never purge LFS objects solely by a withdrawn path prefix.
That audit is read-only, bounded to named current trees and primary receipt directories, and already
predates another checkpoint upload. Refresh it before an approved cleanup; it is not a history-purge
authorization or an inventory of all historical objects.

The paper has a strict scientific-content boundary: do not place packing, padding, execution-path
incidents, candidate-breadth debugging, or other implementation-error narratives anywhere in the
manuscript, including its appendix. Those records remain repository engineering provenance only.
Do not reintroduce historical exploratory numbers as headline evidence. The final paper may import
only the primary matrix and its audited weight-space, dimension, retrieval, and factorial outputs.

Never inspect, read, edit, signal, stop, replace, or otherwise touch `gpu.py` or its processes. It
is outside this repository and automatically yields to study jobs.

Preserve all existing checkpoints and evidence. Use new output namespaces for corrected reruns;
never overwrite the 34 completed historical Dense runs. Treat protocol thresholds and failed gates
as results, not knobs to relax. In particular, the candidate-breadth width-7 reproduction failure
and `reports/candidate-breadth/packing_invariance.json` must remain disclosed.

Before committing, run the tests and release/audit commands appropriate to the changed surface,
check `git diff --check`, and verify that the manuscript has no pending result macros or Type 3
fonts. Do not push generated evidence or change GitHub pull-request state until its source-bound
audits pass. Never print or commit credentials.

For a fast repository handoff check, run `python scripts/portable_evidence.py --audit-only`, then the
strict Dense paper audit documented in `README.md`. The portable closure is the clean-clone evidence
boundary; the public Hugging Face checkpoint archive is required for full model-state
reconstruction. Never relax either audit to turn a failure into a pass.
