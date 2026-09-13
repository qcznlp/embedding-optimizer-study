# Agent handoff instructions

## Read first

1. [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md): current completion and remaining work.
2. [PROJECT_STATUS.md](PROJECT_STATUS.md): results and authoritative evidence map.
3. [README.md](README.md) and [current paper](paper/current/README.md).
4. The frozen protocol, accepted receipts and historical instructions for the surface being changed.

The full preceding handoff is preserved byte-for-byte in
[the historical archive](reports/engineering-archive/dense-v3-portable-handoff-v1/before/AGENTS.md).
Its older liveness counts are history, not instructions to restart jobs.
All preservation, scientific acceptance, source-release gates, safety and authority limits remain binding.
Before changing a historical source/runtime surface, read its detailed archived instructions;
this short handoff does not waive or replace a frozen contract.

## Goal and scientific scope

Deliver a defensible NAACL paper and reproducible repository/model-analysis artifacts
for **DenseOn only**, comparing AdamW, Muon and NorMuon. No new LateOn or blog work.
The latest direct owner authority is: “你有权做一切事情，目标是尽快完成任务”.
Training-first priority was fulfilled; source/publication cleanup is now required.

The primary design remains the same revised deterministic 500K queries, seed 42,
one positive plus seven distinct fixed hard negatives, no in-batch/cross-device negatives,
8192 context, global batch 128, one epoch, four rates per optimizer and five stages.
Primary rates are not independent seeds. Keep every run/rate/stage/task and frozen selection rule.

The story is reached weights and trajectories -> functional representation utility ->
held-out-rate retrieval prediction, with a separately bounded crossed continuation.
Intrinsic spectral/row behavior, one-step proxies and engineering tests are not optimizer
quality or mechanism findings. No packing, padding or other implementation-error narrative
belongs anywhere in the paper, including the appendix. Preserve it only as engineering evidence.
Never put held/historical or synthetic outcomes into current paper results.
Do not hand-edit generated findings or erase pending markers to obtain acceptance.

The crossed design uses two fixed genuine 60% states, reset AdamW/Muon, a fixed initial
gradient-history calibration and 50K queries under three order seeds. These seeds do not
replicate source training. Main contrasts are averaged endpoints, not gains from each source.
MM-AA = state + operator; interaction is not a third additive term.
Initial probe matching does not match subsequent update norms or angular dynamics.
Common auxiliary continuation does not remove different primary source-recipe histories.
Marginal crossed intervals are not simultaneous coverage, equivalence or mediation.

Coordinate attribution deletes and renormalizes; it is not additive raw score attribution.
Native-coordinate differences and three sampled rotations do not establish arbitrary-basis
invariance or greater useful capacity. Average effects within task before splitting helpful/
degrading mass. Preserve frozen zero-denominator handling and all undefined coverage.
Full-spectrum entropy differs from truncated-spectrum entropy; retain both measurement branches.
Held-out-dose prediction is not held-out-task generalization or a causal explanation.
Post-result recipe controls and cosine-sensitivity analyses remain explicitly exploratory.

## Current state — 2026-09-13

All 24 scientific runs, 120 checkpoints, primary 840-task evaluation, baseline 14 tasks,
12 validation selections, weight/functional analyses, 168 continuation tasks, five-stage probes,
statistics, tracking and required result backups are complete. Original complete numerical/PDF
replay has passed. **No currently registered study job is live.**
First paired-backward diagnostic sessions 17406 / 663091 and 39489 / 33677f
are terminal failures before the second pass. Preserve their source/outputs:
/tmp/dense-v3-paired-backward.lNjcKc0t/RUNNING.md. The diagnostic wrongly required
model_accepts_loss_kwargs=false; original Trainer normalization also applies
when the actual item count is None. A corrected diagnostic observes that native
count explicitly. Its 17 helper/native-normalization tests pass; it does not
change original training math or relax any endpoint comparison.
The corrected paired entry is terminal: 32954 / c06ef0 and 85232 / 6ee966
exit zero, all eight rank exits zero. Read
/tmp/dense-v3-paired-backward-fixed.JXOJBril/RUNNING.md. Same weights, optimizer,
scheduler, actual inputs and first-input RNG are verified. All 536 local leaf
contributions per rank match across passes, but 123/134 post-DDP gradient tensors
differ in both cases. DDP bucket rebuilding changes from 0 to 1. This locates a
paired difference downstream of local differentiation; it is not yet proof that
the original long endpoint failure is fully explained or repaired. No optimizer
update or clipping was performed; do not poll/restart these completed sessions.
The new first-backward diagnostic is also terminal: sessions 50164 / 84aa35
and 69733 / 4a2137 both exit zero, all eight ranks zero. Read
/tmp/dense-v3-first-gradient.FDK2kcqs/RUNNING.md; do not poll or restart these handles.
Both genuine saves load model/optimizer/scheduler states bit-exactly on device,
and all four actual resumed microbatches match expected token features per rank.
All 134 post-DDP/pre-clip gradients agree across ranks. No clipping or optimizer
update was executed. Original uninterrupted gradients are not available here;
this does not establish exact endpoint resume or identify the previous divergence's cause.
The subsequent all-element CPU rank-mean comparison has relative L2 differences
2.32e-8 / 2.47e-8 and fitted scale approximately one; no tolerance is changed.
See reports/engineering-archive/dense-v3-first-gradient-boundary-v1/README.md.
The completed paired CPU comparison covers all 149,014,272 elements per case:
relative L2 differences 2.0473483980114333e-8 / 2.1862897063898537e-8;
maximum absolute difference 1.4901161193847656e-8 in both. Its output artifacts
are complete, but the original launch tool/session exit was lost and remains unknown.
See reports/engineering-archive/dense-v3-paired-backward-boundary-v1/README.md.
Next test an explicit restoration-only warm-reducer adapter, gating its first
real-update gradient against the authenticated warm reference before any update.
The original exact endpoint comparison remains required and unchanged.
Do not repeat these completed tasks as missing work or poll their terminal sessions.

The reviewed manuscript is in **paper/current**, with all twelve authenticated inputs.
Use the documented current-paper entry and a new explicit output directory.
The complete document checker is byte-identical to the accepted original component.
Actual Make and extracted-wheel builds pass; the latter uses 28 wheel-local study modules
and reproduces the reviewed PDF text. The 158-word abstract/eight-page main and all 13 pages
have been checked. Final current-paper focused integration has 86 passing tests.
The historical parent paper and default all/release targets remain separate and unchanged.
A document-only success is not full scientific-consumer integration or source release.

Both repaired GPU resume checks are terminal failures of exact endpoint equality:
sessions 39663 / 75985, actual exits 1 (8ceca3 / 1672c5). All eight ranks reach step 391
and exit zero, but both fresh comparison readers fail. All 134 model tensors differ in
each case, as do moments; scheduler, four-rank RNG and counters match. Cause is unresolved.
The first resumed loss log covers seven updates, not the uninterrupted ten.
Do not infer a backend cause, relax tolerance, change kernels, overwrite old outputs or
automatically retry. Fresh scientific checkpoints remain unchanged; diagnostic resumed
states do not replace them. The device-counter helper is not yet a general restore integration.

The CPU boundary diagnostic now passes: all three seeds/four logical ranks retain
the entire resumed index suffix; eight-worker seed-314159 checks agree. Both genuine
step-313 saves load all 134 model tensors and complete optimizer/scheduler states
bit-exactly on CPU using the original loaders. See
reports/engineering-archive/dense-v3-resume-cpu-boundaries-v1/README.md.
Do not repeat those checks as missing. The subsequent real GPU boundary above
now covers device loading/tokens and records leaf contributions plus actual
post-DDP gradients. No backend cause is proven.

The original full distribution audit now passes (actual build/audit exits zero);
see reports/engineering-archive/dense-v3-portable-input-roles-v1/README.md.
Historical ten/six/one-finding failures remain preserved. Recorded namespaces now come from
pinned genuine input audits; the changed reader exactly reproduces the accepted input result.
The relevant 43 local tests, 58 diagnostic-assembly calibration cases and one cold CLI pass.
The latter preserves and explicitly exercises all original producer-root refusals.
Never hide strings, remove negative controls or weaken the scanner. Successful packaging does not prove
every runtime entry works. Same-host relocation is not a physical second-host test.

## Next work

Primary numerical source is now integrated byte-for-byte: all 33 primary modules and
56 assembly bindings match the actual training snapshot; all twelve source identities
load with repository and training root both this checkout. The 271 relevant tests and
three actual one-tree/CLI controls pass. Read docs/current-training-source.md and
reports/engineering-archive/dense-v3-numerical-source-integration-v1/README.md.
Do not repeat the source copy as missing work. The historical candidate YAML is a
test/preparation fixture, not the current v3 data/run matrix.

Saved factorial checkpoint inspection now has a separate explicit-local-path entry:
docs/saved-factorial-checkpoints.md. Genuine AdamW/Muon step-313 wheel reads exactly
match the original native reader, with saved identities unchanged and old directories
refused. This is artifact inspection, not fresh-run source admission or exact GPU resume.
See reports/engineering-archive/dense-v3-portable-factorial-checkpoint-v1/README.md;
the original source/checkpoint validators and fresh-run rejection remain unchanged.

The already-completed full numerical/PDF replay now has a stable local closed bundle:
reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed.
Read docs/paper-results-reproduction.md; invoke its unchanged entry from closed/primary.
All 189 input files and eight generated current-paper inputs match their accepted bytes.
This reuses completed replay, not a new experiment or remote/source-release admission.
Its document is the original complete-result prose; paper/current builds the reviewed revision.

1. Integrate the completed scientific consumers and explicitly resolve factorial source
   compatibility. Its original 66-file parent still differs only in factorial_v3_inputs.py
   after the portable-role change. Preserve the original rejection and failed 152-case
   source-bound test evidence; do not replace old hashes or relabel the old checkpoints.
   Calibration/numerics now work in this checkout; this is not full factorial/release admission.
2. Retain the passing distribution/portability checks through source consolidation, including
   historical provenance and semantic negative controls. Do not rerun the resolved findings as missing.
3. Localize the first resume divergence using matched loaded-state/sample/gradient boundaries
   before another explicitly recorded GPU attempt; keep original exact comparison requirements.
4. Complete the reviewed assembled-source/runtime/release-parent transition, relevant tests,
   strict paper/publication gates and authorized source publication.

Use accepted native evidence and completed replay where reuse is allowed; do not counterfeit
an old guard passing, rewrite a failed binding, or endlessly rerun finished preparation.
The actual one-vs-two-identical-selection history amendment remains explicit.
Two final primary NorMuon OS exits are unobserved; do not invent zero exits from artifact success.
Other specifically unknown tool exits in the historical evidence remain unknown.

## Runtime and external authority — hard limits

Never inspect, read, edit, signal, stop, replace or otherwise touch **gpu.py or its processes**.
No broad process/GPU-process inspection, pgrep/ps scans or GPU process enumeration.
Only exact registered study identities may be inspected through their accepted readers.
The narrow historical handoff reader is scripts.audit_dense_natural_data.handoff().

The old stopped controller identities remain protected:
196647/start246327790, 870313/start257721545, 870864/start257733056.
Their four-step main prefix and zero-step waiting factorial are not fresh dispatches.
Original main ledger SHA: 95c69483f60badddd3c13e175951bcb52682523589c62bf898a57eb786d4e4ef.
Never resume/kill them casually, launch competing controllers or apply a zero-step migration
to the nonzero main prefix. Read archived exact contracts/locations before any proposed transition.

Both original GPU lease namespaces remain mandatory for admitted GPU work.
Children inherit both lease FDs; closing a parent must not explicitly LOCK_UN a surviving child.
Do not force-unlock, infer a free slot from a stale observation or change frozen live source.
An expired observation is not process termination; use the same handle or exact owned identity.

GitHub issue-comment/update writes returned 403. Do not retry those writes, bypass the denial
or switch credentials. Read-only account admin/push metadata does not prove application write
scope. Git push has not been attempted. One existing-integration/Issues-access question remains
pending; do not repeat it. No remote source availability follows from local edits.
Preselected answers and automatic goal continuations are not new authority.
Commit/push, controller transition and WIP publication retain their review/authority boundaries.
The final pre-commit/release gates still require complete results, relevant tests/audits,
no pending macros and no Type 3 fonts. Do not self-waive them.

HF withdrawal was safety-rejected for 9,629 paths / 366,252,912,201 logical bytes.
Nothing was deleted. Never retry, split or bypass that denial. Historical erasure would require
specific approval, exact scope, shared-object/current-backup dependency checks and preserved
current evidence. Existing LateOn/shared artifacts remain protected. Never expose credentials.

## Protocol and work discipline

Read configs/dense_scope_amendment.json and the relevant dense_primary_v3 protocols.
For weights, retain original/exact geometry and bridge contracts and all their candidates.
For dimensions, read configs/dense_dimension_utilization_protocol.json and
docs/dimension-utilization.md. For factorial, read its scientific/implementation/publication/
completion locks, claim wording/handoff amendments and docs/state-operator-factorial.md.
For publication, read paper/README.md and docs/completion-gates.md, retaining historical gates.
Old locks do not automatically admit v3 identities; replacing hashes is not integration.

Treat this file's directory as the repository, not a fixed producer location. On this host use
/usr/bin/python with an explicit absolute repository src/repository PYTHONPATH for every test
and child; the installed package/live .venv can otherwise import an unvalidated source tree.
Use explicit location roles for other trees; exact historical locations and source hashes are
preserved in the linked handoff and native receipts, not portable runtime defaults.
Keep existing WIP and frozen source copies. Use apply_patch for authored edits and rg for search.
Do not commit unrelated changes or mutate sealed archives. Use new output paths and preserve
all failed/unknown-exit evidence. CURRENT_PROGRESS.json is legacy accounting, not a heartbeat
or scientific verdict; refresh only through its original explicit experiment/output context.
No subagents are requested; continue locally unless the owner explicitly asks.

Report concise Chinese updates. Distinguish actual science, diagnostics, portability and release.
Classify each goal turn by real progress or verified waiting and keep the full goal active until
all required paper/repository/reproducibility deliverables are actually verified.
