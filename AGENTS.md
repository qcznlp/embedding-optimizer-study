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

Latest owner workflow direction: do not create GitHub PRs or make intermediate
pushes. Finish the complete local process and relevant verification first, then
publish the finished change set in one push. Do not use hosted CI as an iterative
debugging loop. This supersedes older instructions to dispatch replacement CI
immediately after each repair; it does not waive scientific or release checks.

Deliver a defensible NAACL paper and reproducible repository/model-analysis artifacts
for **DenseOn only**, comparing AdamW, Muon and NorMuon. No new LateOn or blog work.
The latest direct owner authority is: “你有权做一切事情，目标是尽快完成任务”.
Training-first priority and core source/paper publication are fulfilled. Hosted
CI still needs verified repair; separately safety-denied historical HF cleanup
remains unresolved.

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

## Verified completion — 2026-09-13

The complete source payload is public on GitHub main at
**803c70dcc3a2c195c5f56ee5fc0e3c633d645524**. Normal Git push completed and the remote
head was verified; anonymous reads of ten source/handoff/manuscript/manifest files
match local bytes. This status update follows that already-published payload.
See reports/engineering-archive/dense-v3-source-publication-v1/README.md.

- Science: all 24 runs, 120 checkpoints, 840 primary and 168 continuation retrieval
  units, baseline/validation, weight/functional analyses, inference and tracking complete.
- Durability: all current scientific checkpoints backed up; required analysis
  groups anonymously downloaded and verified. See the restoration guides.
- Paper: all twelve reviewed inputs in paper/current; 158-word abstract,
  eight-page main and all 13 pages checked. No engineering-error narrative.
- Reproduction: actual original full numerical graph plus reviewed-paper build
  passes, including eight byte-identical shared inputs. Default/all now build
  paper/current; release uses explicit NUMERICAL_BUNDLE and new RELEASE_OUTPUT.
  Original Make rules remain byte-identical in paper/legacy.Makefile.
- Local tests: all 3,800 cases pass in explicit source roles (2,873 current, 733 original
  analysis, 194 original factorial), with no failures/errors/skips. The current
  role was fully rerun after five legacy Make-routing fixes. All failed attempts
  remain preserved; no frozen contract, numerical assertion or tolerance changed.
- Distribution: wheel/sdist, original full audit, portable evidence, Ruff and
  isolated CFF validation pass. Candidate/index bytes and Git history scanned;
  all flagged wheel RECORD substrings independently verified as actual SHA-256.
- Recovery: two genuine AdamW/Muon 313-to-391 restores pass the unchanged complete
  bitwise endpoint comparison. The separate scoped restore package also passes
  CPU/wheel integration. This does not cover every save, NorMuon or a second host.

The detailed chronological handoff remains in the published payload and
reports/engineering-archive/dense-v3-release-transition-v1/actual/before/AGENTS.md.
Original source locks, failed and unknown exits, exact device-recovery evidence
and all safety limits remain binding; do not repeat completed work as missing.

## Next action and boundaries

No scientific or recovery job remains live. Finish the local CI repair and its
complete verification, then publish once under the owner's workflow above.
Do not add experiments or restart completed controllers. The last verified public
main is b0fa533fceee53906e7de36c7e3e255285be1745; the latest hosted run,
34769598250, is terminal failure, not a live wait target.

The repaired genuine scientific runtime already passed all 3800 tests remotely
in run34766616352. Earlier dependency/fixture failures, the cancelled source build,
and the authentic binary/Hub1.28 repairs remain in the immutable CI engineering
archives and docs/ci-runtime-binary.md. Do not rebuild or republish that verified
FlashAttention artifact, rewrite original locks, or repeat solved installation work.

Run34768399864 confirmed that the hosted AVX2 CPU's Haswell BLAS produces136
different SVD diagnostic floats, while functional result tables agree exactly.
Run34769598250 then passed the original forced-SDE functional probe exactly, but
its first full numerical child failed in SDE's follow-exec launcher. Local minimal
controls reproduce the interaction with Python's default bulk FD closure.
The isolated original-CPython build selects its existing per-FD closing path;
close_fds=True, explicit pass_fds and all scientific isolation remain unchanged.
No Python/study source patch or disabled FD isolation is used.

The manual configured interpreter now passes the complete original numerical
graph and reviewed-paper Make release (21593/f5cf6c/0), including all eight shared
inputs. The repository builder also completes, and its actual interpreter passes
both real FD cases, the original functional probe and the complete Make replay
(50140/5b497d/0). All3800 final source-role cases pass without failure/error/skip
(53881/82d308/0). The retained evidence is in
reports/engineering-archive/dense-v3-ci-subprocess-runtime-v1; final publication
tracking is in /tmp/dense-v3-flash-wheel.L7A0IiP0/RUNNING.md.
See docs/cpu-numerical-replay.md. Local passes are not a claim of hosted acceptance.
The current document parent needs explicit checkout src/repository PYTHONPATH;
the original numerical child still clears it and executes its separate closure.

Never force SkylakeX natively on AVX2-only hardware, attach to existing processes,
disable child following, modify host security settings, redistribute the SDE binary,
or relax original numeric/source/document comparisons. Its licenses stay with the kit.

One earlier requested cleanup is unresolved: the safety-denied historical HF
withdrawal was never executed. Do not retry, split or bypass that denial. It
requires separate exact-scope direction and current/shared-artifact protection,
not another training run. GitHub issue-write 403 is a separate historical API
limitation; normal Git source publication succeeded without bypassing it.

For reproduction, use docs/versioned-paper-reproduction.md and
docs/source-version-testing.md. Every test module is assigned exactly once.
Historical numerical/worker consumers execute their authenticated original
source roles; a passing matrix is not admission of a changed fresh training
worker. Current primary training source retains its original 33 modules/56
bindings. The saved-checkpoint reader and restore add-on remain scoped entries,
not blanket new-host launchers.

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
scope. Normal Git push has now succeeded for the reviewed source payload, independently
of that denied API surface. One existing-integration/Issues-access question remains
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
