# Agent handoff instructions

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
   must follow `configs/dense_no_packing_publication_protocol.json`.
6. For the prospective state-by-operator mechanism follow-up, read both
   `configs/dense_no_packing_state_operator_factorial_protocol.json` and
   `configs/dense_no_packing_state_operator_factorial_implementation_protocol.json`, then follow
   `docs/state-operator-factorial.md`. Do not execute it from changed source bytes.
7. `docs/dense-no-packing-retrain.md` — exact corrected-matrix operational commands.

If `logs/dense-no-packing-v1/recovery-supervisor-state.json` exists, read it after
`CURRENT_PROGRESS.json`. It is the atomic control-plane state for the recovery defined in
`configs/dense_no_packing_control_plane_recovery.json`; do not launch a competing matrix while its
phase is `waiting_for_adopted_training` or `matrix_running`.

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
