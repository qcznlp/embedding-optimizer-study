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
   `configs/dense_no_packing_sensitivity_implementation_protocol.json`. Final blog and paper
   rendering must follow `configs/dense_no_packing_publication_protocol.json`.
6. `docs/dense-no-packing-retrain.md` — exact operational commands.

If `logs/dense-no-packing-v1/recovery-supervisor-state.json` exists, read it after
`CURRENT_PROGRESS.json`. It is the atomic control-plane state for the recovery defined in
`configs/dense_no_packing_control_plane_recovery.json`; do not launch a competing matrix while its
phase is `waiting_for_adopted_training` or `matrix_running`.

If `logs/dense-no-packing-finalization/pipeline-ledger.json` exists, it is the atomic handoff state
for incremental corrected checkpoint backup and the post-training evaluation/analysis/publication
chain. Do not launch a competing corrected finalizer while its controller lease is held. Resume it
only with `python -m embed_optim.corrected_completion_pipeline --resume`; the exact operational
source/protocol/command contract must still match.

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
checkpoint into the blog or paper.

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
corrected publication block only through `python -m embed_optim.corrected_publication`, which
verifies all four upstream manifests and the source-bound publication protocol. Do not hand edit
its marked blog block or generated paper include. Do not launch or interpret corrected runs from
an uncommitted matrix.

The active paper scope is DenseOn only. LateOn files are historical provenance and must not be
promoted into primary inference or used to justify new computation.

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
