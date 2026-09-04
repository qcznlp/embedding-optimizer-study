# Corrected Dense no-packing replication

This is the active experiment. It corrects the material batch dependence found in the historical
SentenceTransformers flattened/packed path. The 34 historical Dense runs and their evidence remain
immutable; all corrected artifacts use new `dense-no-packing` namespaces.

## Read order

1. `PROJECT_STATUS.md`
2. `configs/dense_no_packing_execution_protocol.json`
3. `configs/dense_no_packing_evaluation_protocol.json`
4. `configs/dense_no_packing_analysis_protocol.json`
5. `configs/dense_no_packing_outcome_protocol.json`
6. `configs/dense_no_packing_bridge_implementation_protocol_v2.json`
7. `configs/dense_no_packing_sensitivity_implementation_protocol.json`
8. `configs/dense_no_packing_publication_protocol.json`
9. `configs/dense_no_packing_retrain.yaml`
10. GitHub issue #41

The formal matrix contains 12 DenseOn runs: four fixed learning rates for each of AdamW, Muon, and
NorMuon. Every run uses the same 500k rows, seed 42, seven explicit random hard negatives, no
in-batch negatives, context length 8192, global batch 128, and five checkpoints. The only intended
change from the historical discovery matrix is `dense_can_flatten_inputs: false` plus the new IDs
and output root.

## Artifact-only progress

```bash
python -m embed_optim.corrected_progress
```

This command reads only matrix outputs and study logs. It does not inspect system processes. A run
is complete only after its five checkpoints, final model, terminal state, accepted timing, and
`independently_padded` completion receipt agree.

At a meaningful handoff boundary, update the tracked machine-readable snapshot with:

```bash
python -m embed_optim.corrected_progress --output CURRENT_PROGRESS.json
```

The checked-in snapshot is deliberately updated on state transitions rather than every step. Also
update the narrative in `PROJECT_STATUS.md` and tracking issue #41 when a run completes or fails,
evaluation advances, a checkpoint backup is remotely verified, or the scientific interpretation
changes.

Fatal training markers and control-plane warnings are intentionally separate. In particular, the
repeated `ProcessGroupNCCL` TCPStore heartbeat warning caused by a vanished launcher is counted as
`control_plane_warning_markers.tcpstore_heartbeat_disconnect`, not as an NCCL data-plane error.
The total is a repeated log-line count; `control_plane_warning_affected_runs` is the run-level
count.

## Active control-plane recovery

The first two formal AdamW runs outlived their original interactive matrix controller. They remain
adopted in place until all eight recorded top-level training PIDs exit. A detached supervisor then
runs the unchanged matrix, whose deep gate either skips a complete run or resumes an incomplete
run from its latest valid scheduled checkpoint. The incident, exact PIDs, hashes, unchanged
configuration, and no-signal rule are recorded in
`configs/dense_no_packing_control_plane_recovery.json`.

Read the live recovery phase without inspecting processes:

```bash
python -m json.tool logs/dense-no-packing-v1/recovery-supervisor-state.json
```

Do not start another corrected matrix while that receipt says `waiting_for_adopted_training` or
`matrix_running`.

The original supervisor state cannot publish a heartbeat while its blocking matrix child runs, so
the mere `matrix_running` value is not sufficient evidence that the parent will launch the next
pair. A source-bound, artifact-only guard closes that handoff gap without inspecting or signaling
any process:

```bash
python -m embed_optim.matrix_handoff_guard
```

Its contract is `configs/dense_no_packing_matrix_handoff_guard.json` and its live state is
`logs/dense-no-packing-handoff-guard/state.json`. It first requires both active AdamW runs to pass
the existing deep completion gate. It then waits five minutes for either declared Muon successor
log/output to appear. Any successor artifact proves that the existing matrix advanced and makes
the guard yield. Only a full absence grace plus one final race check permits the guard to invoke
the unchanged recovery supervisor. The guard has its own exclusive lease and contains no process
inspection, signaling, evaluation, or checkpoint mutation.

## Training and resume

```bash
python -m embed_optim.matrix \
  --matrix configs/dense_no_packing_retrain.yaml \
  --families dense \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7 \
  --port-a 29610 \
  --port-b 29620 \
  --log-dir logs/dense-no-packing-v1 \
  --max-retries 2
```

The matrix runner skips deeply complete runs and resumes only deeply valid checkpoints. Never
overwrite `outputs/dense`, change the fixed micro-batch by optimizer, or treat a persistent
high-learning-rate failure as a run that may be dropped.

## Corrected evaluation

Saved SentenceTransformers checkpoints do not retain the runtime flattening flag. Therefore use
only the corrected entrypoints below; directly calling the historical evaluator would silently
restore automatic packing.

```bash
python -m embed_optim.corrected_validation_matrix

python -m embed_optim.corrected_beir_evaluation \
  --results-root results/dense-no-packing-beir \
  --log-dir logs/dense-no-packing-beir \
  --gpus 0,1,2,3,4,5,6,7
```

The validation stage selects a final learning rate per optimizer without BEIR. The BEIR stage
evaluates all 60 checkpoints over the same 14 pinned decontaminated tasks (840 task units). Both
entrypoints force and verify `can_flatten_inputs=false` after every model reload. Result roots bind
the protocol, matrix, checkpoint weights, package versions, and evaluator source hashes.

After a run is deeply complete, upload and remotely size-audit its entire output directory with:

```bash
python -m embed_optim.corrected_checkpoint_backup \
  --run-ids padded-adamw-1e-6
```

The default public destination is
`qcz/embedding-optimizer-study-checkpoints/corrected-dense-no-packing-v1/dense/<run-id>`.
The command refuses incomplete runs and writes one local audit receipt per uploaded run.

The completion controller waits for all five checkpoints before uploading a run. To close the
durability gap when the host may be shut down, upload an already sealed scheduled checkpoint with:

```bash
python -m embed_optim.incremental_checkpoint_backup \
  --run-ids padded-adamw-1e-5 padded-adamw-3e-5 \
  --steps 782
```

This command requires explicit runs and steps. It checks the frozen checkpoint schedule, all model,
optimizer, scheduler, trainer, and four-rank RNG payloads, the trainer's exact global step, a stable
local file snapshot, and the safetensors index. After upload it verifies every file by byte size and
either the Hugging Face LFS SHA-256 or Git-blob SHA-1 digest. Receipts live under
`reports/dense-no-packing/incremental-checkpoint-backup/` and explicitly set
`scientific_completion=false`; the ordinary completed-run backup remains authoritative for final
closure.

For unattended protection of every later checkpoint, launch the independent sealed-checkpoint
supervisor once from the repository root:

```bash
mkdir -p logs/dense-no-packing-sealed-backup
nohup python -u -m embed_optim.sealed_checkpoint_supervisor \
  --workdir "$PWD" \
  > logs/dense-no-packing-sealed-backup/supervisor.log 2>&1 &
```

Its atomic state is `logs/dense-no-packing-sealed-backup/state.json`. The state counts all five
stages of a run as covered when a valid whole-run receipt exists, counts an intermediate stage only
after its digest-verified receipt exists, and exits after all 60 stages are remotely covered. It
holds an exclusive lease and binds the watcher source, the underlying uploader source, the matrix,
destination, and timing arguments. A source or matrix change while it is active fails closed.

The watcher does not import CUDA, reserve a GPU, launch training, or modify the training and
completion controllers. For a final-stage checkpoint it waits three minutes and checks the
completion ledger so that the existing whole-run upload gets priority; if no whole-run backup takes
ownership, the per-checkpoint path becomes the durability fallback. Every watcher-generated
receipt still has `scientific_completion=false`.

Audit the live source runs on W&B without changing their histories or metadata:

```bash
python -m embed_optim.corrected_wandb_audit --allow-partial
```

Partial mode permits matrix runs that have not started yet, but it still fails on a missing remote
run after local completion, configuration drift, wrong deterministic run ID/name/group/tags, or a
finished run with the wrong terminal step or epoch. After 12/12 training completion, omit
`--allow-partial`; the complete audit then requires all 12 exact remote runs to be finished. The
receipt is a post-output operational provenance check and does not add a scientific endpoint.

The detached completion controller performs that upload incrementally while later training runs
continue, then executes the entire locked validation, BEIR, geometry, inference, publication, and
release-audit chain after training reaches 12/12:

```bash
python -m embed_optim.corrected_completion_pipeline
```

Its atomic state is
`logs/dense-no-packing-finalization/pipeline-ledger.json`. It acquires an exclusive lease, binds
its exact source/protocol/command contract, retries resumable operations, and refuses to proceed
from merely partial checkpoints. Resume an interrupted controller only with `--resume`; an exact
contract mismatch fails closed. This is an operational handoff controller, not a new scientific
protocol, and it does not alter the frozen training, evaluation, or analysis definitions.

The owner-directed removal of the parallel Markdown article changed only publication-bound source
hashes and therefore intentionally stopped the already running controller. The one-time transition
is frozen in `configs/dense_no_packing_completion_contract_migration.json`. After confirming that
the controller lease is free, migrate and resume with:

```bash
python -m embed_optim.completion_contract_migration
python -m embed_optim.corrected_completion_pipeline --resume
```

The first command accepts only the exact old and new contract hashes, unchanged controller,
matrix, execution protocol, arguments, and step order, plus the six explicit non-scientific
paper-only amendments. It archives the original ledger byte-for-byte before updating the live
binding. It must not be reused for later source drift.

The first audit-only backup pass after that migration exposed a separate receipt bug: a successful
audit rewrote the original upload commit fields as null because it had not itself uploaded. The
fix preserves an existing commit identity only after the run, repository, prefix, and inventory
digest all match, and fails closed on mismatched provenance. The follow-up contract hardening also
adds the invoked backup implementation to the completion controller's bound sources. Its one-time
transition is frozen in `configs/dense_no_packing_backup_provenance_migration.json` and runs as:

```bash
python -m embed_optim.completion_backup_contract_migration
python -m embed_optim.corrected_completion_pipeline --resume
```

The two original full-run upload commit identities remain authoritative. Redundant per-checkpoint
receipts created by the independent sealed-backup supervisor during the brief null-receipt window
remain valid durability evidence but do not replace the full-run receipts or change scientific
completion.

## Corrected weight-space analysis

The weight-space definitions and retrieval-bridge scientific plan were frozen before the first
corrected checkpoint existed. After one or more runs are deeply complete, their source-bound
geometry can be materialized incrementally with:

```bash
python -m embed_optim.corrected_geometry_matrix \
  --allow-partial \
  --skip-summary \
  --local-files-only
```

After all 12 runs are complete, omit the partial flags to generate the complete 60-row checkpoint
table, 660 all-rate run-pair subspace comparisons, and 60 optimizer-pair stage summaries:

```bash
python -m embed_optim.corrected_geometry_matrix --local-files-only
```

The analysis distinguishes the displacement between retained checkpoints from an instantaneous
optimizer update. Rank-16 left/right subspace overlaps are computed for every unordered run pair;
zero serialized displacements remain undefined and their coverage is reported rather than
imputed. Exact definitions and the predeclared retrieval-prediction test are in
`configs/dense_no_packing_analysis_protocol.json`.

## Corrected outcome summary

After all validation and BEIR units are complete, generate the inference and dynamics tables only
through the source-bound corrected entrypoint:

```bash
python -m embed_optim.corrected_outcome_summary
```

It refuses anything other than the complete 12-run, 60-checkpoint, 840-task-unit grid. The primary
estimand averages all four learning rates within optimizer before taking paired task contrasts. The
secondary estimand reads recipes chosen by independently padded validation loss. Both use a common
50,000-sample paired-task bootstrap and simultaneous max-T intervals over all three optimizer
contrasts. Five-stage AUC covers the observed 20%–100% range only and never imputes an initialization
score. See `configs/dense_no_packing_outcome_protocol.json` for the exact locked definition.

## Corrected retrieval bridge

After both complete summaries above exist, run the source-bound bridge implementation:

```bash
python -m embed_optim.corrected_retrieval_bridge
```

It joins all 60 run-stage rows to the nine predeclared geometry features. For each feature
separately, it compares the locked optimizer/stage/within-optimizer-rate baseline with the same
model plus that feature in four leave-dose-index-out folds. A feature is called predictively useful
only when pooled held-out RMSE decreases and at least three of four fold RMSEs decrease. The
implementation source was bound after the two first step-782 checkpoints existed but before any
corrected validation, BEIR, geometry, outcome, or bridge output existed; that timing and all
implementation choices are disclosed in
`configs/dense_no_packing_bridge_implementation_protocol_v2.json`. The v1 receipt was never run on
corrected outputs: an integration test caught that it looked up the future outcome manifest by
filename rather than its declared logical key. The v2 receipt fixes only that fail-closed interface
lookup and records the repair before any corrected outcome existed.

## Historical/corrected execution sensitivity

After the corrected outcome summary exists, generate the predeclared descriptive execution-path
sensitivity tables with:

```bash
python -m embed_optim.corrected_execution_sensitivity
```

This matches all 60 optimizer-rate-stage rows between historical packed training and corrected
independently padded training. It reports optimizer rankings and Muon/NorMuon-minus-AdamW deltas at
each normalized stage, keeping the two executions separate. It intentionally computes no pooled
confidence interval or causal packing estimate. The historical table, implementation, matching
rules, and output counts are source-bound in
`configs/dense_no_packing_sensitivity_implementation_protocol.json`.

## Final publication rendering

After the complete outcome, geometry, bridge, and sensitivity manifests exist, generate all
corrected result prose and tables with:

```bash
python -m embed_optim.corrected_publication
```

The renderer verifies every parent protocol and every consumed file by byte count and SHA-256. It
then writes the standalone corrected evidence report, writes
`paper/generated/corrected-no-packing.tex`, and emits its own source-addressed manifest. It
publishes every frozen contrast and all nine geometry features, including null or adverse results;
do not hand edit the generated sections. The complete rendering contract was frozen before any
corrected validation, BEIR, geometry, outcome, bridge, or sensitivity output existed in
`configs/dense_no_packing_publication_protocol.json`.

If only one four-GPU training job remains, the other disjoint pool may evaluate already completed
runs by passing their IDs and the idle GPU list to `corrected_beir_evaluation`. Do not overlap an
evaluation GPU with the active training pool.

## Interpretation boundary

The primary result is the final-stage optimizer-family contrast averaged over all four learning
rates, with simultaneous intervals defined in the execution protocol. The query-disjoint selected
recipe comparison is secondary. Training dynamics and weight-space analyses are reported at all
five checkpoints, but they cannot replace the full-corpus retrieval endpoint. Historical and
corrected results are compared descriptively and never pooled into one causal estimate.

Never inspect, read, edit, signal, stop, or otherwise touch `gpu.py` or its processes.
