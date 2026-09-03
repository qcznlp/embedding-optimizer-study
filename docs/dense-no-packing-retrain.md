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
6. `configs/dense_no_packing_bridge_implementation_protocol.json`
7. `configs/dense_no_packing_retrain.yaml`
8. GitHub issue #41

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
`configs/dense_no_packing_bridge_implementation_protocol.json`.

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
