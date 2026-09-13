# Third-stage DenseOn evaluations: original pool B

All six original pool-B configurations at **step 2345** now have complete
fourteen-task BEIR results. The accepted cohort is **42 / 60 checkpoints**:
all twelve configurations at steps 782, 1563 and 3907, plus these six states.
Only **six checkpoints / 84 task values** are new relative to the
[previous 36-state readback](../dense-v3-complete-second-stage-evaluations-v1/README.md).
All 36 previous records are exactly unchanged.

The [six-row table](tables/third-stage-pool-b-checkpoint-scores.md) and
[84 task scores](tables/third-stage-pool-b-task-scores.csv) retain the original
queue membership, not a performance-selected subset. This is **not** a complete
twelve-configuration third-stage comparison or an optimizer-wide verdict.

## Executed evidence

The unchanged original [native reader](source/readback.py), SHA-256
`765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2`,
returned zero at **2026-09-11 22:39:50 UTC**. It authenticates both original
56-file source assemblies, dispatcher/authority, training and checkpoint parents,
complete native receipts, fourteen-task identities and revisions, original
successful workers, and raw score/metadata bytes. CUDA was not initialized;
no model or retrieval ranking was recomputed.

The [native bundle](actual/complete-checkpoint-readback.json) is 8,738,403 bytes,
SHA-256 `77a0d0bdb3251b86a3e1ab9d6c6fe56e96f9e8af388c16f3acd7bd4d035f3d24`.
It includes original source snapshots and remains a local engineering artifact,
not a public data-only upload.

The [independent reconstruction](actual/independent-raw-reconstruction.json)
returned zero at **22:40:25 UTC**. It checks **588 raw task values, 588 original
exit-zero workers and 672 score/metadata snapshots**, and reconstructs all 42
exact rational means. The old 36 records are unchanged; the six new states are
exactly the original pool-B step-2345 cohort. No missing rate or task is imputed.

The [independent reader](source/independent_readback.py), SHA-256
`2519cd98713ef0ad1bd5db2c12757d9a23d29a0ffa463f89df87db60b7cf2b91`,
is the prior accepted reader with cohort/parent/output-label changes only.
Its actual diff is retained in [commands.json](commands.json). No numerical
algorithm or original runtime source changed, and no new unit-test suite is
claimed. The genuine raw reconstruction is the executed check for this cohort.

The byte-identical archived reader returned zero at **22:40:47 UTC**.
All three CSV/Markdown tables reproduce byte-for-byte; the
[relocated receipt](actual/source-relocated-replay.json) preserves the actual
result. This was same-host source relocation with original raw paths, not an
anonymous HF recovery, a second-host model run, new inference or source release.

## Files and replay

- [tables/third-stage-pool-b-checkpoint-scores.csv](tables/third-stage-pool-b-checkpoint-scores.csv): six complete means.
- [observations.json](observations.json): separately timestamped exact original
  worker/observer reads, ending at **604 / 840** with eight live workers.
- [native-primary-exits.json](native-primary-exits.json): 96 original exit-zero
  task records between the prior material 508-task snapshot and the 604-task
  snapshot at 22:35:44 UTC. They are not 96 additional complete checkpoints.
- [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md): unchanged prior
  handoff; [verification.json](verification.json) binds this local archive.
- [commands.json](commands.json): actual native/independent/replayed executions
  and copies. The diff exit 1 denotes expected text changes. The native Torch
  import warning is retained; no GPU-process enumeration was performed.

On this host, with the original raw files and immutable parents present:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -I -B \
  /absolute/path/to/this-archive/source/independent_readback.py \
  --bundle /absolute/path/to/this-archive/actual/complete-checkpoint-readback.json \
  --output-directory /absolute/path/to/new-reconstruction
```

Use a new output directory; existing attempts are not overwritten. This reader
is host-bound, not yet a portable recovered-data entry point.

## Remaining scope

These six outcomes are **local only** at this milestone. The previous
[complete second-stage HF snapshot](../../../docs/second-stage-evaluation-restoration.md)
remains unchanged and does not contain step-2345 results.
No HF/GitHub write, original controller/source change, functional recovery or
manuscript change was made here.

The six pool-A step-2345 states and all twelve step-3126 states remain required:
**18 checkpoint outcomes**, with 236 task cells unfinished at the final snapshot.
Pool A continues step 2345; pool B has naturally entered step 3126. Partial
trajectory coverage does not establish convergence, seed robustness, an
optimizer-wide comparison or useful-dimension mechanism. The endpoint inference
is unchanged. Functional features, crossed continuation, portable analysis/source
delivery and the final NAACL paper remain incomplete. This engineering archive
belongs outside the manuscript.
