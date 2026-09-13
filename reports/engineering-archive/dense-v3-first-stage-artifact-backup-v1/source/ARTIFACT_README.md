# Corrected DenseOn: complete first-stage BEIR results

This data-only snapshot preserves every declared AdamW, Muon and NorMuon
configuration at step 782: twelve checkpoints, fourteen full-corpus tasks each,
168 task scores. It is not a new experiment, significance test or source release.

The accepted native readback completed on 2026-09-11 at 03:20:45 UTC.
Independent original raw-score reconstruction passed. The source observation
contains the twelve first-stage and twelve final checkpoints; this snapshot
selects all twelve first-stage checkpoints by declared stage, never by score.
No final-checkpoint scores or validation outputs are duplicated here.

## Contents

- native/beir-first-stage: 216 unchanged native task, metadata and complete-checkpoint files.
- provenance/beir-workers: all 336 original started/exited worker records.
- tables/task_scores.csv: all 168 first-stage nDCG@10 values.
- tables/checkpoint_means.csv: all twelve equal-task macro averages, in fixed configuration order.
- provenance/acceptance.json: data-only source identities and scope.
- artifact_manifest.json: complete file inventory with byte, SHA-256 and Git-blob identities.

There are 557 files including this README and the manifest. No model tensors,
raw query/document texts, program source, credentials, functional-analysis outputs
or failure logs are included. Historical paths, commands and PIDs in original
worker records are provenance, not instructions to execute or signal processes.

## Interpretation and reconstruction

Macro values use all fourteen original pinned tasks with equal weights. The
0-to-100 column is nDCG@10 multiplied by 100. Original auxiliary undefined metrics
remain in raw task files; no primary score is imputed. Four learning-rate cells
are not four independent training seeds. This snapshot alone does not establish
full training dynamics, optimizer-wide superiority, useful dimensions or mediation.

Restore this immutable prefix and verify every file against the trusted manifest
digest supplied by the restoration guide. Task JSON contains aggregate retrieval
metrics, not per-query ranking traces. Those aggregates suffice to reconstruct the
168 task values and twelve means without a GPU; full forward reproduction also
requires the pinned models, datasets and program versions.

All 36 checkpoints at steps 1563, 2345 and 3126, functional measurements and crossed
continuations remain separate unfinished requirements. The final-checkpoint and
validation snapshot remains unchanged at dataset revision
3883b677f87b1982f06016e9fadb8bb95e0cfc96. Its provenance/primary-v3-checkpoints.json
provides the immutable sixty-model download index. The dataset is
qcz/embedding-optimizer-study-analysis-artifacts; the model repository is
qcz/embedding-optimizer-study-checkpoints.

No source-code or scientific-completion flags are promoted by this artifact backup.
