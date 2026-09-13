---
pretty_name: Embedding Optimizer Study Analysis Artifacts
tags:
  - information-retrieval
  - dense-retrieval
  - optimization
  - reproducibility
---

# Embedding optimizer study analysis artifacts

This repository preserves analysis artifacts for the current DenseOn comparison
of AdamW, Muon and NorMuon. The [source repository and paper](https://github.com/qcznlp/embedding-optimizer-study/tree/b09b334973e69d6990e2811d5de5c1d0f471f170)
contain the completed experiments, protocols, exact analysis and restoration tools.
Model and optimizer states are in the separate
[checkpoint repository](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints).

## Current scientific artifacts

Use the immutable revisions and manifests in these guides, not a broad download
of the mixed-history repository. The current scientific namespace is
`corrected-dense-correctness-v3/`.

- [Primary retrieval trajectories](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/retrieval-trajectories-restoration.md):
  all sixty primary checkpoints across fourteen BEIR tasks.
- [Weight analyses](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/weight-analysis-restoration.md).
- [Functional vectors and calibration](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/functional-analysis-restoration.md):
  624 files, anonymously downloaded and checksum-verified.
- [Continuation probes and checkpoint links](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/continuation-probe-restoration.md):
  638 files, including all sixty five-stage continuation probes.
- [Continuation retrieval outcomes](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/docs/continuation-outcomes-restoration.md):
  593 files covering all 168 final full-corpus task results and associated records.

Eight-candidate functional probes are diagnostics, not full-corpus compressed
retrieval. Twelve primary learning-rate configurations are not twelve independent
training seeds; the continuation-order seeds do not replicate source training.
See the paper for supported conclusions and their scope.

The older `corrected-dense-no-packing-v1/` namespace is not the current v3 study.
Retention of any historical file or configuration does not make its results valid
or current. Runtime artifacts and separately identified shared inputs are preserved.

## Owner-approved withdrawal of obsolete results

This commit removes **5,839 explicitly reviewed old files** from the current
tree, following the owner's confirmation. The targets comprise invalidated Dense
retrieval, representation and weight-space outputs, affected follow-up analyses,
mixed result summaries and associated historical reports and execution logs.
Their former values must not be cited as results of the current study.

The [exact reviewed deletion manifest](https://github.com/qcznlp/embedding-optimizer-study/blob/b09b334973e69d6990e2811d5de5c1d0f471f170/reports/hf-obsolete-cleanup/plan-v2.json)
has SHA-256 `1193dbc2a1a483cba79b3532b4ef47f66eb9198fefc7492ea4521a9d00f1d026`.
The commit changes only those paths and this README. Git history, LFS objects,
original immutable restoration references and local engineering records are not
deleted. Old revisions can still contain withdrawn files; availability is not
scientific validity. Shared objects remain intact.

Shared `project/data/`, historical configurations, independently encoded untrained
Dense BEIR baselines and separately identified LateOn artifacts are retained.
LateOn remains outside the current paper; retaining it is not a correctness claim.
This repository is no longer a complete mirror of an old local `results/` folder.
Do not re-upload withdrawn files from broad historical snapshots.
