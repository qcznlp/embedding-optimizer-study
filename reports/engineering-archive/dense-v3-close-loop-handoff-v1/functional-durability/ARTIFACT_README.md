# Actual DenseOn functional measurements and continuation calibration

This immutable addition preserves all 61 corrected primary vector states, all
61 coordinate-attribution arrays and native feature tables, the complete
functional inference and post-result recipe sensitivity, and both genuine
eight-gradient calibration histories used by the crossed continuations.

`vectors/` and `features/` retain their original relative file layouts and
manifests. `calibration/` contains two states, each with its original request,
eight FP32 hidden-gradient shards, parameter mapping, direction norms and
derived learning rates. `results/` contains numerical results, not executable
source code. `provenance/` contains original completion/readback evidence.
`artifact_manifest.json` binds every file by size, SHA-256 and Git blob SHA-1.

Primary source checkpoints are separately indexed in
`qcz/embedding-optimizer-study-checkpoints`. The experiment remains DenseOn-only.
The fixed probe uses 224 queries, fourteen tasks, eight candidates and 768
coordinates. The four functional predictors and all controls are retained,
including unsupported findings. No robust dimension-use mechanism is claimed.

These artifacts are valid actual measurements, but they are not a source-code
release, a completed NAACL manuscript, a cross-host GPU resume demonstration,
or a new independent training replication. Original local provenance paths are
historical identities; consumers should resolve payloads relative to this
snapshot. No old HF paths are removed or replaced by this addition.
