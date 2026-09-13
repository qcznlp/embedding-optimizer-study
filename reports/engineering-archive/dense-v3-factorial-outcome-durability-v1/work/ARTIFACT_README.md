# Complete DenseOn continuation outcomes

This data-only snapshot preserves the complete current 2 source states × 2 reset
operators × 3 data-order seeds experiment. Each branch trains the same 50,000
query-positive-seven-negative groups, uses maximum context 8192, and retains five
stages. All twelve final checkpoints have fourteen full-corpus decontaminated
BEIR tasks: **168 task cells**. All **60 stage probes / 840 task-stratified probe
rows** remain separate eight-candidate measurements, not full-corpus retrieval.

`tables/` holds all six original table families, native readout and independent
arithmetic verification. `native/beir/` holds all 168 unmodified MTEB results,
final shared model/run metadata and complete checkpoint-result receipts.
`provenance/` preserves original native collectors, actual evaluation-worker
start/exit records, calibration/training/measurement identities and fixed design.
`artifact_manifest.json` binds every file by byte count, SHA-256 and Git blob ID.
Original absolute paths in native receipts are provenance, not download targets;
the manifest supplies the portable relative file inventory.

The three final post-continuation contrasts are the averaged source-state effect,
averaged continuation-operator effect, and their interaction. All are reported
with their original 100,000-resample two-way seed/task bootstrap, seed 20260904,
and **three marginal 95% intervals, not simultaneous coverage**. No preferred
contrast is selected. The endpoint diagonal difference equals state plus
operator, not state plus operator plus interaction. These order seeds are not
independent primary source-training seeds. This experiment does not establish
feature mediation, universal optimizer superiority or full-trajectory co-adaptation.

`related-artifacts.json` identifies the already verified immutable raw probe
snapshot, including all sixty model checkpoint links. These large existing
payloads are not duplicated here. No executable source, raw example text or
manuscript is included. This is result durability, not final source/paper release,
a new native model-admission run or physical second-host GPU verification.
