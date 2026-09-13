# Complete fourth-stage evaluation backup

All twelve corrected step-3126 checkpoints now have an immutable, public data-only
snapshot. The 557 files / 1,959,719 bytes contain all 168 native task scores and
their original checkpoint/worker records. Six pool-B states overlap the preceding
snapshot; only six pool-A states / 84 scores are newly backed up. No run is counted
twice and no previous snapshot was replaced.

The [complete trajectory readout](../../dense-v3-complete-trajectories-v1/README.md)
is the accepted numerical parent. It verifies all sixty states / 840 task values,
preserves the previous 54 records exactly, and reproduces every prior endpoint
comparison. This backup selects only the twelve fourth-stage records, not the
aggregate bundle containing embedded source.

## Actual transport and reconstruction

- Preparation and source/payload checks passed. The new adapter passes
  **50 focused tests**; the independent reader passes **18**, including semantic
  corruption of rehashed copied snapshots. These are data/transport tests, not
  new model experiments.
- The single upload completed at **2026-09-12 11:20:17 UTC**. Revision
  `5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc` is based on
  `dd16fe4cfe3e21ba8c8b4583038ed93fe4ce6e92`.
- The remote audit at **11:20:18 UTC** verifies all 557 paths and preserves
  twenty other root entries, all twelve prior corrected subtrees and root
  attributes. All added files are new regular Git blobs; no LFS/root rule was
  added and no path was deleted or overwritten.
- All 557 files were anonymously recovered at **11:23:59 UTC**, with every
  payload and manifest hash matching.
- The byte-identical archived standalone reader ran with `python -I -B` and
  returned zero at **11:25:49 UTC**. It used only the supplied recovered snapshot
  to reconstruct **168 native scores, 168 successful original workers, twelve
  exact means and 180 CSV rows**. Producer paths were not opened.

The exact prefix and commands are in the
[recovery guide](../../../docs/fourth-stage-evaluation-restoration.md).
The manifest SHA-256 is
`21f7c01540162767e86180ca3119e81df41e5b95e55b3044614fd9aeec0f604c`.
Actual original tool commands/results are retained in [commands.json](commands.json),
while [actual/](actual/) contains the preflight, upload mode guard, single upload,
remote audit, full download proof, isolated reconstruction and both test XMLs.
The complete local preservation check is [verification.json](verification.json).

## Scope

Together with the unchanged endpoint and earlier-stage snapshots, all sixty primary
checkpoint outcomes now have off-host raw-score backups. All sixty model checkpoints
were already backed up separately. Latest complete-trajectory tables and figures
still require their own data-only addition; this is not that snapshot.

The [source](source/) and guide are local WIP and were not uploaded to HF or GitHub.
All actual recovery happened on this same physical host; it is not second-host
model execution. No new training, retrieval, statistical inference, functional
recovery, continuation branch or manuscript/source release occurred. Existing
failure records and scientific/authority boundaries remain unchanged. The full
NAACL/reproducibility goal is not complete.
