# Fourth-stage pool-B evaluation backup and recovery

The six accepted original pool-B step-3126 outcomes now have an immutable public
data-only backup: **281 files / 983,157 bytes**, including 84 task scores, six
macro means and original native evaluation/worker provenance. All 54 currently
complete checkpoint outcomes now have off-host backups across the original
snapshots. This addition does not duplicate a new training run or establish an
optimizer-wide fourth-stage comparison.

Use the [restoration guide](../../../docs/fourth-stage-pool-b-evaluation-restoration.md)
and copy its local stdlib reader before closing this host. Repository code and
the guide are local; no complete GitHub/source release is claimed.

- Public dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `dd16fe4cfe3e21ba8c8b4583038ed93fe4ce6e92`.
- Parent: `9b2925acd9339b5ce25fd332ab8840ac50319407`.
- Manifest SHA-256: `ba3e103b503d52533ead4f2f0823a5937cf09111245563835369f2493fcb41d7`.
- Addition: `corrected-dense-correctness-v3/partial-fourth-stage-pool-b-evaluations`.

## Actual execution

The accepted [native outcome archive](../dense-v3-fourth-stage-pool-b-evaluations-v1/README.md)
retains 54 complete records; the backup selects exactly the original six pool-B
step-3126 records, never a score-selected subset. Its source/acceptance and all
prior numerical records remain unchanged.

The initial 34 adapter tests passed (2a04fb). Twelve new archive-semantic refusal
controls extend the final suite to **46 passing cases** (c7f54a); these are not
34 additional cases. The **18 recovered-reader controls** pass (958db9, terminal
76043d), including semantic corruptions after manifest/payload rehashing.
The actual staged readback passed at **09:16:40 UTC** (bafce0).

The single upload, session **92228**, ended with exit zero (870150) and wrote the
revision at **2026-09-12 09:17:04 UTC**. All files are new regular Git blobs;
the twenty other root entries and eleven prior corrected subtrees, including
root attributes and the repository card, remain unchanged. No old file was
overwritten/deleted, no new LFS rule was introduced, and no source was published.
The upload's original terminal stdout is retained in the conversation, not
reconstructed from a later artifact.

Anonymous recovery, session **92911**, completed at **09:19:12 UTC**, exit zero
(31d5f1). The separately copied stdlib reader, run with `python -I -B` outside
the project, passed at **09:19:29 UTC** (6980ee): 281 authenticated files, all
84 raw scores, 84 original exit-zero workers, six exact means and 90 CSV rows.
The anonymous-client warning is preserved. This is same-host recovery, not a
physical second-host experiment or new model evaluation.

Read [actual commands](commands.json), [test accounting](actual/tests.json),
[remote audit](actual/remote-audit.json), [anonymous recovery](actual/download-verified.json),
[independent replay](actual/recovered-independent-readback.json) and
[local verification](verification.json). Earlier backup exceptions retain their
original failed records and separate reconciliations; this addition does not
retry or relabel them.

## Remaining scope

The [separate native snapshot](observations.json) has **772 / 840** completed
task cells and four exact live original pool-A workers at **09:19 UTC**.
Pool B completed all 420 cells at 08:59 UTC. The remaining six pool-A step-3126
outcomes, functional analysis, crossed continuation and complete NAACL/source
delivery are still required. No queue, GPU lease, numerical source, statistical
rule, manuscript finding or functional-recovery authority changed.

The [before-copy](before/CURRENT_EXPERIMENT.md) preserves the preceding local-only
milestone. This evidence record belongs outside the manuscript.
