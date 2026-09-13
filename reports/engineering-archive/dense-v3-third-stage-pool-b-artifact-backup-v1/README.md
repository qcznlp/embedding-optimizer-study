# Data-only backup of six pool-B third-stage evaluations

All **281 files / 982,548 bytes** for the six accepted original pool-B
step-2345 checkpoints are now in a new immutable public snapshot. Each state
has fourteen complete BEIR task outcomes: **84 task values and six means**.
This does not add another training run, a complete twelve-configuration third
stage, a statistical test or a mechanism finding.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `dbd16adcc835c2c6b7537a61b8d511f027a2c351`.
- Parent: `aa1515cbf98cb1b2af0be58cf05a4985bf1254cb`.
- Manifest SHA-256: `1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a`.
- Prefix: `corrected-dense-correctness-v3/partial-third-stage-pool-b-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/1d238cb93e9e2b6791ebf5b6b6e2cd236a12a253e13d7d8f4fa0ecef6523828a`.

Use the [recovery guide](../../../docs/third-stage-pool-b-evaluation-restoration.md)
and [standalone reader](source/verify_recovered.py). The reader is local source,
not included in the public data snapshot or claimed as an updated GitHub release.

## Accepted input and remote scope

The [42-state readback](../dense-v3-third-stage-pool-b-evaluations-v1/README.md)
independently verifies all 588 complete task values and preserves the previous
36 records unchanged. Only its six original pool-B step-2345 states are selected
for this snapshot, by original queue membership rather than performance.

The snapshot contains 108 original native score/metadata/complete-admission files,
168 original started/exited worker records, two CSV tables, a data-only acceptance
summary, README and manifest. The source-bearing host readback is not uploaded;
neither are executable source bodies, model tensors or raw query/document examples.
Original paths, commands and PIDs are provenance, never recovery instructions.

The one actual upload, session **49769**, exited zero. It committed at
**2026-09-11 22:51:45 UTC** and passed its remote audit at **22:51:46 UTC**.
All 281 operations were regular new files, not LFS, overwrite or deletion
operations. All twenty other root entries, seven earlier corrected subtrees,
the repository card and `.gitattributes` remain unchanged. The original
training-artifact root-byte exception and all earlier snapshots are preserved.

## Actual tests and recovery

The [adapter](source/backup.py) and [reader](source/verify_recovered.py) are
scope-only adaptations of the previously accepted pool-B backup components;
the prior numerical transport and hashing helper remain byte-identical.
All five actual source diffs are retained in [commands.json](commands.json).

The final adapter call passed **34 tests**; the independent reader passed
**18 tests**. The latter include semantic corruption after complete rehashing,
wrong stage/task/worker identity, wrong external anchor and symlink refusal.
Actual [XML counts](actual/tests.json) are independently checked: 52 passing
final cases, no final failure, error or skip. These are bounded CPU data/transport
checks, not GPU/model tests or extra scientific experiments.

The initial adapter-test invocation omitted `PYTHONPATH` and produced seventeen
fixture setup errors plus seventeen passes. Its [original XML](actual/adapter-tests.xml)
and failed command remain intact. Supplying the explicit local module path fixed
the invocation; no source was changed and no remote write preceded acceptance.
The corrected [XML](actual/adapter-tests-with-module-path.xml) retains all 34 cases.
After a successful upload, an unauthenticated-download warning before its JSON
line caused a wrapper parse error. The original uploader remained terminal zero;
its exact output was reconciled and **no upload was retried**.

Full anonymous download, session **45770**, exited zero at **22:54:21 UTC**.
All 281 files match SHA-256/Git-blob/size identities. The copied stdlib reader,
run with `python -I -B` from `/tmp`, returned zero at **22:54:42 UTC** and
reconstructed all 84 raw scores, 84 original exit-zero workers, six exact means
and 90 CSV rows. This is same-host data recovery, not a second-host model run.

- [preflight.json](actual/preflight.json) and [artifact_manifest.json](actual/artifact_manifest.json): externally anchored population.
- [remote-audit.json](actual/remote-audit.json): complete immutable remote inventory and preservation check.
- [download-verified.json](actual/download-verified.json): actual complete anonymous recovery.
- [independent-recovered-readback.json](actual/independent-recovered-readback.json): isolated recovered-data score reconstruction.
- [observations.json](observations.json): exact primary snapshots, from 604 to **608 / 840**, and four original successful task exits.
- [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md): preserved pre-backup handoff.
- [verification.json](verification.json): sealed local archive identities and final checks.

All owned transfer/test/readback handles are terminal. The original eight primary
workers remain live; complete checkpoint coverage is **42 / 60**. Functional
recovery is still unapproved and no crossed continuation branch was launched.
No original numerical source, runtime authority, controller or manuscript changed.
Remaining outcomes, functional/causal work, portable source release and the final
NAACL paper remain incomplete. This engineering record belongs outside the paper.
