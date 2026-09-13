# Exact owned calls — joint reconstruction

Observed 2026-09-07 07:09 UTC. Only the following two owned calls are live:

- **18914**, corrected full cold audit:
  `/tmp/dense-v3-joint-cold.yR0bGv`.
  Complete fixture and independent numerical/join oracles have returned pass.
  The child full reconstruction is running. Its progress is in
  `cold-reconstruction/stdout.txt`; errors, if any, are in `stderr.txt`.
  The main result.json does not exist yet and no success is claimed.
- **47895**, isolated full regression:
  `/tmp/dense-v3-joint-tests.0UZiTG/full.xml`.
  Last output reaches 30% without a failure marker. This is not a terminal result.

Do not edit the 13 source files in
`scripts.audit_dense_v3_joint_reconstruction.SOURCES` while these calls are live.
No independent complete cold replay has started. After a genuinely complete first
audit, use its exact result hash for replay; do not infer completion from a timeout.

## Terminal earlier calls

- 82055: first full-width fixture attempt, exit 1. All 61 raw vector/feature states
  finished; new fixture inference then rejected serialized, not yet decoded rate
  metadata. Inputs/source/failure are preserved under
  `/tmp/dense-v3-joint-smoke.wXGdbr`.
- 54182: initial 12 unit cases, exit 1 (one authoring-order mismatch).
- 96665: independent original nine-feature exact systems, exit 0 (540 predictions).
- 99616: typed metadata readback, exit 0 (zero identity mismatches in every table).
- 77505: synthetic raw-component preservation/transport, exit 0; not numerical acceptance.
- 65111: corrected 25 focused tests, exit 0.

The component transport is
`/tmp/dense-v3-joint-smoke.wXGdbr/raw-components`, externally anchored by
`331b9b00b6f78b0a020be9fda2032614ad732fdaeef9a8b7e574804000ab4d2a`.
Its 1,774 files all belong to the same explicitly simulated run population. The
earlier incomplete producer and every failed source remain unchanged.

Old geometry calls 13684/20415 and their tests are terminal/pass; do not rerun them.
All formal training/controller/source-publication restrictions remain in AGENTS.md.
