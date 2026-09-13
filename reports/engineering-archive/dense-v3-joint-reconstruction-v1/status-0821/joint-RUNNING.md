# Exact owned calls — joint reconstruction

Observed 2026-09-07 08:08 UTC. Only the following owned call is live:

- **74613**, independent complete replay:
  `/tmp/dense-v3-joint-replay.ygfZJm`.
  Started 07:45 UTC from the completed first result's external anchor below.
  It first verifies all 49,376 first-audit artifacts, then repeats the independent
  oracles, full positive reconstruction and all 21 controls. Child progress is in
  `cold-reconstruction/stdout.txt`; errors, if any, are in `stderr.txt`.
  No terminal replay result exists yet and no success is claimed.
  The independent oracles and all 61 vector states have finished. The complete
  positive result matches all 477 first-audit outputs; semantic controls are now
  running. No final replay/acceptance receipt is claimed.

Do not edit the 13 source files in
`scripts.audit_dense_v3_joint_reconstruction.SOURCES` while these calls are live.
The independent replay is live; do not infer completion from a timeout.

## Terminal earlier calls

- 18914: complete first cold audit, exit 0; all 477 numerical outputs match and
  all 21 semantic controls are refused. Only 75 archived package modules load.
  `/tmp/dense-v3-joint-cold.yR0bGv/result.json`, 14,303,063 bytes, SHA-256
  `9e1a06975185aa92b1e1114cef164f1df41637470a58f3a230055698d798fa77`.
  The separate payload-manifest anchor is
  `246927afda59eeefc6e260709d93a119b9464f3097cd4800e716282e9236fbfc`.
- 61164: read-only preservation/source/test precheck, exit 0; all 56 direct
  parent bindings, 13 current sources, both numerical cores and manuscript match.
- 91445: first completed audit/oracle receipt inspection, exit 0. This does not
  supply an independent replay or the combined final acceptance.
- 47895: isolated full regression, exit 0; **2,573 tests**, zero failures/errors/skips.
  `/tmp/dense-v3-joint-tests.0UZiTG/full.xml`, SHA-256
  `5e9c9904cbb1534bae18cd0d754c497bf0314eb5ccb529f077a8adbcaf62fa2f`.
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

## After the live replay genuinely finishes

Do not poll 18914, 47895, 61164 or 91445 again; they are terminal. Keep the 13 bound
sources unchanged while 74613 runs. If replay passes, retain its exact result hash,
copy its result without overwrite to `complete-replay.json`, and verify the full
acceptance with `validate.py --repository REPOSITORY --audit FIRST_RESULT
--audit-sha256 FIRST_RESULT_SHA --replay REPLAY_RESULT --replay-sha256 REPLAY_SHA
--output NEW_ACCEPTANCE_JSON`. It requires assertions, CPU-only environment,
both real completed receipts, preserved source/failed attempts and all payload
bindings. `validation.json` does not exist yet. The prepared validator has passed
format/lint plus the first-audit and preservation prechecks, not its final two-audit
acceptance. Do not write an acceptance receipt by hand.

The subsequent v3 publication interface is inspected in
[next-publication-boundary.md](next-publication-boundary.md), not implemented.
