# Diagnostic execution record

Historical observation at 2026-09-07 02:33 UTC; not a current heartbeat or acceptance.

All work is CPU-only. Numerical kernels, training sources, manuscript and protected runtime
remain unchanged. Exact before-documents and the first failed full-suite source are retained.

- Full 768D synthetic fixture generation: owned session **72591**, started 02:15 UTC.
  Root `/tmp/dense-v3-vector-reconstruction-audit.wM4h97`; last observed at least 52/61 states.
  Expected terminal artifact: `fixture.json`. Do not adopt a partial fixture or infer completion.
- First full suite: session **24211**, terminal exit 1; 2,418 cases / 17 failures, all mixed
  historical test-source rejection. Exact JUnit is retained in `attempt-1/full-failed.xml`.
- Isolation retry reproducer: session **55999**, last observed 49 cases reached 100%, terminal
  status still to be checked. Output `/tmp/dense-v3-vector-reconstruction-tests.zaQFHV/order-reproducer-retry.xml`.
- Focused retry: session **18986**, terminal exit 0, 38 cases. Output `focused-retry.xml` in the
  same test directory. Original 32-, 36- and 37-case green attempts remain separate files there.
- Full retry: session **15981**, running when this note was written; expected 2,419 cases.
  Output `/tmp/dense-v3-vector-reconstruction-tests.zaQFHV/full-retry.xml`.

The full-width fixture has not yet undergone the two independent cold reconstruction audits.
Use the exact new scripts and new empty audit roots; source-only or reduced-kernel controls
cannot replace full 768D replay. Freeze current sources and final receipts before acceptance.
The source/transport parent remains `bc18daa331507bc95481f7e99430157906d1c8ed4e3eebb5848915f872f283f0`.

## Later observation: 2026-09-07 02:38 UTC

- Fixture session **72591** is terminal exit 0. All 61 full-width states finished. Fixture receipt
  SHA-256 is `e7850b6ee6fe8acb97d8a49c21dc4e261511efbdce13bda82ccae72e13c09ffe` (1,265 bytes).
- Ordered reproducer **55999** is terminal exit 0, 49 cases. Full retry **15981** is terminal
  exit 0, 2,419 cases. Together with focused retry **18986**, no test process remains pending.
- First full-width cold audit **72073** is live. Root:
  `/tmp/dense-v3-vector-reconstruction-cold.osaUb4`. Its owned child writes progress to
  `cold-reconstruction/stdout.txt`; last observed completed state is 6/61. The script subsequently
  launches its two numerical-mutation refusal children before writing `result.json`.
- **No audit result, independent second replay or acceptance exists yet.** Continue polling the
  same owned session, not a new duplicate launch. After terminal success, bind its actual result
  SHA and run the same audit script with `--replay` / `--replay-sha256` in a fresh work directory.
  Then archive exact audit/replay records and run the new validator. All gates remain false until
  actually verified. Main/live front documents report this in-progress boundary.

## Later observation: 2026-09-07 02:42 UTC

Cold audit **72073** is confirmed live on the same handle; its child has completed 20/61 states.
No `result.json` has been observed and no second audit was launched. The four in-progress
documentation regressions are terminal exit 0 (session 52166), receipt
`/tmp/dense-v3-vector-reconstruction-tests.zaQFHV/in-progress-document-regressions.xml`.
All 82 current local documentation links resolve. Six numerical core files in both trees,
manuscript, original main ledger and exact stopped study handles were independently rechecked
unchanged. The next action is to continue this exact cold audit, then its two mutation children,
then independently replay the completed audit. The overall goal remains active and incomplete.

## Later observation: 2026-09-07 02:59 UTC

First cold audit **72073** is terminal exit 0. All 61 full-768D synthetic states, four complete
tables and 432 numerical output files compare exactly. Both one-unit table/attribution changes
were refused after complete envelope rehashing; partial numerical outputs remain preserved and
no success receipt was produced for either change. Actual missing primary authoring is still
refused. The cold child imported only 70 archived package modules.

Its exact result is copied to [audit.json](audit.json), SHA-256
`3a5ff804562f772aa29039beee9d03106141f55c98abea12c7c100b7c657212e`
(916,271 bytes), from `/tmp/dense-v3-vector-reconstruction-cold.osaUb4/result.json`.
The result binds 2,855 artifacts and does not grant primary/scientific/publication acceptance.

Independent second replay **7506** is now live, started 02:59 UTC at
`/tmp/dense-v3-vector-reconstruction-replay.9D4FHM`. It uses the first receipt's external SHA,
checks all original artifacts, then launches the actual archived CLI on a newly relocated copy.
It must repeat all 61 states and both numerical refusals before writing its own `result.json`.
Poll this exact handle and its `cold-reconstruction/stdout.txt`; do not start a duplicate.
No second-replay result or acceptance receipt exists at this observation.

## Later observation: 2026-09-07 03:04 UTC

Replay **7506** remains live on the same handle; 10/61 full-width states have completed exactly.
The first-audit documentation regressions are terminal exit 0, four cases, session **7265**;
their exact JUnit is [first-cold-document-regressions.xml](first-cold-document-regressions.xml).
The six numerical core hashes in both trees, manuscript and original ledger remain unchanged;
the narrow owned-handle reader confirms all three original dispatcher identities remain stopped
in place. No lease/controller transition, protected-helper or broad process inspection occurred.

This goal turn is **progress**: the first full cold audit is now complete and preserved, and an
independent replay is executing. Do not call the bounded milestone accepted until replay 7506
actually finishes its full positive and two negative children, its result is independently bound,
and the acceptance validator succeeds. Continue the exact owned handle before starting more work
on this frozen source surface. No formal, primary scientific or manuscript acceptance is supplied.

## Terminal replay observation: 2026-09-07 03:23 UTC

Independent replay **7506** is terminal exit 0. Its actual result is copied to
[replay.json](replay.json), SHA-256
`dc74d5e584967c4585138a7eb40d6d9751f9b59b2499f4e8f3efa4a233fa21d3`
(922,134 bytes). It independently verifies the first audit's retained artifacts and reconstructs
all 61 synthetic full-width states. The two complete child results are exactly equal, including
432 numerical file identities, four aggregate tables and all 70 archived package imports.
Both one-unit numerical changes again fail with retained partial output and no success receipt.
The source checks and exact original dispatcher-state comparison pass. No numerical audit is live.

While waiting, separate owned session **22465** prepared the next original-outcome raw fixture
and exited 0. It is preserved in its [own unaccepted preparation archive](../dense-v3-outcome-reconstruction-v1/README.md);
its synthetic raw-scoring coverage is not added to this vector reader's numerical acceptance.
The vector production/audit/test sources remained frozen throughout both processes.

This goal turn is **progress**: the second full-width cold audit is complete and the next
reader's raw-scoring fixture exists. Final source/documentation checks and the new-only
[acceptance record](validation.json) bind the completed vector evidence. The acceptance must
keep original outcome/geometry/inference closure, actual primary admission, formal replication
and scientific completion false. Do not restart the completed audits as missing work.
