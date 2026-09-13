# Offline functional job-record layout candidate

This is **local recovery preparation, not an installed fix or authorized recovery**.
The original functional coordinator remains terminal. Actual functional progress
is still **one accepted pretrained vector state and zero feature states**.
No model, lease, process observer, feature kernel or new coordinator was run.
The [original failure and accepted baseline](../dense-v3-first-stage-evaluations-v1/functional-first-state/README.md)
are unchanged. This engineering evidence belongs nowhere in the manuscript.

## Candidate and executed checks

[record_layout.py](source/record_layout.py) provides two operational components:

- Prepare the twelve required run subdirectories under an exclusively created,
  **empty** job-record root, keeping all original state IDs and record filenames.
- Enumerate the six JSON record families across all 61 declared states, checking
  filenames against their cell payloads and refusing unknown entries and symlinks.

Existing roots/partial attempts cannot be initialized again. The original exclusive
JSON/log writers are not changed. There is no dispatch, retry or authorization CLI.
The candidate is **not installed** in either original `dispatch.py` or `observe.py`.
It is not a complete recovery coordinator; observer provenance and exact-process
validation remain separate mandatory checks.

The [new tests](source/test_record_layout.py) completed at
**2026-09-11 06:31:31 UTC**, original tool chunk `0e97a3`, **exit 0**:
**19 tests, zero failures/errors/skips**. Actual [JSON](actual/tests.json) and
[JUnit XML](actual/tests.xml) are retained. Their source/file bindings and all
nineteen XML cases were independently reread; the console display was truncated,
not these files. No test rerun was performed to replace that display.

The tests execute the original `encode_one` and `worker` bodies using explicitly
mocked authentication, priority, leases, process creation, encoding and vector IO.
The unchanged original exclusive writer is extracted as an AST without importing
the geometry/model package. All fixture process IDs are deliberately `-1`.

- The original nested-path exception is reproduced **before any mocked child**.
- After preparing parents, all 61 cells traverse the original single-state/worker
  bodies; all **366 JSON records and 61 logs** are visible under the expected names.
  Model and vector-readback calls are stubs, not numerical acceptance.
- The old top-level started/feature glob sees only the pretrained record on that
  same fixture; the candidate sees all 61 without recursive unbound traversal.
- A nonzero mocked child retains its exit record and stops before verification or
  retry. Other cases cover missing/extra/duplicated/reordered states, traversal,
  existing evidence, unknown entries, symlinks, renamed payloads and partial JSON.
- **66 original bound files** are unchanged, including the frozen functional
  dependencies, failed receipt, accepted pretrained records and raw-vector bytes.
  No Torch, Transformers, NumPy or project numerical package was imported.

Full fixtures remain at the original isolated work directory
`/tmp/dense-v3-functional-record-candidate.snYoPDHo/fixtures`. They are synthetic
filesystem evidence, never experimental outcomes. The test runner is one-shot and
host-bound; do not run it inside this archived directory or call it a portable
model test. Passing this component does not clear a source/release/runtime gate.

## Still required before functional work can resume

1. Obtain the separately scoped recovery execution approval. Automatic goal
   continuations and these passing tests grant no such permission.
2. Build and review a new source-bound recovery coordinator/observer in a new
   namespace, without modifying the failed original attempt or numerical sources.
3. Authenticate and reuse the accepted pretrained vectors without re-encoding;
   preserve their original provenance rather than fabricate new worker exits.
4. Integrate and verify complete record writing/observation, original dual leases,
   unchanged priorities, native state acceptance and stop-on-failure boundaries.
5. Actually encode the remaining sixty checkpoints and complete all 61 feature
   states, rotations, inference and the held-dose bridge.

No authority file or recovery launch was created by this preparation. The earlier
one-GPU priority request remains unanswered and was not repeated. No primary
evaluation, training or manuscript source was changed, and no remote write occurred.

The [dated observations](actual/observations.json) have **348 / 840** primary task
cells and eight live original workers. The [native exits](actual/native-primary-exits.json)
reconcile all eight additions since the preceding 340-cell handoff. This does not
complete another checkpoint cohort; full-checkpoint results remain **24 / 60**.
See the [preserved prior handoff](before/CURRENT_EXPERIMENT.md),
[actual commands](actual/commands.json) and [verification](verification.json).
