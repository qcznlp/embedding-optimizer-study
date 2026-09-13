# First actual functional vector state and terminal coordinator failure

This is an engineering-only handoff, not a manuscript result. No implementation
or debugging narrative belongs in the paper or appendix.

## Verified native state

The original coordinator 42916/start 299159282 obtained GPU token 0 through both
existing lease namespaces. Its only model worker,
325637/start 306837451, encoded the **pretrained** state and exited **0**.
Actual encoding took **19.103928430005908 seconds**; this is not whole-campaign
runtime or optimizer speed evidence.

The worker [saved its output](native/pretrained.encoded.json) at
2026-09-11 03:17:38 UTC. The original coordinator then performed the full native
inspect_vectors save/readback comparison and wrote
[pretrained.verified.json](native/pretrained.verified.json) at 03:17:40 UTC.

The [independent receipt/payload read](actual-payload-readback.json) checks that
the started/exited worker identities and commands agree, the native saved and
verified receipts are equal, and the original manifest/NPZ bytes match their
hashes. CPU NumPy loading with allow_pickle=False checks shapes, storage dtypes
and finite embedding values:

- Query embeddings: 224 × 768, float32.
- Document embeddings: 224 × 8 × 768, float32.
- Sample groups: 224 strings; sample IDs: 224 int64 values.

The [raw vectors](vectors/vectors.npz), 6,208,554 bytes, have SHA-256
e57057107312363619ebb1ba19b55892fe539568d9545726f3c9b9277325112f.
The [native manifest](vectors/manifest.json), 217,494 bytes, has SHA-256
90d9f044b53c05065b00d16a53a31a5c82a37d86e33552d797f6c38364e64fac.
These are byte-exact local preservation of an accepted native state, not a model
rerun, feature intervention, complete 61-state artifact or portable source release.

**Accepted vector states: 1 / 61. Feature states: 0 / 61.**
No trained checkpoint has been functionally encoded.

## Exact failure and cause

The original [failed.json](native/failed.json) records FileNotFoundError at
**2026-09-11 03:17:40.699222 UTC**, before the second worker was created. The
original coordinator session **39956** was read to terminal **exit 1**; its
actual output is retained in [commands.json](../commands.json).
Do not poll its stale live-only handle or relaunch the old coordinator.

The first checkpoint cell is
verified-v3-adamw-1e-5/checkpoint-782. The unchanged coordinator concatenates this
cell into a receipt filename beneath run/jobs. It creates run/jobs, but not the
checkpoint cell's run subdirectory; the original exclusive write_new helper
does not create parents. Creating that cell's admission record therefore fails
before subprocess.Popen for the next model.

The [actual 61-state path audit](actual-path-audit.json) finds sixty nested cells
and twelve absent required run directories. Only pretrained.started.json and
pretrained.exited.json exist; the second admission file was never created.
The retained [dispatcher](source/dispatch.py) and [observer](source/observe.py)
also show that the observer scans only top-level job receipts. That is a
**separate source-level coverage defect for future nested records**, not evidence
of an already hidden checkpoint worker or additional executed states.

The retained [25-test source](source/test_dispatch.py) tests actual population
mapping, source routing and bounded lease/argument behavior, but never exercises
the filesystem receipt lifecycle for a nested checkpoint cell. Its original
[receipt](authority/tests-first.json) is retained with its stated bounded scope;
those passing component tests do not clear this actual coordinator failure.
No optimizer, training kernel or primary BEIR evaluator was implicated by this
pre-dispatch path exception.

## Recovery boundary

The original failed record has automatic_retry_authorized=false. No source,
input, authority, native vector or failed attempt has been changed or deleted.
No automatic retry, replacement coordinator, new model worker or feature
computation was launched by this handoff. The original eight primary evaluators
continue on their own queues.

A separately authorized, source-bound recovery would need to:

1. Preserve the failed attempt and authenticate/reuse the already accepted
   pretrained state without repeating its model encoding.
2. Handle every original cell and every receipt suffix safely, preserving cell
   identities, exclusive-write protections and nested-record observation coverage.
3. Verify the complete 61-cell record lifecycle and recovery boundaries before
   GPU dispatch; retain both original lease namespaces and unchanged numerical
   encoding/feature kernels, masks, rotations and inference definitions.
4. Continue all sixty trained states, then the complete 61-state feature/readback
   and inference chain; no partial-state inference or goal reduction.

This recovery has **not been authorized or launched**. An automatic goal
continuation is not approval. The previous one-GPU priority question is separate
and remains unanswered; do not repeat it or use this failure to alter priority.
The source copies in this archive are forensic evidence, not ready-to-run recovery
commands. Original [authority](authority/authorization.json) and
[all-state inputs](authority/inputs.json) are preserved byte-for-byte.
