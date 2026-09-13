# Offline integration of the complete functional recovery flow

**Preparation only; no real coordinator was restarted and no execution authority
was created.** Actual functional progress remains one accepted pretrained vector
state and zero feature states. This milestone extends the earlier completed
[record-layout component](../dense-v3-functional-record-layout-candidate-v1/README.md),
without repeating its nineteen tests or counting simulation as scientific progress.

## What was implemented and checked

The [candidate](source/flow_candidate.py) composes eleven unchanged original
operational function bodies with exactly two changes in an isolated coordinator
body: prepare all required nested job-record parents, and route pretrained to
origin-preserving reuse instead of encoding it again. Original `worker`,
`encode_one`, `finalize_vectors` and `feature_worker` bodies are unchanged. The
candidate constructor admits explicit offline process fixtures only; it has no
production authentication, authorization writer or launch CLI.

The reuse component requires the original source/authority/plan, matching
started/exited records, successful native-verification proof, and exact vector
file bindings. It calls the required native inspector before and after copying.
Only the newly returned manifest's absolute location may differ; manifest/vector
bytes and all other receipt fields must agree. Original provenance is retained
in a separate reuse record, not invented new started/encoded/exited records.
**These inspector calls were stubbed in this test; real checkpoint-backed copied
pretrained readback is not established by this milestone.**

The nested record observer validates source, authority, plan, exact worker argv,
parent/PID/start identity, terminal and verification chains, copied baseline
origin/bytes and feature-file anchors. It delegates only recorded unterminated
handles to an injected exact-process reader. A missing handle remains unresolved,
not an inferred successful exit. Production source/authority/coordinator binding
must still be performed by its future admitted caller.

## Actual CPU test results

The final [test receipt](actual/tests-third.json), SHA-256
`6b704be5ad2377437f42778deee8ff97b79053b27f03ae1c8b3963d4d09dfdc3`,
records **18 tests, zero failures/errors/skips**, finished **2026-09-12 13:26:39 UTC**.
Original tool session 84850 ended with chunk `41dd47`, exit zero.

The complete synthetic flow covers:

- sixty new worker bodies plus one byte-preserved reused pretrained fixture;
- both inherited descriptor arguments for all sixty mocked GPU workers;
- the unchanged validation-priority body, full sixty-one-state vector finalizer,
  and full feature-worker body with 122 stub computations (two per state);
- all sixty nested worker chains and all sixty-one feature records in observation;
- worker exit 7, feature exit 8, incomplete vector iteration, feature calculation
  and readback failures, each preserving outputs and stopping without retry;
- changed origin, copied-vector corruption, feature-file corruption, wrong worker
  source/authority/parent/terminal identity and incorrect verification anchors;
- refusal of a production runtime by the offline constructor.

The [representative complete fixture](synthetic-complete-flow/) is deliberately
synthetic. Its `.npz` files are NOT NumPy archives, model calls are stubs, all
process handles/descriptors are fake, and table-count dependencies are reduced
test values. No Torch, NumPy, project numerical package, actual subprocess,
original lease or process inspector is loaded/executed by the tests. Therefore
this does not verify real encoding, 768-dimensional kernels, full scientific table
counts, GPU leasing or production dispatch. Other attempt fixtures are preserved
under `/tmp/dense-v3-functional-flow-candidate.TD3eLq4a`.

The first attempt retained fifteen tests with four failures and eight errors:
the synthetic manifest reader incorrectly required serialized JSON mapping order.
The coordinator already checks the execution order before serialization; the
fixture was corrected to compare mapping membership. The second fifteen-test
attempt passed. The final eighteen-test version adds observer checks for bound
reuse origin and actual referenced file bytes. Both earlier sources and receipts
remain intact; this is not an original scientific implementation change.

## Remaining recovery requirements

1. Obtain the still-unanswered scoped recovery execution approval.
2. Implement/review production source-bound recovery authentication and a fresh
   coordinator/observer namespace, without touching the failed original attempt.
3. Verify the genuine accepted pretrained copy with unchanged native checkpoint-
   backed inspection; preserve its original worker provenance.
4. Verify exact launched handles and original dual-lease ownership under the new
   authority, then encode the sixty remaining states and compute all real features.

The [verification](verification.json) independently checks the representative
filesystem records and all 66 original bound inputs unchanged. No model, GPU,
scientific analysis, manuscript or remote publication changed. Engineering
details in this archive belong nowhere in the paper.
