# Functional recovery: actual execution started

The owner explicitly approved proceeding on September 12. The new production
entry was tested, installed and genuinely launched, without restarting the old
failed attempt or changing numerical kernels. This closes the launch blocker;
it does not yet complete functional analysis or the paper.

- Final focused checks: **19 passed**, 14:01:54 UTC; session 6921, terminal
  `65da11`, exit 0. Eleven operational function bodies match the original AST.
- Actual unchanged native input admission: session 86421, terminal `84f65b`,
  exit 0. All 61 original jobs and input evidence are admitted.
- Real coordinator: session **27585**, PID **766302 / start 319358599**,
  started **14:04:07 UTC**. Read its exact source-bound observer; never infer
  current liveness from this launch record.
- At **14:06:33 UTC**, three new checkpoint vectors and the reused pretrained
  vector are accepted: **4 / 61**, with zero feature states. All three new
  workers exited zero and passed native vector readback. The fourth was live.
- Those first three genuine encodings averaged **19.64 seconds encoding** and
  **39.04 seconds worker wall time**. The observed start-to-start intervals were
  39.24 and 39.54 seconds. Extrapolation suggested about 37 more minutes for the
  remaining encodings at that snapshot, excluding whole-matrix readback and CPU
  feature computation. This is an estimate, not a completion deadline.

The latest dated state is in [CURRENT_EXPERIMENT.md](../../../CURRENT_EXPERIMENT.md).
The [deployed entry guide](launch/README.md) gives the exact read-only observer
command. Do not relaunch or mutate the live entry. Its failure policy preserves
partial results without an implicit retry; missing handles require reconciliation.

`verification.json` binds 66 unchanged original files and the initial archived
launch/test/three-state evidence. This README is a subsequent handoff explanation,
not an additional experimental result. `source-first/` and the first failed test
receipt preserve a synthetic fixture's coordinator-ID mismatch; the fixed fixture
and production observer agree. No such implementation narrative enters the paper.

The positive orchestration tests use explicit synthetic process/model/lease
dependencies. Actual progress is instead established by `launch/run/` worker
receipts and the genuine observer snapshots. Pretrained reuse preserves its
original successful worker provenance and writes no fabricated new worker exit.

The active output is a separate `dense-primary-v3-functional-dimensions-recovery-v1`
namespace. Existing primary training, BEIR, weight analyses, protocols, paper and
historical controllers remain unchanged. No protected-helper access, external
write or source publication was performed. Remaining work is all real functional
features/inference, crossed continuation, durable data recovery and the paper/repo.
