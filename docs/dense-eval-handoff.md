# Dense-only early evaluation handoff

`embed-optim-dense-eval-handoff` can overlap canonical confirmatory BEIR evaluation with
the final short-branch training run. It is intentionally fail closed and does not alter,
restart, or stop either training queue.

## Exact launch gate

The supervisor waits until every condition below is simultaneously true:

1. `configs/dense_training_queue.json` and all six bound matrices still match the frozen
   hashes, and the active scope amendment resolves to exactly `dense`.
2. The two queue ledgers account for exactly 18 unique jobs in frozen order. Exactly 17
   have both a durable successful ledger record and a terminal training receipt; exactly
   one is incomplete.
3. The incomplete job is the final short-branch job in its pool. The other pool has
   `complete=true`, all nine of its records are complete, and its original queue PID has
   exited.
4. The two canonical four-GPU token lists are disjoint. The candidate idle pool is not
   the pool assigned to the remaining run.
5. The supplied active queue PID is the expected Dense queue command and has exactly one
   direct matrix-training child. That child selects the remaining run, its bound matrix,
   `--families dense`, and only the active pool's four GPU tokens. The supervisor reads
   only those explicitly supplied queue PIDs and their direct child; it does not scan
   unrelated processes.
6. All nine Dense confirmatory runs and all 45 checkpoints pass the specialized derived-
   data deep audit before evaluation is launched.

If any identity is malformed or contradictory, the command exits with an error. A merely
transient condition remains in the waiting state until the configured condition timeout.

## Invocation

Run from the repository root and supply the original two queue coordinator PIDs:

```bash
uv run embed-optim-dense-eval-handoff \
  --ledger-a logs/dense-only-runtime/training-queue-a.json \
  --ledger-b logs/dense-only-runtime/training-queue-b.json \
  --queue-pid-a QUEUE_A_PID \
  --queue-pid-b QUEUE_B_PID \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
```

The default condition timeout is 48 hours. The evaluation, cooperative GPU-lease, and
subprocess watchdog timeouts are each explicit and default to 24 hours. An incomplete
handoff ledger requires `--resume`; a completed ledger is re-audited and exits
idempotently.

## Canonical reuse and collision safety

The early job runs the normal `embed_optim.confirmatory_evaluation` entrypoint with
`--families dense`, the Dense scope amendment, stage 5 only, and a fixed eight-task
subset: SciFact, NFCorpus, SCIDOCS, ArguAna, FiQA2018, NQ, QuoraRetrieval, and TRECCOVID.
Across three seeds and three optimizers this is exactly 72 units, recorded separately in
`reports/confirmatory/early-partial-evaluation-receipt.json`; it never overwrites the
canonical 126-unit receipt. Results remain under `results/confirmatory-beir`. Each
selected checkpoint is content-addressed in an additive
`evaluation_inputs.json` manifest before MTEB cache reuse. A changed model payload or an
older unbound result folder is rejected. The later canonical call therefore skips only
tasks produced for the identical checkpoint content and evaluation runtime.

Every BEIR evaluator now takes cooperative per-GPU leases before launching workers. The
same protocol encloses the complete short-branch validation/unseen-probe phase and the
spectral-transplant GPU matrix; audit-only, training-audit-only, summary, and dry-run paths
remain unlocked. The early evaluator leases only the proven-idle pool. Any later all-GPU
completion step records a waiting lease ledger and waits (up to its explicit timeout)
instead of sharing those GPUs. Lease state is stored under
`logs/dense-only-runtime/gpu-leases`, per-process lease ledgers are written below the
evaluation log directory, and the full handoff outcome is recorded in
`logs/dense-only-runtime/early-evaluation-handoff.json`.

Exit code `0` means strict 72-unit early-subset Dense confirmatory coverage and all provenance
manifests were verified. Exit code `2` means the one-run condition was not reached before
the condition timeout. Other failures are recorded with a typed error in the handoff
ledger.

Resume is fail-closed. Once the 17/18 boundary has been observed, the ledger freezes the
plan, scope, queue provenance, evaluator command, receipt path, and child process identity.
An interrupted supervisor can continue after the final training run has ended without
requiring that transient boundary again, but only after revalidating the frozen decision.
Catchable termination signals, timeouts, and non-zero evaluator exits clean the complete
evaluator process group before a retry. A present partial receipt must exactly equal a fresh
72-unit audit; only an absent receipt is treated as incomplete work.
