# Main recovery and successor handoff — prepared, not deployed

The current live main remains `4152531e...`. This archive preserves a reproduced post-training
resume defect and the exact candidate repair. It is engineering evidence only, not manuscript
content or an optimizer-quality result.

## Reproduction and repair

The old controller clears the step list on resume, reruns completed steps and opens prior attempt
logs in overwrite mode. The archived real loop reproduces this on temporary copies. The candidate
preserves the exact completed prefix, retries only the pending step, uses exclusive log creation,
retains orphan attempt logs and rejects changed commands or recorded log identities. All 17 main
commands and arguments are unchanged. Neither live nor isolated `src/.../corrected_completion_pipeline.py`
has been replaced; the candidate lives only in `after/`.

The first reproduction suite has one pass and seven failures against the unmodified controller;
the later 11 recovery tests pass. The original failed receipt and exact predecessor source remain
retained. The preceding full-suite receipt has 1,092 passes. These are temporary-loop checks, not
four-GPU checkpoint-resume tests, and do not exhaust every possible crash interleaving.

## Exact transition

The prior six-file proposal `af646ecf...` repairs evaluation-source authentication and table-header
syntax but not interrupted finalizer recovery. Adding only the archived controller candidate yields
the seven-file target:

`4380a3631ec196307c775cdf4ca766298075de0312ad9d7158c2795577424a56`

`combined-seven-file.patch` passes a read-only applicability check against live main. No patch was
applied. The older `preparation.json` is a timestamped projection made before the migration tool
existed; its `runtime_transition_implemented=false` must not be rewritten retroactively.

The new `scripts/migrate_main_handoff_recovery.py` is separately bound by `migration-protocol.json`.
It reconstructs the exact source/command target and migrates only a waiting, incomplete ledger with
zero post-training steps. It archives the original bytes, preserves all backups and previous
migrations, and changes only the contract, appended receipt and observation time. It cannot deploy
source, stop/start a controller, use a GPU or mutate HF. Its default is a read-only source/ledger
preflight; `ready` does not assert that a live controller lease is free.

Twenty-five dedicated tests exercise the actual migration on small temporary copies of the original
eight-backup ledger. They cover both archive/write interruption points, idempotence, held leases,
source/command drift, started ledgers, symlink rejection, changed archives and changed prior records.
No real controller lease was acquired. The seven-file patch, migration script and protocol are
separate deployment inputs; the old JSON-only publication migration is not valid for this change.

After a separately authorized sole-controller deployment of the exact candidate and its migration
inputs, the read-only check is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-study/src /usr/bin/python3 \
  /root/embedding-optimizer-study/scripts/migrate_main_handoff_recovery.py \
  --workdir /root/embedding-optimizer-study \
  --protocol /root/embedding-optimizer-study/reports/engineering-archive/main-resume-v1/migration-protocol.json \
  --expected-protocol-sha256 f337854db1266edd645bf117398f44e929182ecb67ded88f8121aa462accdf06
```

Appending `--apply` is a separate mutation: it requires the existing controller lease to be free and
rechecks all inputs under that lock. It does not restart the controller. Do not run it against an
active finalizer, a pipeline that has begun its post-training steps, a different source contract,
or as a substitute for the unresolved runtime-deployment authorization.

## Consistent successor

The third preparation layer is
`configs/dense_no_packing_state_operator_main_recovery_amendment.json`. It changes only the required
main hash, one amendment binding and observation timestamp in the isolated successor protocol.
Its predecessor protocol, auditor and tests are byte-identically retained under
`reports/engineering-archive/successor-resume-v1/before/` at their original receipt digests.

All three layers reconstruct exactly; all five real downstream protocol loaders pass. The local
successor is `ca0c7f5b...`, projected host successor `6749baa1...`, and neither is live `6605090d...`.
The successor still requires all 12 main runs, all 17 ordered main steps and all backups before its
unchanged 47-step plan. It rejects the older six-file parent and incomplete parents. Its actual
dry run reports main incomplete and starts no controller.

The focused selection passes 81 tests and the full isolated suite passes 1,124, with zero failures,
errors, skips or exclusions. Strict paper release remains incomplete because the primary and
dimension publications and three scientific includes are pending. No numerical analysis or
manuscript source was changed by this dependency repair.

Use module invocation from the isolated repository for the read-only projection:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src \
  /usr/bin/python3 -m scripts.audit_successor_contract \
  --repository /root/embedding-optimizer-story-refactor
```

The first direct script invocation failed to import the repository's `scripts` package; it produced
no plan or runtime action. `main-recovery-successor-projection-v2.json` records the successful module
invocation. This command-level failure is not a training or evaluator failure.
