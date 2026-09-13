# Complete second-stage evaluation artifact backup

All twelve declared DenseOn optimizer/rate configurations at step 1563 now have
an immutable, data-only public snapshot: **557 files / 1,961,293 bytes**.
Full anonymous recovery and independent reconstruction of **168 raw task scores,
168 successful native workers, twelve exact means and 180 CSV rows** pass.
This closes the off-host data gap for the six newly completed pool-A states;
the six pool-B states overlap an earlier preserved snapshot, not new experiments.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `aa1515cbf98cb1b2af0be58cf05a4985bf1254cb`.
- Parent revision: `35ce505d3cf3389f4f1a2f35b46259749bc1d7ed`.
- Manifest SHA-256: `8ead2e55fa3c800199fc1da8051f1b1d8c0fe15246f9d0f0e5f688e34942c277`.

Use the [complete recovery guide](../../../docs/second-stage-evaluation-restoration.md)
and the [standalone reader](source/verify_recovered.py). The reader is local code,
not part of the public data-only snapshot or an updated GitHub release.

## Actual execution

1. A new adapter selects all twelve step-1563 states from the accepted
   [36-state native readback](../dense-v3-complete-second-stage-evaluations-v1/README.md).
   Exact configuration membership is checked against both original queues.
   The original transport primitives and numerical/dispatch sources are unchanged.
2. **34 adapter tests and 18 reader tests pass**, including deliberately corrupted
   copied data with recalculated hashes. These are bounded backup checks, not GPU
   training tests or additional optimizer experiments. See [test receipts](actual/tests.json).
3. The sole upload completed at **2026-09-11 14:22:54 UTC**. The remote audit
   confirms all twenty other root entries and six older corrected subtrees,
   including root attributes and the repository card, are unchanged.
   All 557 operations are regular new-file additions; no overwrite or deletion.
4. Anonymous recovery of every file completed at **14:27:01 UTC**.
   A byte-identical copied reader ran from the temporary working directory with
   `python -I -B` and reproduced all scores/means at **14:29:06 UTC**.
   It imports only the standard library and reads only the supplied recovered snapshot.

Native files, generated tables, worker identities and artifact hashes are bound by
[the manifest](actual/artifact_manifest.json). Original command outputs, including
the harmless anonymous-client warning and expected `diff` exit codes of 1, are
preserved in [commands.json](commands.json). No failed upload or model execution
was retried or relabeled. Earlier failures and backup exceptions remain untouched.

## Evidence boundary and live work

This is same-physical-host anonymous data recovery and numerical reconstruction,
not a second-host model experiment, source publication, new statistical test or
mechanism finding. The snapshot contains aggregate task values, not query-level
ranking traces. Only six states / 84 values are newly backed up.

At the separately timestamped **2026-09-11 14:29 UTC** observation, the original
primary observer and eight exact matching workers are live: **508 / 840 tasks**.
The previously verified **36 / 60** complete checkpoints are unchanged; all twelve
states at steps 2345 and 3126 still require full outcomes. The four newer task
exits since the preceding 504-task handoff are individually verified exit zero
and retained in [observations.json](observations.json). No worker was replaced.

Functional analysis remains **one accepted vector state, zero feature states**;
the original coordinator is terminal with its failure preserved. Recovery is not
authorized by automatic continuation and was not launched. Crossed continuation
remains **0 / 12**. No paper, frozen numerical source, original controller, protected
helper, GitHub boundary or old HF object changed.

The final local audit is recorded in `verification.json`. The
[previous handoff](before/CURRENT_EXPERIMENT.md) is an exact preserved copy.
See [the current handoff](../../../CURRENT_EXPERIMENT.md) for remaining whole-goal work.
