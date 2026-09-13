# Six complete fourth-stage evaluations: original pool B

All six original pool-B step-3126 checkpoints now have complete fourteen-task
retrieval outcomes. Complete coverage is **54 / 60 checkpoints**: all twelve
configurations at steps 782, 1563, 2345 and 3907, plus these six step-3126 states.
The six pool-A step-3126 states remain required. This is a descriptive data
milestone, not a full twelve-configuration trajectory or a new inference.

The [generated table](tables/fourth-stage-pool-b-checkpoint-scores.md) retains
all six configurations and [all 84 task values](tables/fourth-stage-pool-b-task-scores.csv).
The cohort is the original queue, not a selection based on observed scores.
All 48 previous complete records are unchanged; overlapping records are not new runs.

## Actual reconstruction

- The unchanged native reader completed at **2026-09-12 09:09:50 UTC**.
  Actual CPU session **55629** exited zero (terminal chunk **5379b7**).
  It checks the original source assemblies, training/checkpoint/data bindings,
  complete native results and successful original evaluation workers.
- Independent reconstruction passed at **09:10:48 UTC**, chunk **9b7fb8**.
  It verifies **756 raw task scores, 756 original exit-zero workers, 864
  score/metadata snapshots and all exact rational means**. Only six states /
  84 values are new. The 11,218,622-byte native bundle has SHA-256
  `7d12ba8f36caa05d890a75c3598176324086433408f3a322ed45c75eb66521f5`.
- The independent reader changes only cohort/parent bindings, counts and output
  labels from the prior pool-B reader. Six helper ASTs and the entire raw-score,
  worker and rational-mean loop remain unchanged (static check **14e3fc**).
- An archived-source replay passed at **09:11:27 UTC**, chunk **a311bc**.
  Its three CSV/Markdown tables are byte-identical. This is same-host source
  relocation, not a second-host model execution.

Read the [native bundle](actual/complete-checkpoint-readback.json),
[independent result](actual/independent-raw-reconstruction.json),
[relocated replay](actual/archived-source-replay.json) and
[actual commands](commands.json). The first local invocation failed to import
the package before writing output; the unchanged reader succeeded with the
original explicit PRIMARY PYTHONPATH. That failure and the Torch import warning
are retained, not described as a training or evaluation failure. The terminal
tool stdout is retained in the conversation; it was not fabricated from the bundle.

## Runtime and remaining work

The original [pool-B completion](actual/pool-b-completed.json) was written at
**08:59:47 UTC**: 420 task cells and 30 checkpoint outcomes. The separate
[native receipt reconciliation](actual/native-pool-reconciliation.json) verifies
all 772 original successful task exits, original source/authority bindings, and
no pool-B jobs without exits. It is not new independent score reconstruction.
The [09:11 UTC observation](observations.json) confirms four exact live original
pool-A workers and **772 / 840** task cells. No task, controller, lease or source
was restarted or modified.

These six new outcomes are **locally archived only** at this milestone, not yet
added to the earlier HF backups. The previous 48 complete outcome backups remain
unchanged. Functional analysis remains one accepted vector state / zero feature
states; its failed attempt is unchanged and recovery is not authorized. Formal
crossed continuation remains zero of twelve. No statistical rule, manuscript
finding or public source release was installed.

The [prior handoff](before/CURRENT_EXPERIMENT.md) and
[archive verification](verification.json) preserve scope and source identities.
This record is repository engineering provenance, not manuscript content.
