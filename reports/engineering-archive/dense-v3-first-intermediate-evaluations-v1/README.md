# First six complete intermediate checkpoint evaluations

The unchanged native reader finished successfully at **2026-09-11 01:39:29 UTC**.
There are now **18 / 60 complete fourteen-task checkpoint evaluations**: the
previous twelve final checkpoints and six newly complete checkpoints at step 782.
The six new states are exactly the original evaluation pool B, not a score-selected
subset. The other six configurations at step 782 and all 36 states at steps
1563, 2345 and 3126 remain required: **42 intermediate checkpoints remain**.

## New complete states

Every row averages all fourteen original pinned full-corpus BEIR tasks with equal
weight. Values are nDCG@10 multiplied by 100. Rows follow optimizer/rate order,
not performance order. This is only half of the first-stage configuration grid.

| Optimizer | Learning rate | Step | Complete tasks | Macro score |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 3e-6 | 782 | 14 / 14 | 56.4409 |
| AdamW | 1e-5 | 782 | 14 / 14 | 58.2073 |
| Muon | 3e-4 | 782 | 14 / 14 | 58.8121 |
| Muon | 3e-3 | 782 | 14 / 14 | 54.2492 |
| NorMuon | 1e-3 | 782 | 14 / 14 | 58.5261 |
| NorMuon | 3e-3 | 782 | 14 / 14 | 53.4834 |

The [complete native bundle](actual/complete-checkpoint-readback.json) retains all
252 task scores and original worker/metadata snapshots for these eighteen states.
Exactly **84 task values and six states are new**; the twelve final records overlap
the [previous endpoint archive](../dense-v3-all-final-evaluations-v1/README.md).
Overlapping evidence is not additional runs, seeds or independent replications.

## Actual verification

- The [unchanged reader](source/readback.py) has SHA-256
  765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2,
  exactly matching the previously executed source. It authenticates both 56-file
  training assemblies, the original dispatcher/authorization, training-completion
  parents and actual checkpoint files, then calls the original inspect_evaluation.
- All eighteen rereads equal their original complete-checkpoint receipts. All
  fourteen task identities/revisions/splits and matching original worker successes
  are checked for each state. The CPU reader exited zero without initializing CUDA
  or recomputing embeddings, rankings, model updates or retrieval scores.
- [Independent raw-score reconstruction](actual/independent-raw-reconstruction.json)
  checks all 252 raw nDCG values, 252 original exit-zero worker records and 288
  raw-score/metadata snapshots. Each fourteen-task rational mean exactly matches
  the native bundle. The twelve earlier final records remain equal as complete
  parsed records, including their embedded raw snapshots; the prior bundle file
  also retains its original hash.
- The new 3,775,865-byte bundle has SHA-256
  288b7f328b40f3cf0d2384cc208338fe07360f4c1aa6af96ec5ae0beb8bb744d.
  [commands.json](commands.json) retains actual invocations and terminal outcomes.

The first independent observation attempt incorrectly tried to parse every
metadata snapshot as one JSON document; a multi-record metadata file raised
JSONDecodeError before that check completed. Its [failed invocation](actual/independent-attempt-1.json)
is retained. The corrected observation byte-verifies every metadata file and parses
only actual task-score JSON, matching the already established exporter distinction.
It exits zero. Neither the original successful native reader nor any evaluator,
numerical source, score, tolerance or historical receipt was changed.

## Ongoing execution and scope

At the separately timestamped **01:43 UTC** observation, primary task coverage is
**268 / 840**, eight exact original evaluation workers are live, and the functional
coordinator is live with **0 / 61** encoded states while awaiting a GPU. Pool B has
naturally started step-1563 ClimateFEVER; pool A continues step-782 tasks. These are
dated [observations](observations.json), not a permanent heartbeat.

This is new intermediate-checkpoint evidence, not a complete first-stage optimizer
comparison, all-rate inference, full-trajectory result, seed-robustness finding or
dimension-usefulness explanation. No optimizer-wide partial-cohort average, new
selection rule, significance test, mechanism conclusion or manuscript result is
introduced. Original endpoint inference remains unchanged. All 840 primary task
cells, 61 functional states and twelve crossed-continuation branches remain in scope.

The native bundle and report are local preservation, not a new public HF backup,
portable second-host reconstruction, committed-source release or completed paper.
No training was restarted; original live dispatchers, authorization files, numerical
sources, protected helper and historical stopped processes were not modified.
The [before copy](before/CURRENT_EXPERIMENT.md) preserves the preceding handoff.

The reader accepts a new, non-existing output file and enumerates the complete
checkpoint receipts present at invocation time. Its exact actual command is in
commands.json; a later invocation may include additional states and must not be
presented as the same frozen eighteen-record observation. Preserve this bundle for
the recorded snapshot. [verification.json](verification.json) binds this archive,
the handoff update and unchanged source/manuscript identities.
