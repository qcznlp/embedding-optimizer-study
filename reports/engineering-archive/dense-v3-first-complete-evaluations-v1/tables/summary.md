# First six complete v3 checkpoint evaluations

Native full-task readback: 2026-09-10T14:59:43.227818+00:00.
Each row is a final checkpoint (step 3907), averaged equally over all fourteen pinned decontaminated BEIR tasks.
Scores below are nDCG@10 multiplied by 100. Rows follow optimizer/rate order, not performance order.

| Optimizer | Learning rate | Complete tasks | Macro score |
| --- | ---: | ---: | ---: |
| adamw | 3e-6 | 14 / 14 | 58.3416 |
| adamw | 1e-5 | 14 / 14 | 58.8630 |
| muon | 3e-4 | 14 / 14 | 59.2431 |
| muon | 3e-3 | 14 / 14 | 55.7179 |
| normuon | 1e-3 | 14 / 14 | 59.1447 |
| normuon | 3e-3 | 14 / 14 | 55.8496 |

This is the complete first pool-B final-checkpoint cohort, not the complete primary experiment.
The other six final checkpoints and all forty-eight intermediate checkpoints remain required.
No available-task average, optimizer-wide ranking, validation-based selection, significance test, or mechanism conclusion is produced.
The 84 original task values are in [task-scores.csv](task-scores.csv).
Original source-release and scientific-completion flags remain false.
