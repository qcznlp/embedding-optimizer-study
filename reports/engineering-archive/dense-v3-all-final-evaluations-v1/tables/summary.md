# All twelve complete v3 final-checkpoint evaluations

Native full-task readback: 2026-09-10T16:47:07.332219+00:00.
Each row is a final checkpoint (step 3907), averaged equally over all fourteen pinned decontaminated BEIR tasks.
Scores below are nDCG@10 multiplied by 100. Rows follow optimizer/rate order, not performance order.

| Optimizer | Learning rate | Complete tasks | Macro score |
| --- | ---: | ---: | ---: |
| adamw | 1e-6 | 14 / 14 | 56.4954 |
| adamw | 3e-6 | 14 / 14 | 58.3416 |
| adamw | 1e-5 | 14 / 14 | 58.8630 |
| adamw | 3e-5 | 14 / 14 | 58.9263 |
| muon | 1e-4 | 14 / 14 | 59.1016 |
| muon | 3e-4 | 14 / 14 | 59.2431 |
| muon | 1e-3 | 14 / 14 | 59.1914 |
| muon | 3e-3 | 14 / 14 | 55.7179 |
| normuon | 1e-4 | 14 / 14 | 59.0974 |
| normuon | 3e-4 | 14 / 14 | 59.3637 |
| normuon | 1e-3 | 14 / 14 | 59.1447 |
| normuon | 3e-3 | 14 / 14 | 55.8496 |

This is the complete original twelve-run final-checkpoint cohort, not the complete primary experiment.
All forty-eight intermediate checkpoints and the original validation/analysis requirements remain necessary.
No available-task average, optimizer-wide ranking, validation-based selection, significance test, or mechanism conclusion is produced.
The 168 original task values are in [task-scores.csv](task-scores.csv).
Original source-release and scientific-completion flags remain false.
