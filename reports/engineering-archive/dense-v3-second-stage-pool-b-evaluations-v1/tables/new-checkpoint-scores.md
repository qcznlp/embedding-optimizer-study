# First six complete step-1563 checkpoint outcomes

All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.
This is the original pool-B cohort, not the full twelve-configuration stage.
Rows follow optimizer/learning-rate order; no selection or inference is performed.

| Optimizer | Learning rate | Complete tasks | Step-1563 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 3e-6 | 14 / 14 | 57.5478 |
| AdamW | 1e-5 | 14 / 14 | 58.5075 |
| Muon | 3e-4 | 14 / 14 | 59.2762 |
| Muon | 3e-3 | 14 / 14 | 53.7881 |
| NorMuon | 1e-3 | 14 / 14 | 59.2096 |
| NorMuon | 3e-3 | 14 / 14 | 53.9686 |

The previous 24 complete checkpoints are unchanged. Six step-1563 states and
all 24 states at steps 2345/3126 remain required. This cohort does not establish
an optimizer-wide trajectory, convergence rate, seed robustness or mechanism.
No prior endpoint inference or manuscript finding is changed.
