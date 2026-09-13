# Complete step-3126 checkpoint outcomes: original pool B

All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.
All six original pool-B configurations are retained; this is not the full twelve-rate cohort.
Rows follow optimizer/learning-rate order; no selection or inference is performed.

| Optimizer | Learning rate | Complete tasks | Step-3126 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 3e-6 | 14 / 14 | 58.4019 |
| AdamW | 1e-5 | 14 / 14 | 58.7965 |
| Muon | 3e-4 | 14 / 14 | 59.1362 |
| Muon | 3e-3 | 14 / 14 | 54.8374 |
| NorMuon | 1e-3 | 14 / 14 | 58.9692 |
| NorMuon | 3e-3 | 14 / 14 | 55.1803 |

The previous 48 complete checkpoint records are unchanged. Only six original
pool-B step-3126 states / 84 task values are new. The six pool-A step-3126
states remain required. This partial rate cohort does not establish an
optimizer-wide comparison, full twelve-configuration trajectory, convergence
rate, seed robustness or mechanism. No prior endpoint inference or manuscript
finding is changed.
