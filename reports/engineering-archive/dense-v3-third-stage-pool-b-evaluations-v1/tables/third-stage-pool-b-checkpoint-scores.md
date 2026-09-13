# Complete step-2345 checkpoint outcomes: original pool B

All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.
All six original pool-B configurations are retained; this is not the full twelve-rate cohort.
Rows follow optimizer/learning-rate order; no selection or inference is performed.

| Optimizer | Learning rate | Complete tasks | Step-2345 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 3e-6 | 14 / 14 | 58.1574 |
| AdamW | 1e-5 | 14 / 14 | 58.9643 |
| Muon | 3e-4 | 14 / 14 | 59.1254 |
| Muon | 3e-3 | 14 / 14 | 54.3264 |
| NorMuon | 1e-3 | 14 / 14 | 58.5034 |
| NorMuon | 3e-3 | 14 / 14 | 54.4666 |

The previous 36 complete checkpoint records are unchanged. Only six original
pool-B step-2345 states / 84 task values are new. The six pool-A step-2345
states and all twelve step-3126 states remain required. This partial rate
cohort does not establish an optimizer-wide comparison, full trajectory,
convergence rate, seed robustness or mechanism. No prior endpoint inference
or manuscript finding is changed.
