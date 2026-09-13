# Complete step-2345 checkpoint outcomes

All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.
All twelve declared optimizer/learning-rate configurations are retained.
Rows follow optimizer/learning-rate order; no selection or inference is performed.

| Optimizer | Learning rate | Complete tasks | Step-2345 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 1e-6 | 14 / 14 | 56.0818 |
| AdamW | 3e-6 | 14 / 14 | 58.1574 |
| AdamW | 1e-5 | 14 / 14 | 58.9643 |
| AdamW | 3e-5 | 14 / 14 | 58.7964 |
| Muon | 1e-4 | 14 / 14 | 58.9630 |
| Muon | 3e-4 | 14 / 14 | 59.1254 |
| Muon | 1e-3 | 14 / 14 | 58.5017 |
| Muon | 3e-3 | 14 / 14 | 54.3264 |
| NorMuon | 1e-4 | 14 / 14 | 58.9582 |
| NorMuon | 3e-4 | 14 / 14 | 59.1522 |
| NorMuon | 1e-3 | 14 / 14 | 58.5034 |
| NorMuon | 3e-3 | 14 / 14 | 54.4666 |

The previous 42 complete checkpoint records are unchanged. Only six original
pool-A step-2345 states / 84 task values are new; the twelve rows above include
the six previously accepted pool-B states. All twelve step-3126 states
remain required. Four observed stages do not establish a full trajectory,
convergence rate, seed robustness or mechanism. No prior endpoint inference
or manuscript finding is changed.
