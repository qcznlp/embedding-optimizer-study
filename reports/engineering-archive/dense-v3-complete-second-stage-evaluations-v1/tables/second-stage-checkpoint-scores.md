# Complete step-1563 checkpoint outcomes

All rows use the complete fourteen-task macro nDCG@10, multiplied by 100.
All twelve declared optimizer/learning-rate configurations are retained.
Rows follow optimizer/learning-rate order; no selection or inference is performed.

| Optimizer | Learning rate | Complete tasks | Step-1563 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 1e-6 | 14 / 14 | 55.5450 |
| AdamW | 3e-6 | 14 / 14 | 57.5478 |
| AdamW | 1e-5 | 14 / 14 | 58.5075 |
| AdamW | 3e-5 | 14 / 14 | 58.9759 |
| Muon | 1e-4 | 14 / 14 | 58.7939 |
| Muon | 3e-4 | 14 / 14 | 59.2762 |
| Muon | 1e-3 | 14 / 14 | 58.7669 |
| Muon | 3e-3 | 14 / 14 | 53.7881 |
| NorMuon | 1e-4 | 14 / 14 | 58.6914 |
| NorMuon | 3e-4 | 14 / 14 | 59.1243 |
| NorMuon | 1e-3 | 14 / 14 | 59.2096 |
| NorMuon | 3e-3 | 14 / 14 | 53.9686 |

The previous 30 complete checkpoint records are unchanged. Only six original
pool-A step-1563 states / 84 task values are new; the twelve rows above include
the six previously accepted pool-B states. All 24 states at steps 2345/3126
remain required. Three observed stages do not establish a full trajectory,
convergence rate, seed robustness or mechanism. No prior endpoint inference
or manuscript finding is changed.
