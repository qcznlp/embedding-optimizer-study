# Complete retained retrieval trajectories

All scores are fourteen-task macro nDCG@10 multiplied by 100. Every declared rate
and retained stage is included. The last column is the trapezoidal mean over
the observed 20–100% range, not an initialization-to-endpoint or wall-time area.

| Optimizer | Learning rate | 20% | 40% | 60% | 80% | 100% | Observed-range mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AdamW | 1e-6 | 54.257 | 55.545 | 56.082 | 56.337 | 56.495 | 55.835 |
| AdamW | 3e-6 | 56.441 | 57.548 | 58.157 | 58.402 | 58.342 | 57.875 |
| AdamW | 1e-5 | 58.207 | 58.508 | 58.964 | 58.796 | 58.863 | 58.701 |
| AdamW | 3e-5 * | 58.577 | 58.976 | 58.796 | 58.876 | 58.926 | 58.850 |
| Muon | 1e-4 | 58.230 | 58.794 | 58.963 | 59.034 | 59.102 | 58.864 |
| Muon | 3e-4 * | 58.812 | 59.276 | 59.125 | 59.136 | 59.243 | 59.141 |
| Muon | 1e-3 | 58.186 | 58.767 | 58.502 | 58.857 | 59.191 | 58.704 |
| Muon | 3e-3 | 54.249 | 53.788 | 54.326 | 54.837 | 55.718 | 54.484 |
| NorMuon | 1e-4 | 58.012 | 58.691 | 58.958 | 58.989 | 59.097 | 58.798 |
| NorMuon | 3e-4 * | 58.754 | 59.124 | 59.152 | 59.072 | 59.364 | 59.102 |
| NorMuon | 1e-3 | 58.526 | 59.210 | 58.503 | 58.969 | 59.145 | 58.879 |
| NorMuon | 3e-3 | 53.483 | 53.969 | 54.467 | 55.180 | 55.850 | 54.571 |

* indicates selection by final validation loss, never by BEIR.

## Validation-selected trajectories

| Progress | AdamW | Muon | NorMuon | Muon − AdamW | NorMuon − AdamW |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20% | 58.577 | 58.812 | 58.754 | +0.235 | +0.176 |
| 40% | 58.976 | 59.276 | 59.124 | +0.300 | +0.148 |
| 60% | 58.796 | 59.125 | 59.152 | +0.329 | +0.356 |
| 80% | 58.876 | 59.136 | 59.072 | +0.260 | +0.196 |
| 100% | 58.926 | 59.243 | 59.364 | +0.317 | +0.437 |

## Descriptive interpretation and limits

- Selected Muon exceeds selected AdamW at 5/5 retained stages. Its first retained point above selected AdamW’s final score is 40%.
- Selected NorMuon exceeds selected AdamW at 5/5 retained stages. Its first retained point above selected AdamW’s final score is 40%.

These are post-hoc descriptions of fixed validation-selected configurations, not
new significance tests or deployable early-stopping rules. Selection uses full-horizon
validation results, so a 40% crossing does not establish 60% compute or wall-time savings.
Four rates are not four training seeds. No additional uncertainty band is inferred.

The existing final-stage primary four-rate contrasts remain inconclusive. The secondary
validation-selected NorMuon–AdamW contrast retains its positive task-level simultaneous
interval; Muon–AdamW and NorMuon–Muon remain inconclusive. The complete trajectory
does not alter those tests, establish useful-dimension allocation, or explain a mechanism.
