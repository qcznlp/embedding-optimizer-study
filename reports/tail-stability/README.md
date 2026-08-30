# Mean improvement versus tail stability

This analysis is post hoc for the completed discovery intervention and prospectively frozen for the three-seed shared-start branches.

## Same-state discovery diagnostic

Every row compares a per-tensor Frobenius-matched `1e-3` virtual step with AdamW. Negative mean-margin contrasts favor AdamW on average; negative loss-tail contrasts and positive margin-tail contrasts favor the challenger on the worst queries.

| Family | Challenger | Mean margin Δ | p05 margin Δ | p95 loss Δ | p99 loss Δ | p99 anchor wins | Trade-off |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| dense | muon | -3.034e-04 | +2.178e-03 | -4.445e-02 | -7.579e-02 | 10/10 | yes |
| dense | normuon | -3.671e-04 | +2.344e-03 | -5.109e-02 | -9.361e-02 | 10/10 | yes |
| late | muon | -3.976e-04 | +1.678e-04 | -1.025e-01 | -2.933e-01 | 10/10 | yes |
| late | normuon | -4.527e-04 | +1.848e-04 | -1.139e-01 | -3.165e-01 | 10/10 | yes |

The result is a mean–tail trade-off, not a claim that Muon has a better average local step. Muon-family directions produce smaller average margin gains while reducing severe per-query regressions. Because this pattern was found after inspecting the discovery intervention, it is explanatory rather than confirmatory.

## Which queries occupy the bad tail?

This secondary diagnostic was added after its preliminary values were visible. It symmetrically selects each operator's worst 5% loss-change set (12/224 queries). A negative contrast on both selected sets indicates severity suppression on a shared fragile-query tail; an advantage only on AdamW's selected set indicates that the operator mainly changes which queries occupy its tail.

| Family | Challenger | Δ on AdamW tail | Δ on challenger tail | tail Jaccard | AdamW-tail baseline-margin percentile | Regime |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense | muon | -1.366e-01 | +2.032e-02 | 0.279 | 0.192 | tail redistribution |
| dense | normuon | -1.402e-01 | +1.097e-02 | 0.270 | 0.192 | tail redistribution |
| late | muon | -2.330e-01 | -1.779e-01 | 0.711 | 0.036 | shared-tail severity suppression |
| late | normuon | -2.585e-01 | -2.022e-01 | 0.670 | 0.036 | shared-tail severity suppression |

Late interaction shows a largely shared fragile-query set whose regression severity is reduced even when the challenger defines the tail. Dense retrieval shows much lower tail overlap and reverses sign on the challenger-selected set, which is evidence of tail redistribution rather than uniform query-wise dominance.

## Prospective shared-start confirmation

Pending: short-branch summary manifest is not available.

> Claim boundary: The discovery tail analysis is explicitly post hoc and can only describe a mean-tail tradeoff at fixed weights. The frozen short-branch rule is prospective relative to all three-seed shared-start outcomes and can test whether the signature persists under accumulated training, but even a positive result would not prove that tail stability alone mediates full-corpus BEIR gains.
