# Complete final-checkpoint statistical readout

All scores/differences below are nDCG@10 multiplied by 100. Each interval is simultaneous
over three contrasts within its declared family, not across both families together.

| Estimand | Contrast | Difference | Simultaneous 95% interval | Decision |
| --- | --- | ---: | --- | --- |
| primary | muon − adamw | +0.1569 | [-1.0808, +1.3947] | inconclusive |
| primary | normuon − adamw | +0.2073 | [-0.8903, +1.3049] | inconclusive |
| primary | normuon − muon | +0.0504 | [-0.1975, +0.2982] | inconclusive |
| secondary | muon − adamw | +0.3168 | [-0.1631, +0.7967] | inconclusive |
| secondary | normuon − adamw | +0.4374 | [+0.1463, +0.7285] | positive |
| secondary | normuon − muon | +0.1206 | [-0.2330, +0.4743] | inconclusive |

Primary: all four predeclared rates, equally averaged within each task.
Secondary: minimum-loss validation-only recipe per optimizer; BEIR never selects rates.
Both use 50,000 paired-task draws, seed 20260903 and the original fixed-observed-SE max-T procedure.
Intervals describe task variation on this fixed trained grid, not training-seed or per-query uncertainty.
Inconclusive does not establish equality, equivalence or absence of an effect.

All 168 final cells are present. This is not the complete 840-cell outcome campaign,
full trajectory analysis, functional/causal evidence, source release or completed manuscript.
The original whole-grid acceptance/publication gates remain unchanged and unpassed.
