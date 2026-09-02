## Causal controls and confirmation

The tables in this section are generated only after all frozen routing, local-step, shared-start, and confirmatory manifests pass their cardinality and content-hash contracts. They separate four questions that a single optimizer leaderboard cannot.

### Does AdamW parameter routing explain the result?

| Family | LR | AdamW | hybrid AdamW | difference | task W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: |
| DenseOn | 1e-06 | 0.5650 | 0.5652 | 0.0002 | 9/1/4 |
| DenseOn | 3e-06 | 0.5834 | 0.5831 | -0.0003 | 5/2/7 |
| DenseOn | 1e-05 | 0.5881 | 0.5885 | 0.0004 | 7/2/5 |
| DenseOn | 3e-05 | 0.5899 | 0.5899 | 0.0001 | 7/1/6 |

All four native AdamW learning rates are retained. The paired difference isolates Muon-style hidden/auxiliary parameter routing; it does not isolate orthogonalization.

### Do matched optimizer directions have immediate functional effects?

| Family | Direction source | Applied sign | delta loss | delta margin | delta MRR | delta top-1 | anchors lowering loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | descent | -0.0135 | 0.0009 | 0.0032 | 0.0049 | 0.90 |
| DenseOn | AdamW | sign reversal | 0.0232 | -0.0014 | -0.0040 | -0.0049 | 0.00 |
| DenseOn | Muon | descent | -0.0099 | 0.0006 | 0.0025 | 0.0031 | 0.90 |
| DenseOn | Muon | sign reversal | 0.0109 | -0.0008 | -0.0024 | -0.0045 | 0.10 |
| DenseOn | NorMuon | descent | -0.0091 | 0.0005 | 0.0037 | 0.0054 | 0.90 |
| DenseOn | NorMuon | sign reversal | 0.0088 | -0.0006 | -0.0021 | -0.0031 | 0.10 |

Every row uses the common relative scale 0.001 at fixed weights with per-tensor Frobenius matching; the sign-reversal row is the directionality control. These are immediate virtual-step effects, not claims that one step reproduces a native trajectory.

### Do direction effects accumulate from a shared checkpoint?

| Family | Final-stage contrast | delta loss (W/T/L) | delta margin (W/T/L) | delta MRR (W/T/L) | delta top-1 (W/T/L) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon - AdamW | -0.0681 (3/0/0) | 0.0024 (3/0/0) | 0.0161 (3/0/0) | 0.0229 (3/0/0) |
| DenseOn | NorMuon - AdamW | -0.0484 (3/0/0) | 0.0017 (3/0/0) | 0.0098 (3/0/0) | 0.0133 (3/0/0) |
| DenseOn | NorMuon - Muon | 0.0196 (1/0/2) | -0.0007 (1/0/2) | -0.0063 (0/0/3) | -0.0095 (0/0/3) |

These are final-stage means over three independently ordered 50K-query branches starting from the same 60% AdamW checkpoint and calibrated to the same hidden update-to-weight target. They use frozen probes rather than a second full BEIR run.

### Does the tail signature survive accumulation?

| Family | Challenger | delta on AdamW tail | delta on challenger tail | tail Jaccard | post-hoc regime |
| --- | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon | -0.1366 | 0.0203 | 0.2787 | tail redistribution |
| DenseOn | NorMuon | -0.1402 | 0.0110 | 0.2698 | tail redistribution |

The fixed-state cross-tail identity diagnostic is post hoc: it distinguishes severity suppression on a shared fragile-query set from redistribution to a new worst set. The separately frozen three-seed endpoint rule tests whether the loss-tail and unseen-margin signs persist after shared-start accumulation:

| Family | Challenger | validation loss p95 delta | loss seed wins | unseen margin p05 delta | margin seed wins | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon | -0.1752 | 3/3 | -0.0063 | 0/3 | mixed |
| DenseOn | NorMuon | -0.1335 | 3/3 | -0.0040 | 1/3 | mixed |

This accumulated persistence test is prospective relative to the branch outcomes, but it does not establish that tail stability mediates a full-training BEIR gain.

### Post-hoc spectrum-versus-basis causal decomposition

| Family | Immediate metric | spectrum main effect | basis main effect | interaction |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | contrastive loss | -0.0000 | 0.0044 | -0.0104 |
| DenseOn | positive margin | -0.0000 | -0.0003 | 0.0006 |

The 2x2 transplant holds the checkpoint and evaluation examples fixed while swapping singular values and singular vectors. It therefore causally decomposes the immediate functional difference at these fixed states, but it is a post-hoc explanatory intervention rather than a confirmatory retrieval analysis. Its query-tail readout is:

| Family | Condition | loss p95 delta | margin p05 delta | delta on AdamW tail | delta on condition tail | tail Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon native | 0.1114 | -0.0074 | -0.1143 | 0.0268 | 0.3333 |
| DenseOn | Adam basis + Muon spectrum | 0.1394 | -0.0078 | -0.1237 | 0.0245 | 0.2632 |
| DenseOn | Muon basis + Adam spectrum | 0.1450 | -0.0078 | -0.1431 | 0.0460 | 0.2316 |

These fixed-state contrasts can attribute an immediate effect to spectrum versus basis; they cannot show that either component causes the full-training BEIR outcome.

### Does the validation-frozen recipe replicate?

| Family | Contrast | mean delta nDCG@10 | hierarchical 95% CI | familywise 95% CI | seed W/T/L | task W/T/L |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon - AdamW | -0.0306 | [-0.0426, -0.0181] | [-0.0464, -0.0138] | 0/0/3 | 2/0/12 |
| DenseOn | NorMuon - AdamW | -0.0304 | [-0.0413, -0.0182] | [-0.0446, -0.0138] | 0/0/3 | 2/0/12 |
| DenseOn | NorMuon - Muon | 0.0002 | [-0.0087, 0.0061] | [-0.0126, 0.0084] | 2/0/1 | 8/0/6 |

Recipes were selected on the query-disjoint validation set before these runs. Intervals independently resample seeds and tasks; aggregate MTEB files do not support a query-level significance claim. The nominal interval is shown beside a Bonferroni familywise 95% interval over all six comparisons prespecified before the post-hoc Dense-only scope amendment. Only the familywise interval determines positive, negative, or inconclusive headline language; every contrast and all win counts remain visible.

### Frozen causal-chain numerical tests

Overall frozen chain: **claimable negative**. Temporal: **claimable negative**; dose/band/forward bridge: **claimable negative**.

#### Shared-start temporal decision

| Criterion | Decision | Audited numerical evidence |
| --- | --- | --- |
| treatment_shift | pass | muon=3/3/normuon=3/3 |
| outcome_shift | fail | muon=0/3/normuon=1/3 |
| held_out_prediction | fail | validation loss p95=-0.237277 (decision gap -2.372766e-01); unseen margin p05=-0.688154 (decision gap -6.881539e-01) |
| negative_control | pass | validation loss p95 primary=-0.237277, update/weight=-2.46124/-0.900387, decision gaps=+2.223960e+00/+6.631100e-01; unseen margin p05 primary=-0.688154, update/weight=-3.55619/-0.832549, decision gaps=+2.868040e+00/+1.443956e-01 |
| coefficient_behavior | fail | validation loss p95 muon abs(beta)=0.182353 to 5.20136 (gap -5.019009e+00); normuon abs(beta)=0.131624 to 5.58902 (gap -5.457397e+00); unseen margin p05 muon abs(beta)=0.00439217 to 0.620675 (gap -6.162827e-01); normuon abs(beta)=0.00213449 to 0.656986 (gap -6.548516e-01) |

The decision is all-required: failure of any row is a complete negative result.

#### Fixed-state dose, band, and basis tests

| Criterion | Supporting anchors | Threshold | Decision |
| --- | --- | --- | --- |
| loss_dose_monotone | 0/10 | 8 | fail |
| margin_dose_monotone | 0/10 | 8 | fail |
| tail_band_best_both_metrics | 0/10 | 8 | fail |
| basis_swap_negative_control | 2/10 | 8 | fail |

#### Held-run retrieval bridge (84 rows)

| Predictor | Kind | RMSE | Improvement | Matched control |
| --- | --- | --- | --- | --- |
| baseline | baseline | 0.0110162 | 0 | — |
| spectrum_loss | spectrum | 0.0111671 | -0.000150847 | basis_loss |
| spectrum_margin | spectrum | 0.0110875 | -7.12409e-05 | basis_margin |
| basis_loss | basis_negative_control | 0.0110172 | -9.78894e-07 | — |
| basis_margin | basis_negative_control | 0.0110502 | -3.39735e-05 | — |

| Spectrum predictor | ΔRMSE | Matched basis | Basis ΔRMSE | Baseline gap | Control gap | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| spectrum_loss | -0.000150847 | basis_loss | -9.78894e-07 | -1.508471e-04 | -1.498682e-04 | fail |
| spectrum_margin | -7.12409e-05 | basis_margin | -3.39735e-05 | -7.124088e-05 | -3.726742e-05 | fail |

> Temporal boundary: The shared-start randomization identifies optimizer-level accumulated effects. The post-treatment spectral predictor analysis is a small-sample, falsifiable causal-chain triangulation, not a formally identified causal mediation estimate.

> Dose/bridge boundary: The transplant randomizes spectral components at fixed weights and can identify immediate functional effects. Its task-aligned forward prediction is out-of-run evidence for a causal-chain bridge, but it is not a trained spectral-operator intervention and cannot by itself identify formal mediation of final BEIR gains.

## Conclusion

On the validation-frozen three-seed DenseOn retrieval comparison, Muon versus AdamW was negative (mean delta nDCG@10 -0.0306; familywise 95% CI [-0.0464, -0.0138]), while NorMuon versus AdamW was negative (mean delta nDCG@10 -0.0304; familywise 95% CI [-0.0446, -0.0138]). Across DenseOn's four frozen learning rates, routing-matched hybrid AdamW minus native AdamW averaged +0.0001 nDCG@10, with 3 positive, 1 negative, and 0 zero learning-rate points. This is descriptive evidence about parameter routing as an alternative explanation; it does not by itself identify the matrix rule or prove that routing accounts for the confirmatory Muon-family contrast. The frozen shared-start tail endpoint for DenseOn concluded Muon: mixed; NorMuon: mixed. The frozen temporal spectral bridge was a claimable negative, the fixed-state dose/band chain was a claimable negative, and their joint spectral-component account was a claimable negative. This explains only the tested chain: it does not identify formal mediation or establish a universal optimizer ranking.
