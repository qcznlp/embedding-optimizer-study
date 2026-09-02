# Dense short-branch temporal mechanism

Status: **complete**.

## Frozen decision

Overall spectral temporal bridge: **not supported (claimable negative result)**.

| Criterion | Passed |
| --- | --- |
| treatment_shift | true |
| outcome_shift | false |
| held_out_prediction | false |
| negative_control | true |
| coefficient_behavior | false |

## Predictor diagnostics

| Outcome | Predictor | Kind | LOSO RMSE improvement | Muon shrinkage | NorMuon shrinkage |
| --- | --- | --- | ---: | ---: | ---: |
| validation_loss_p95 | update_tail_energy_fraction | mechanism | -0.237 | -27.524 | -41.462 |
| validation_loss_p95 | update_stable_rank_fraction | mechanism | -0.603 | -59.757 | -75.268 |
| validation_loss_p95 | update_entropy_rank_fraction | mechanism | -1.013 | -12.911 | -18.061 |
| validation_loss_p95 | update_head_energy_fraction | mechanism | -2.865 | -53.908 | -75.433 |
| validation_loss_p95 | update_middle_energy_fraction | mechanism | -1.640 | -59.218 | -81.418 |
| validation_loss_p95 | update_row_norm_cv | mechanism | -2.195 | -7.239 | -1.606 |
| validation_loss_p95 | update_frobenius_norm | negative_control | -2.461 | -10.687 | -15.502 |
| validation_loss_p95 | weight_frobenius_norm | negative_control | -0.900 | -145.270 | -200.781 |
| unseen_margin_p05 | update_tail_energy_fraction | mechanism | -0.688 | -140.314 | -306.795 |
| unseen_margin_p05 | update_stable_rank_fraction | mechanism | -0.052 | -194.235 | -360.953 |
| unseen_margin_p05 | update_entropy_rank_fraction | mechanism | -1.401 | -169.659 | -352.520 |
| unseen_margin_p05 | update_head_energy_fraction | mechanism | -4.189 | -209.731 | -431.438 |
| unseen_margin_p05 | update_middle_energy_fraction | mechanism | -2.244 | -187.073 | -378.619 |
| unseen_margin_p05 | update_row_norm_cv | mechanism | -2.238 | -11.048 | -9.654 |
| unseen_margin_p05 | update_frobenius_norm | negative_control | -3.556 | -15.129 | -30.999 |
| unseen_margin_p05 | weight_frobenius_norm | negative_control | -0.833 | -35.387 | -74.749 |

> The shared-start randomization identifies optimizer-level accumulated effects. The post-treatment spectral predictor analysis is a small-sample, falsifiable causal-chain triangulation, not a formally identified causal mediation estimate.
