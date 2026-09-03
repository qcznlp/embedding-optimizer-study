# Local steps lose, trajectories win

This is a **post-hoc exploratory analysis** declared after all 1,680 discovery BEIR units, the common-state intervention, and the mechanism bridge were complete, but before any confirmatory or shared-start-branch result existed.

## Local-to-global reversal

The local column compares per-tensor Frobenius-matched virtual steps at relative scale `1e-3`. Long-horizon columns compare final-stage medians over all four frozen learning rates. Positive values favor the challenger over AdamW.

| Family | Challenger | Local margin Δ vs AdamW | Final unseen-margin Δ | Final BEIR Δ | Reversal |
| --- | --- | ---: | ---: | ---: | --- |
| dense | muon | -3.034e-04 | +0.0110 | +0.0043 | yes |
| dense | normuon | -3.671e-04 | +0.0112 | +0.0052 | yes |
| late | muon | -3.976e-04 | +0.0017 | +0.0051 | yes |
| late | normuon | -4.527e-04 | +0.0017 | +0.0049 | yes |

All four contrasts reverse sign. The completed native trajectories therefore cannot be explained by Muon-family directions producing a larger immediate margin increase under a matched parameter-space step budget.

## Acquisition–preservation mismatch

The validation-selected recipe is the prospectively valid choice for confirmation. The best discovery-BEIR point is shown only as a descriptive oracle and never replaces that choice.

| Family | Optimizer | Validation-selected LR | BEIR-oracle LR | Selected BEIR | Oracle BEIR | Regret | Drift excess |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | adamw | 3e-05 | 3e-05 | 0.5899 | 0.5899 | +0.0000 | +0.0000 |
| dense | muon | 3e-03 | 3e-04 | 0.5608 | 0.5923 | +0.0315 | +0.0591 |
| dense | normuon | 3e-03 | 3e-04 | 0.5634 | 0.5934 | +0.0300 | +0.0469 |
| late | adamw | 3e-05 | 3e-05 | 0.5958 | 0.5958 | +0.0000 | +0.0000 |
| late | muon | 1e-03 | 3e-04 | 0.5966 | 0.5972 | +0.0006 | +0.0105 |
| late | normuon | 1e-03 | 3e-04 | 0.5962 | 0.5963 | +0.0002 | +0.0113 |

Dense Muon and NorMuon provide the sharpest mismatch: independent validation selects `3e-3`, yet those recipes lose roughly 0.03 mean BEIR nDCG@10 and approximately double the unseen score drift relative to the within-optimizer discovery oracle at `3e-4`. The LateOn mismatch is much smaller at its validation-selected `1e-3` recipe.

The defensible mechanism hypothesis is consequently trajectory-level: spectral reweighting changes future gradients and the acquisition–preservation frontier. At moderate strength this can accumulate useful retrieval margins; at excessive strength it can optimize the training-domain objective while eroding zero-shot rankings. The shared-start branches and spectral interventions must decide whether that hypothesis is causal.

> Claim boundary: This post-hoc analysis can establish a descriptive local-to-global sign reversal and an acquisition-preservation mismatch in the completed discovery seed. It cannot establish that spectral equalization causes the long-horizon gain; that requires the frozen shared-start branches and additional spectral interventions.
