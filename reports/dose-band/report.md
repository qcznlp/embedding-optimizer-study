# Spectral dose/band causal analysis

Overall frozen spectral-component chain: **not supported (claimable negative result)**.

- dense: local_supported=false (0/10 all-criterion anchors)
- forward_bridge_supported=false over 84 held-out rows

| Predictor | RMSE | Improvement over task+transition baseline |
| --- | ---: | ---: |
| baseline | 0.0110162 | 0 |
| spectrum_loss | 0.0111671 | -0.000150847 |
| spectrum_margin | 0.0110875 | -7.12409e-05 |
| basis_loss | 0.0110172 | -9.78894e-07 |
| basis_margin | 0.0110502 | -3.39735e-05 |

> The transplant randomizes spectral components at fixed weights and can identify immediate functional effects. Its task-aligned forward prediction is out-of-run evidence for a causal-chain bridge, but it is not a trained spectral-operator intervention and cannot by itself identify formal mediation of final BEIR gains.
