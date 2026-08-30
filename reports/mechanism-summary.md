The formal mechanism tier evaluates every optimizer transform at the same frozen weights and on the same ordered eight-gradient history. The values below are generated only after the complete 20-anchor matrix, 540 basis comparisons, 360 exact spectra, both 122-job representation tiers, and the 1,680-unit retrieval matrix pass their content-hash audits.

### Retrieval time to an AdamW reference

![Retrieval quality versus useful wall time](../reports/retrieval-dynamics/quality_vs_useful_wall_time.svg)

| Family | Optimizer | AdamW reference | LR points reaching | fastest hours | median hours | right-censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.5858 | 2/4 | 1.407 | 1.416 | 2 |
| DenseOn | Muon | 0.5858 | 3/4 | 0.749 | 1.476 | 1 |
| DenseOn | NorMuon | 0.5858 | 3/4 | 0.756 | 1.507 | 1 |
| LateOn | AdamW | 0.5898 | 2/4 | 1.673 | 2.523 | 2 |
| LateOn | Muon | 0.5898 | 3/4 | 1.673 | 3.377 | 1 |
| LateOn | NorMuon | 0.5898 | 3/4 | 1.692 | 1.693 | 1 |

The reference is the within-family median final nDCG@10 of the four AdamW learning-rate points. Passage is observed only at the five saved checkpoints; no interpolation is used, and non-reaching points remain right-censored. Checkpoint time is a step-proportional estimate from audited useful terminal wall time. The rule was locked after 160/1,680 discovery units were visible, so this is exploratory rather than a preregistration or a substitute for the three-seed confirmation.

### Same-state optimizer fingerprints

| Family | Operator | row CV / AdamW | top-1% row energy / AdamW | stable rank / AdamW | spectral norm / AdamW | cosine with AdamW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon | 2.855 | 1.586 | 28.916 | 0.015 | 0.470 |
| DenseOn | NorMuon | 0.711 | 0.979 | 22.021 | 0.018 | 0.480 |
| LateOn | Muon | 2.989 | 1.603 | 31.965 | 0.014 | 0.442 |
| LateOn | NorMuon | 0.667 | 0.974 | 23.996 | 0.017 | 0.452 |

Each cell is the median over ten frozen anchors. Ratios use raw optimizer directions but are scale-invariant except for the explicitly reported spectral-norm ratio; the exact-spectrum intervention below uses per-tensor Frobenius-matched directions. Weight decay is excluded from this comparison.

### Function-preserving basis sensitivity

| Family | Operator | mapped cosine | relative direction error | absolute norm-ratio error | predicted-descent error | Q/K spectrum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.96940 | 0.24738 | 0.00022 | 0.00488 | 0.02540 |
| DenseOn | Muon | 0.99946 | 0.03277 | 0.00007 | 0.00034 | 0.00148 |
| DenseOn | NorMuon | 0.99832 | 0.05793 | 0.00007 | 0.00360 | 0.00946 |
| LateOn | AdamW | 0.96906 | 0.24878 | 0.00025 | 0.00242 | 0.02389 |
| LateOn | Muon | 0.99943 | 0.03374 | 0.00005 | 0.00025 | 0.00150 |
| LateOn | NorMuon | 0.99760 | 0.06935 | 0.00005 | 0.00228 | 0.01079 |

Each row is the median over 90 fixed comparisons: ten common-state anchors, three QKV layers, and three seeded RoPE-commuting rotations. Query and key share each split-half plane rotation, value rows are unchanged, and every direction is inverse-mapped before comparison. The transform preserves attention logits, so this table measures implementation-level coordinate dependence rather than retrieval quality; bfloat16 Newton--Schulz rounding is retained as part of the Muon runtime.

### Exact update spectra

![Exact common-state update spectra](../reports/common-state/exact-update-spectra.svg)

| Family | Operator | stable rank / rank | entropy rank / rank | condition number |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0195 | 0.6984 | 61.26 |
| DenseOn | Muon | 0.5772 | 0.9698 | 16.72 |
| DenseOn | NorMuon | 0.4153 | 0.9603 | 23.82 |
| LateOn | AdamW | 0.0178 | 0.6711 | 68.14 |
| LateOn | Muon | 0.5037 | 0.9386 | 20.76 |
| LateOn | NorMuon | 0.3385 | 0.9334 | 23.98 |

The six matrices were fixed by early/middle/final depth and attention/MLP role before formal spectra existed. Values are medians over 60 exact spectra per family/operator; the figure shows the full normalized curves and interquartile bands.

### Representation and score geometry

![Representation dynamics](../reports/representation-space/representation-dynamics.svg)

| Family | Optimizer | training margin | unseen margin | unseen query rank | pretrained top-1 agreement | Late document-token coverage | mean BEIR nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0997 | 0.2499 | 0.6748 | 0.9286 | — | 0.5858 |
| DenseOn | Muon | 0.1240 | 0.2609 | 0.6770 | 0.9040 | — | 0.5901 |
| DenseOn | NorMuon | 0.1250 | 0.2611 | 0.6787 | 0.8996 | — | 0.5910 |
| LateOn | AdamW | 0.0061 | 0.0146 | 0.6439 | 0.9420 | 0.1702 | 0.5898 |
| LateOn | Muon | 0.0083 | 0.0163 | 0.6372 | 0.9085 | 0.1751 | 0.5949 |
| LateOn | NorMuon | 0.0084 | 0.0163 | 0.6369 | 0.9129 | 0.1744 | 0.5947 |

Rows are final-stage medians across all four frozen learning rates, not test-selected winners. Training and unseen probes remain separate; the latter contains 224 fixed examples balanced over all 14 decontaminated tasks.

### Late-interaction token utilization

![LateOn token-utilization dynamics](../reports/representation-space/late-token-dynamics.svg)

This panel reports the four prespecified MaxSim evidence summaries on both probe tiers. It is kept separate from the shared DenseOn/LateOn figure so a LateOn-only signal cannot change the cross-architecture metric definition after results are visible.

### Descriptive temporal bridge

| Family | Predictor change | Outcome change | Transitions | Spearman ρ |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | weight-delta row CV | unseen margin | 48 | -0.067 |
| DenseOn | unseen margin | mean BEIR nDCG@10 | 48 | 0.531 |
| DenseOn | unseen query effective rank | mean BEIR nDCG@10 | 48 | -0.027 |
| LateOn | weight-delta row CV | unseen margin | 48 | -0.439 |
| LateOn | unseen margin | mean BEIR nDCG@10 | 48 | 0.188 |
| LateOn | unseen query effective rank | mean BEIR nDCG@10 | 48 | -0.219 |
| LateOn | document-token coverage | mean BEIR nDCG@10 | 48 | 0.305 |
| DenseOn | trailing training loss (post-hoc) | mean BEIR nDCG@10 | 48 | -0.684 |
| LateOn | trailing training loss (post-hoc) | mean BEIR nDCG@10 | 48 | -0.496 |

The first seven geometry associations were fixed in the renderer and use within-run first differences across all optimizers. The final two training-loss rows are explicitly post-hoc diagnostics added after 1,456/1,680 discovery units were visible. All nine are one-seed observational summaries, not a causal mediation analysis. The same-state fingerprints identify what each update rule does; causal claims about later retrieval still require matched short branches or optimizer-switch interventions.
