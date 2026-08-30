Under the disclosed post-hoc DenseOn scope, the formal mechanism tier evaluates every optimizer transform at the same frozen weights and on the same ordered eight-gradient history. The complete historical source artifacts still pass their content-hash and cardinality audits before the renderer selects the active DenseOn slice: 10 common-state anchors, 270 basis comparisons, 180 exact spectra, 60 bridge checkpoints, and 840 retrieval evaluation units.

### Retrieval time to an AdamW reference

| Family | Optimizer | AdamW reference | LR points reaching | fastest hours | median hours | right-censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.5858 | 2/4 | 1.407 | 1.416 | 2 |
| DenseOn | Muon | 0.5858 | 3/4 | 0.749 | 1.476 | 1 |
| DenseOn | NorMuon | 0.5858 | 3/4 | 0.756 | 1.507 | 1 |

The reference is the DenseOn median final nDCG@10 of the four AdamW learning-rate points. Passage is observed only at the five saved checkpoints; no interpolation is used, and non-reaching points remain right-censored. Checkpoint time is a step-proportional estimate from audited useful terminal wall time. This one-seed discovery analysis remains exploratory rather than a substitute for the validation-frozen three-seed confirmation.

### Same-state optimizer fingerprints

| Family | Operator | row CV / AdamW | top-1% row energy / AdamW | stable rank / AdamW | spectral norm / AdamW | cosine with AdamW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon | 2.855 | 1.586 | 28.916 | 0.015 | 0.470 |
| DenseOn | NorMuon | 0.711 | 0.979 | 22.021 | 0.018 | 0.480 |

Each cell is the median over ten frozen DenseOn anchors. Ratios use raw optimizer directions but are scale-invariant except for the explicitly reported spectral-norm ratio; the exact-spectrum intervention uses per-tensor Frobenius-matched directions. Weight decay is excluded from this comparison.

### Function-preserving basis sensitivity

| Family | Operator | mapped cosine | relative direction error | absolute norm-ratio error | predicted-descent error | Q/K spectrum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.96940 | 0.24738 | 0.00022 | 0.00488 | 0.02540 |
| DenseOn | Muon | 0.99946 | 0.03277 | 0.00007 | 0.00034 | 0.00148 |
| DenseOn | NorMuon | 0.99832 | 0.05793 | 0.00007 | 0.00360 | 0.00946 |

Each row is the median over 90 fixed comparisons: ten common-state anchors, three QKV layers, and three seeded RoPE-commuting rotations. Query and key share each split-half plane rotation, value rows are unchanged, and every direction is inverse-mapped before comparison. The transform preserves attention logits, so this table measures implementation-level coordinate dependence rather than retrieval quality; bfloat16 Newton--Schulz rounding is retained as part of the Muon runtime.

### Exact update spectra

| Family | Operator | stable rank / rank | entropy rank / rank | condition number |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0195 | 0.6984 | 61.26 |
| DenseOn | Muon | 0.5772 | 0.9698 | 16.72 |
| DenseOn | NorMuon | 0.4153 | 0.9603 | 23.82 |

The six matrices were fixed by early/middle/final depth and attention/MLP role before formal spectra existed. Values are medians over 60 exact spectra per optimizer on the active DenseOn anchors.

### Representation and score geometry

| Family | Optimizer | training margin | unseen margin | unseen query rank | pretrained top-1 agreement | mean BEIR nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0997 | 0.2499 | 0.6748 | 0.9286 | 0.5858 |
| DenseOn | Muon | 0.1240 | 0.2609 | 0.6770 | 0.9040 | 0.5901 |
| DenseOn | NorMuon | 0.1250 | 0.2611 | 0.6787 | 0.8996 | 0.5910 |

Rows are final-stage medians across all four frozen learning rates, not test-selected winners. Training and unseen probes remain separate; the latter contains 224 fixed examples balanced over all 14 decontaminated tasks.

### Descriptive temporal bridge

| Family | Predictor change | Outcome change | Transitions | Spearman ρ |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | weight-delta row CV | unseen margin | 48 | -0.067 |
| DenseOn | unseen margin | mean BEIR nDCG@10 | 48 | 0.531 |
| DenseOn | unseen query effective rank | mean BEIR nDCG@10 | 48 | -0.027 |
| DenseOn | trailing training loss (post-hoc) | mean BEIR nDCG@10 | 48 | -0.684 |

The first three geometry associations were fixed in the renderer and use within-run first differences across all optimizers. The final training-loss row is an explicitly post-hoc diagnostic. All four are one-seed observational summaries, not a causal mediation analysis. Same-state fingerprints identify what each update rule does; causal claims about accumulated retrieval behavior still require the matched shared-start branches and fixed-state spectral interventions.
