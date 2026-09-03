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

#### Six randomized paired contrasts

| Seed | Challenger | Δ early tail energy | Δ final loss p95 | Δ final margin p05 |
| --- | --- | --- | --- | --- |
| 314159 | muon | +0.0654264 | -0.219279 | -0.000634968 |
| 314159 | normuon | +0.0693881 | -0.190004 | +0.00197853 |
| 271828 | muon | +0.0648547 | -0.175235 | -0.00626683 |
| 271828 | normuon | +0.0689617 | -0.0713824 | -0.00400244 |
| 161803 | muon | +0.0652134 | -0.152545 | -0.0062747 |
| 161803 | normuon | +0.0693794 | -0.133485 | -0.00437956 |

#### All 16 temporal predictor estimates

| Outcome | Predictor | Kind | Label RMSE | Predictor RMSE | Relative improvement | Muon β label→with (shrink) | NorMuon β label→with (shrink) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation loss p95 | update_tail_energy_fraction | mechanism | 0.0591928 | 0.0732379 | -0.237277 | -0.182353→+5.20136 (-27.5236) | -0.131624→+5.58902 (-41.4621) |
| validation loss p95 | update_stable_rank_fraction | mechanism | 0.0591928 | 0.0949013 | -0.603257 | -0.182353→+11.0792 (-59.7569) | -0.131624→+10.0387 (-75.2683) |
| validation loss p95 | update_entropy_rank_fraction | mechanism | 0.0591928 | 0.119163 | -1.01314 | -0.182353→-2.53666 (-12.9107) | -0.131624→-2.50893 (-18.0614) |
| validation loss p95 | update_head_energy_fraction | mechanism | 0.0591928 | 0.228798 | -2.8653 | -0.182353→+10.0127 (-53.9085) | -0.131624→+10.0604 (-75.433) |
| validation loss p95 | update_middle_energy_fraction | mechanism | 0.0591928 | 0.156252 | -1.63972 | -0.182353→+10.9809 (-59.2177) | -0.131624→+10.8482 (-81.4184) |
| validation loss p95 | update_row_norm_cv | mechanism | 0.0591928 | 0.189106 | -2.19475 | -0.182353→-1.50232 (-7.23855) | -0.131624→+0.342982 (-1.60577) |
| validation loss p95 | update_frobenius_norm | negative_control | 0.0591928 | 0.20488 | -2.46124 | -0.182353→+2.13117 (-10.6871) | -0.131624→+2.1721 (-15.5023) |
| validation loss p95 | weight_frobenius_norm | negative_control | 0.0591928 | 0.112489 | -0.900387 | -0.182353→-26.6727 (-145.27) | -0.131624→-26.5591 (-200.781) |
| unseen margin p05 | update_tail_energy_fraction | mechanism | 0.00418127 | 0.00705863 | -0.688154 | -0.00439217→-0.620675 (-140.314) | -0.00213449→-0.656986 (-306.795) |
| unseen margin p05 | update_stable_rank_fraction | mechanism | 0.00418127 | 0.00439751 | -0.0517169 | -0.00439217→-0.857507 (-194.235) | -0.00213449→-0.772584 (-360.953) |
| unseen margin p05 | update_entropy_rank_fraction | mechanism | 0.00418127 | 0.0100382 | -1.40075 | -0.00439217→-0.749561 (-169.659) | -0.00213449→-0.754586 (-352.52) |
| unseen margin p05 | update_head_energy_fraction | mechanism | 0.00418127 | 0.0216978 | -4.18929 | -0.00439217→-0.925568 (-209.731) | -0.00213449→-0.923034 (-431.438) |
| unseen margin p05 | update_middle_energy_fraction | mechanism | 0.00418127 | 0.0135655 | -2.24436 | -0.00439217→-0.826049 (-187.073) | -0.00213449→-0.810293 (-378.619) |
| unseen margin p05 | update_row_norm_cv | mechanism | 0.00418127 | 0.0135373 | -2.23761 | -0.00439217→+0.0529153 (-11.0476) | -0.00213449→-0.0227398 (-9.6535) |
| unseen margin p05 | update_frobenius_norm | negative_control | 0.00418127 | 0.0190507 | -3.55619 | -0.00439217→-0.0708411 (-15.129) | -0.00213449→-0.068302 (-30.9992) |
| unseen margin p05 | weight_frobenius_norm | negative_control | 0.00418127 | 0.00766238 | -0.832549 | -0.00439217→+0.159818 (-35.387) | -0.00213449→+0.161686 (-74.7492) |

#### Fixed-state dose, band, and basis tests

| Criterion | Supporting anchors | Threshold | Decision |
| --- | --- | --- | --- |
| loss_dose_monotone | 0/10 | 8 | fail |
| margin_dose_monotone | 0/10 | 8 | fail |
| tail_band_best_both_metrics | 0/10 | 8 | fail |
| basis_swap_negative_control | 2/10 | 8 | fail |

#### All 10 fixed-state anchors

| Anchor | Loss dose λ=0/.25/.5/.75/1 | Margin dose λ=0/.25/.5/.75/1 | Loss band H/M/T | Margin band H/M/T | Dose L/M | Tail | Basis | All | Decision gaps L/M/T/B |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense/pretrained | 0/+0.112292/+0.14693/+0.172892/+0.212387 | 0/-0.00390625/-0.00751953/-0.0078125/-0.00976562 | +0.191247/+0.11791/+0.0939777 | -0.0078125/-0.00390625/-0.00390625 | fail/fail | fail | pass | fail | -1.122915e-01/-3.906250e-03/+0.000000e+00/+1.953125e-03 |
| dense/adamw-lr1e-5/checkpoint-782 | 0/+0.0858068/+0.0804761/+0.0924003/+0.140346 | 0/-0.00390625/-0.00390625/-0.00390625/-0.00585938 | +0.119569/+0.096515/+0.0948403 | -0.00556641/-0.00390625/-0.00390625 | fail/fail | fail | pass | fail | -8.580683e-02/-3.906250e-03/+0.000000e+00/+1.953125e-03 |
| dense/adamw-lr1e-5/checkpoint-2345 | 0/+0.0761643/+0.134485/+0.151421/+0.1428 | 0/-0.00390625/-0.00556641/-0.0078125/-0.0078125 | +0.158669/+0.0746775/+0.0928111 | -0.0078125/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -7.616431e-02/-3.906250e-03/-1.813361e-02/+0.000000e+00 |
| dense/adamw-lr1e-5/checkpoint-3907 | 0/+0.104283/+0.105001/+0.136804/+0.164926 | 0/-0.00390625/-0.00390625/-0.0078125/-0.0078125 | +0.159435/+0.0841789/+0.0616835 | -0.0078125/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -1.042826e-01/-3.906250e-03/+0.000000e+00/+0.000000e+00 |
| dense/muon-lr1e-3/checkpoint-782 | 0/+0.109402/+0.120257/+0.114413/+0.119132 | 0/-0.00390625/-0.00390625/-0.00390625/-0.00751953 | +0.0958187/+0.0795917/+0.0513318 | -0.00390625/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -1.094018e-01/-3.906250e-03/+0.000000e+00/-2.929688e-04 |
| dense/muon-lr1e-3/checkpoint-2345 | 0/+0.0713319/+0.110023/+0.112523/+0.138461 | 0/-0.00390625/-0.00390625/-0.00390625/-0.00585938 | +0.100107/+0.0651468/+0.0390793 | -0.00390625/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -7.133193e-02/-3.906250e-03/+0.000000e+00/-1.717831e-03 |
| dense/muon-lr1e-3/checkpoint-3907 | 0/+0.0843915/+0.0584801/+0.0836865/+0.10606 | 0/-0.00390625/-0.00390625/-0.00390625/-0.00390625 | +0.072524/+0.0593137/+0.0773411 | -0.00390625/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -8.439147e-02/-3.906250e-03/-1.802733e-02/-3.709111e-03 |
| dense/normuon-lr1e-3/checkpoint-782 | 0/+0.103577/+0.114531/+0.123545/+0.162345 | 0/-0.00390625/-0.00722656/-0.0078125/-0.0078125 | +0.159967/+0.0918381/+0.0725006 | -0.0078125/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -1.035775e-01/-3.906250e-03/+0.000000e+00/+0.000000e+00 |
| dense/normuon-lr1e-3/checkpoint-2345 | 0/+0.0457112/+0.0982143/+0.129308/+0.12299 | 0/-0.00390625/-0.00390625/-0.00390625/-0.0078125 | +0.116788/+0.0890134/+0.0695362 | -0.00390625/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -5.250309e-02/-3.906250e-03/+0.000000e+00/+0.000000e+00 |
| dense/normuon-lr1e-3/checkpoint-3907 | 0/+0.0832271/+0.0965899/+0.0978884/+0.106444 | 0/-0.00390625/-0.00390625/-0.0078125/-0.0078125 | +0.126736/+0.0805851/+0.0723554 | -0.0078125/-0.00390625/-0.00390625 | fail/fail | fail | fail | fail | -8.322711e-02/-3.906250e-03/+0.000000e+00/+0.000000e+00 |

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

#### All 84 held-run predictions

| Held-out run | Task | Transition | Observed | Baseline | Spectrum loss | Spectrum margin | Basis loss | Basis margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adamw-lr1e-5 | ArguAna | stage1-to-2 | -0.011 | -0.0112363 | -0.0113595 | -0.0111639 | -0.0111671 | -0.0110593 |
| adamw-lr1e-5 | ArguAna | stage3-to-4 | -0.00692 | -0.0165838 | -0.015692 | -0.0179416 | -0.0168258 | -0.0192137 |
| adamw-lr1e-5 | ClimateFEVER | stage1-to-2 | +0.02956 | +0.0356363 | +0.0376499 | +0.0350165 | +0.0351608 | +0.0350584 |
| adamw-lr1e-5 | ClimateFEVER | stage3-to-4 | +0.00641 | +0.0302888 | +0.0317688 | +0.029722 | +0.029489 | +0.028791 |
| adamw-lr1e-5 | DBPedia | stage1-to-2 | +0.00745 | +0.00961875 | +0.0113418 | +0.0082327 | +0.00888256 | +0.00649346 |
| adamw-lr1e-5 | DBPedia | stage3-to-4 | +0.0031 | +0.00427125 | +0.00622021 | +0.00293817 | +0.00374946 | +0.00116954 |
| adamw-lr1e-5 | FEVER | stage1-to-2 | +0.00857 | +0.00548125 | +0.00463995 | +0.00587499 | +0.00569012 | +0.0059412 |
| adamw-lr1e-5 | FEVER | stage3-to-4 | +0.00393 | +0.00013375 | -0.000501231 | +0.000975975 | +0.000263973 | +0.00118338 |
| adamw-lr1e-5 | FiQA2018 | stage1-to-2 | -0.0041 | +0.0163962 | +0.0162757 | +0.0170372 | +0.016316 | +0.0173279 |
| adamw-lr1e-5 | FiQA2018 | stage3-to-4 | +0.00514 | +0.0110488 | +0.0110032 | +0.0118415 | +0.0110038 | +0.0123814 |
| adamw-lr1e-5 | HotpotQA | stage1-to-2 | +0.00529 | +0.00639875 | +0.00626554 | +0.00550709 | +0.00639559 | +0.00332063 |
| adamw-lr1e-5 | HotpotQA | stage3-to-4 | +0.00011 | +0.00105125 | +0.00130269 | +0.000706944 | +0.000976807 | +0.000827172 |
| adamw-lr1e-5 | MSMARCO | stage1-to-2 | +0.01656 | +0.00518125 | +0.00853173 | +0.00409183 | +0.00443829 | +0.00328249 |
| adamw-lr1e-5 | MSMARCO | stage3-to-4 | +0.00208 | -0.00016625 | +0.0067044 | -0.00238923 | -0.00183723 | -0.003551 |
| adamw-lr1e-5 | NFCorpus | stage1-to-2 | +0.0018 | +0.00711375 | +0.00955025 | +0.0071367 | +0.00621309 | +0.00639434 |
| adamw-lr1e-5 | NFCorpus | stage3-to-4 | -0.00162 | +0.00176625 | +0.00593401 | +0.000309572 | +0.000557851 | -0.00128828 |
| adamw-lr1e-5 | NQ | stage1-to-2 | -0.01656 | +0.00674625 | -0.0012359 | +0.00775798 | +0.00915676 | +0.00805533 |
| adamw-lr1e-5 | NQ | stage3-to-4 | +0.0142 | +0.00139875 | -0.00481485 | +0.00196906 | +0.00341734 | +0.00141054 |
| adamw-lr1e-5 | QuoraRetrieval | stage1-to-2 | -0.00177 | +0.00156375 | +0.00129426 | +0.00222941 | +0.00163409 | +0.00268414 |
| adamw-lr1e-5 | QuoraRetrieval | stage3-to-4 | -0.00066 | -0.00378375 | -0.003614 | -0.00365839 | -0.00383925 | -0.00452674 |
| adamw-lr1e-5 | SCIDOCS | stage1-to-2 | +0.00228 | +8.125e-05 | -0.000164537 | +0.000499714 | +1.45267e-05 | +0.000494024 |
| adamw-lr1e-5 | SCIDOCS | stage3-to-4 | +0.00244 | -0.00526625 | -0.00659294 | -0.00449818 | -0.00505845 | -0.0042638 |
| adamw-lr1e-5 | SciFact | stage1-to-2 | -0.0104 | -0.00342125 | -0.00226106 | -0.00394212 | -0.00387484 | -0.00489544 |
| adamw-lr1e-5 | SciFact | stage3-to-4 | +0.00536 | -0.00876875 | -0.00810855 | -0.00913778 | -0.00903637 | -0.0100307 |
| adamw-lr1e-5 | TRECCOVID | stage1-to-2 | +0.01977 | +0.0137237 | +0.0167514 | +0.0126343 | +0.0127033 | +0.0119193 |
| adamw-lr1e-5 | TRECCOVID | stage3-to-4 | -0.00084 | +0.00837625 | +0.0100687 | +0.00763643 | +0.0082183 | +0.0082937 |
| adamw-lr1e-5 | Touche2020 | stage1-to-2 | +0.01465 | +0.0328862 | +0.0355676 | +0.0323407 | +0.0319421 | +0.0304922 |
| adamw-lr1e-5 | Touche2020 | stage3-to-4 | -0.00325 | +0.0275388 | +0.0288226 | +0.0267495 | +0.0272757 | +0.025923 |
| muon-lr1e-3 | ArguAna | stage1-to-2 | -0.01739 | -0.00953875 | -0.00953385 | -0.00938562 | -0.00954026 | -0.0095409 |
| muon-lr1e-3 | ArguAna | stage3-to-4 | -0.01019 | -0.0134513 | -0.0135595 | -0.0136347 | -0.013455 | -0.0134792 |
| muon-lr1e-3 | ClimateFEVER | stage1-to-2 | +0.04431 | +0.0270788 | +0.0268511 | +0.0267319 | +0.0270487 | +0.0270208 |
| muon-lr1e-3 | ClimateFEVER | stage3-to-4 | +0.02302 | +0.0231663 | +0.0223688 | +0.0229677 | +0.0231374 | +0.0231512 |
| muon-lr1e-3 | DBPedia | stage1-to-2 | +0.01544 | +0.00750375 | +0.00717946 | +0.00736902 | +0.00747728 | +0.00742427 |
| muon-lr1e-3 | DBPedia | stage3-to-4 | +0.0007 | +0.00359125 | +0.0031447 | +0.00330177 | +0.00356863 | +0.00352895 |
| muon-lr1e-3 | FEVER | stage1-to-2 | +0.00428 | +0.00632875 | +0.00680979 | +0.00689094 | +0.00633896 | +0.00637601 |
| muon-lr1e-3 | FEVER | stage3-to-4 | +0.00196 | +0.00241625 | +0.00214689 | +0.00246008 | +0.00239347 | +0.00237758 |
| muon-lr1e-3 | FiQA2018 | stage1-to-2 | +0.01882 | +0.0116337 | +0.0115046 | +0.0112869 | +0.0116249 | +0.0115435 |
| muon-lr1e-3 | FiQA2018 | stage3-to-4 | -0.0016 | +0.00772125 | +0.00791558 | +0.0083711 | +0.0077298 | +0.00781147 |
| muon-lr1e-3 | HotpotQA | stage1-to-2 | +0.00654 | +0.00516625 | +0.00519523 | +0.00530423 | +0.00516627 | +0.00517914 |
| muon-lr1e-3 | HotpotQA | stage3-to-4 | +0.00092 | +0.00125375 | +0.00129791 | +0.00105517 | +0.00125434 | +0.00122368 |
| muon-lr1e-3 | MSMARCO | stage1-to-2 | +0.00224 | +0.00745875 | +0.00748076 | +0.00752855 | +0.00748466 | +0.00750708 |
| muon-lr1e-3 | MSMARCO | stage3-to-4 | +0.00442 | +0.00354625 | +0.00029768 | +0.00227957 | +0.00341559 | +0.00338407 |
| muon-lr1e-3 | NFCorpus | stage1-to-2 | +0.01054 | +0.00274875 | +0.00188288 | +0.00237919 | +0.00271068 | +0.00271545 |
| muon-lr1e-3 | NFCorpus | stage3-to-4 | +0.00423 | -0.00116375 | -0.00215986 | -0.00150626 | -0.00118667 | -0.00119705 |
| muon-lr1e-3 | NQ | stage1-to-2 | +0.00908 | +0.00383625 | +0.00546594 | +0.00382272 | +0.00394302 | +0.00391358 |
| muon-lr1e-3 | NQ | stage3-to-4 | -0.00267 | -7.625e-05 | +0.00319691 | +0.00114931 | +8.40367e-05 | +9.55984e-05 |
| muon-lr1e-3 | QuoraRetrieval | stage1-to-2 | -0.0002 | +0.00049875 | +0.000501363 | +0.000894284 | +0.0004984 | +0.000546008 |
| muon-lr1e-3 | QuoraRetrieval | stage3-to-4 | -0.00084 | -0.00341375 | -0.00340546 | -0.00402139 | -0.00341426 | -0.00352975 |
| muon-lr1e-3 | SCIDOCS | stage1-to-2 | -0.00965 | +0.00258375 | +0.00237553 | +0.00258537 | +0.00259178 | +0.00261168 |
| muon-lr1e-3 | SCIDOCS | stage3-to-4 | +0.00149 | -0.00132875 | -0.00125103 | -0.00105767 | -0.0013363 | -0.00131801 |
| muon-lr1e-3 | SciFact | stage1-to-2 | -0.01381 | -0.00461375 | -0.00495469 | -0.00489998 | -0.00464258 | -0.00468034 |
| muon-lr1e-3 | SciFact | stage3-to-4 | +0.01067 | -0.00852625 | -0.00865664 | -0.00860363 | -0.0085347 | -0.00854129 |
| muon-lr1e-3 | TRECCOVID | stage1-to-2 | +0.03216 | +0.0117687 | +0.011497 | +0.0117401 | +0.0117428 | +0.0117516 |
| muon-lr1e-3 | TRECCOVID | stage3-to-4 | -0.00828 | +0.00785625 | +0.00732296 | +0.00755162 | +0.00782096 | +0.0078047 |
| muon-lr1e-3 | Touche2020 | stage1-to-2 | +0.03127 | +0.0179513 | +0.018067 | +0.0176726 | +0.0179591 | +0.017904 |
| muon-lr1e-3 | Touche2020 | stage3-to-4 | +0.037 | +0.0140388 | +0.0144383 | +0.0141811 | +0.0140439 | +0.0140302 |
| normuon-lr1e-3 | ArguAna | stage1-to-2 | -0.02659 | -0.0094925 | -0.00949336 | -0.00917963 | -0.00944865 | -0.00892902 |
| normuon-lr1e-3 | ArguAna | stage3-to-4 | -0.00147 | -0.0132575 | -0.0132586 | -0.0119049 | -0.0132169 | -0.0122718 |
| normuon-lr1e-3 | ClimateFEVER | stage1-to-2 | +0.05218 | +0.0277075 | +0.0277101 | +0.0275207 | +0.0278237 | +0.0278633 |
| normuon-lr1e-3 | ClimateFEVER | stage3-to-4 | +0.01234 | +0.0239425 | +0.0239345 | +0.024529 | +0.0242672 | +0.0241944 |
| normuon-lr1e-3 | DBPedia | stage1-to-2 | +0.01333 | +0.008555 | +0.00854883 | +0.0104667 | +0.00894278 | +0.0105862 |
| normuon-lr1e-3 | DBPedia | stage3-to-4 | -0.00169 | +0.00479 | +0.00478861 | +0.00560965 | +0.00491995 | +0.00544958 |
| normuon-lr1e-3 | FEVER | stage1-to-2 | +0.00479 | +0.0065675 | +0.00657213 | +0.00548137 | +0.00614462 | +0.00423639 |
| normuon-lr1e-3 | FEVER | stage3-to-4 | +0.0002 | +0.0028025 | +0.00280152 | +0.00355553 | +0.00285046 | +0.00399208 |
| normuon-lr1e-3 | FiQA2018 | stage1-to-2 | +0.04251 | +0.0064475 | +0.00644768 | +0.00562785 | +0.00652648 | +0.00558407 |
| normuon-lr1e-3 | FiQA2018 | stage3-to-4 | -0.00484 | +0.0026825 | +0.00268248 | +0.00223639 | +0.00269667 | +0.00158902 |
| normuon-lr1e-3 | HotpotQA | stage1-to-2 | +0.00799 | +0.0050975 | +0.00509678 | +0.00581009 | +0.00516096 | +0.00639483 |
| normuon-lr1e-3 | HotpotQA | stage3-to-4 | -0.00055 | +0.0013325 | +0.00133237 | +0.00215215 | +0.00133418 | +0.00256285 |
| normuon-lr1e-3 | MSMARCO | stage1-to-2 | +0.00166 | +0.0082075 | +0.00820806 | +0.00803738 | +0.00809693 | +0.00826137 |
| normuon-lr1e-3 | MSMARCO | stage3-to-4 | +0.00171 | +0.0044425 | +0.00442759 | +0.00644464 | +0.00545032 | +0.00679399 |
| normuon-lr1e-3 | NFCorpus | stage1-to-2 | +0.00216 | +0.00562 | +0.00561213 | +0.006216 | +0.00634369 | +0.00706002 |
| normuon-lr1e-3 | NFCorpus | stage3-to-4 | +0.00083 | +0.001855 | +0.00185222 | +0.0016254 | +0.00219372 | +0.00241266 |
| normuon-lr1e-3 | NQ | stage1-to-2 | +0.00988 | +0.002895 | +0.0028967 | +0.0043737 | +0.00284477 | +0.00594539 |
| normuon-lr1e-3 | NQ | stage3-to-4 | 0 | -0.00087 | -0.000854382 | -0.00181575 | -0.00178318 | -0.000332724 |
| normuon-lr1e-3 | QuoraRetrieval | stage1-to-2 | -0.00211 | +0.001015 | +0.00101484 | -0.000737318 | +0.00104615 | -0.000337655 |
| normuon-lr1e-3 | QuoraRetrieval | stage3-to-4 | -0.00129 | -0.00275 | -0.00274958 | -0.00252992 | -0.00280039 | -0.0030281 |
| normuon-lr1e-3 | SCIDOCS | stage1-to-2 | -0.00192 | +0.0010225 | +0.00102496 | +0.000535943 | +0.00120282 | +0.000892913 |
| normuon-lr1e-3 | SCIDOCS | stage3-to-4 | -0.00029 | -0.0027425 | -0.00273887 | -0.00325523 | -0.00302672 | -0.00310214 |
| normuon-lr1e-3 | SciFact | stage1-to-2 | -0.02685 | -0.0001625 | -0.000166004 | +0.000383538 | -3.22127e-05 | +0.000523292 |
| normuon-lr1e-3 | SciFact | stage3-to-4 | +0.00561 | -0.0039275 | -0.00392719 | -0.00407382 | -0.00384424 | -0.00379791 |
| normuon-lr1e-3 | TRECCOVID | stage1-to-2 | +0.01492 | +0.012585 | +0.0125801 | +0.0133309 | +0.0127958 | +0.0129446 |
| normuon-lr1e-3 | TRECCOVID | stage3-to-4 | +0.0054 | +0.00882 | +0.008814 | +0.0098062 | +0.00895504 | +0.00878651 |
| normuon-lr1e-3 | Touche2020 | stage1-to-2 | +0.02676 | +0.0218 | +0.0217901 | +0.0228623 | +0.0224503 | +0.0235254 |
| normuon-lr1e-3 | Touche2020 | stage3-to-4 | +0.02582 | +0.018035 | +0.0180284 | +0.0184716 | +0.0184295 | +0.018715 |

> Temporal boundary: The shared-start randomization identifies optimizer-level accumulated effects. The post-treatment spectral predictor analysis is a small-sample, falsifiable causal-chain triangulation, not a formally identified causal mediation estimate.

> Dose/bridge boundary: The transplant randomizes spectral components at fixed weights and can identify immediate functional effects. Its task-aligned forward prediction is out-of-run evidence for a causal-chain bridge, but it is not a trained spectral-operator intervention and cannot by itself identify formal mediation of final BEIR gains.
