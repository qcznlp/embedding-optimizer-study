# Scientific prose and citation readout

Reviewed 2026-09-12 against the genuine development manuscript
`reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/paper-preview-v2/main.tex`
(SHA-256 `36a6b3509d1ad71a0c9609832e37869bde97379daf34bf69832af0f68f7f95fe`)
and the unchanged bibliography (SHA-256
`fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218`).
This is a review and deferred prose candidate, not an installed paper revision.
No live document-consumer input is changed. The full paper remains dependent
on complete continuation outcomes and final reconstruction/review.

## Reference scope

All 14 cited keys resolve locally; two unused bibliography entries, ColBERT and
Singh's basis paper, were not part of this claim check. The following are primary
sources, inspected at the stated scope. Abstract-level checks support narrow
related-work summaries, not independent verification of another paper's results.

| Key | Primary source and inspected scope | Supported use / boundary |
| --- | --- | --- |
| `loshchilov2019decoupled` | [AdamW arXiv record](https://arxiv.org/abs/1711.05101), abstract and ICLR 2019 metadata | Decoupled weight decay; does not establish the prevalence of AdamW in embedding adaptation. The OpenReview page presented a browser challenge, so the public author manuscript was used. |
| `jordan2024muon` | [Author's Muon article](https://kellerjordan.github.io/posts/muon/), algorithm, empirical scope and official citation | Matrix update/auxiliary AdamW description. The official citation identifies Jiacheng You: existing `Jiacheng, You` must become `You, Jiacheng`. |
| `liu2025muon` | [Muon scalability paper](https://arxiv.org/abs/2502.16982), abstract and metadata | Weight decay and update-scale calibration in LLM training; not evidence of dense-retrieval superiority. |
| `li2025normuon` | [NorMuon](https://arxiv.org/abs/2510.05491) and [algorithm text](https://arxiv.org/html/2510.05491v1) | Row/neuron historical second moments after approximate orthogonalization; an intrinsic rule, not a new finding of this study. |
| `qu2026finetune` | [Muon fine-tuning paper](https://arxiv.org/abs/2605.10468), abstract and metadata | Optimizer-switching mismatch and update-strength risks in its tested settings; not a prediction that AdamW wins here. |
| `liu2026consistency` | [Optimizer-model consistency](https://arxiv.org/abs/2605.06654), abstract and metadata | Learning/forgetting trade-offs under optimizer consistency in its settings; not a dense-retrieval result. |
| `pang2026htmuon` | [HTMuon](https://aclanthology.org/2026.findings-acl.1819/), abstract and publication metadata | Heavy-tail/noise-direction motivation for a spectral correction; the present study neither implements nor compares HTMuon. |
| `du2026newtonmuon` | [Newton--Muon](https://arxiv.org/abs/2604.01472), abstract and metadata | Input-second-moment/covariance-aware update motivation; no claim of retrieval transfer from this citation. |
| `massena2026schatten` | [Adaptive Schatten-p optimization](https://arxiv.org/abs/2605.19781), abstract and metadata | Layer-dependent geometry adaptation; not evidence for the current frozen feature bridge. |
| `ruan2026features` | [Robust and transferable features](https://arxiv.org/abs/2606.09658), abstract and metadata | Pretrained-feature robustness, transfer and effective-rank findings, with their pretraining scope retained. |
| `sourty2026denseon` | [DenseOn paper](https://arxiv.org/html/2607.27178v2), training/evaluation appendices and metadata | Table 11 explicitly uses AdamW in pretraining and fine-tuning, and gives 8,192-token supervised context, 768-dimensional dense embeddings and seven sampled mined negatives. Our reduced data/objective is not represented as the complete original distillation/Matryoshka recipe. |
| `thakur2021beir` | [BEIR author manuscript](https://arxiv.org/abs/2104.08663), abstract and NeurIPS 2021 metadata | Heterogeneous zero-shot retrieval benchmark; DenseOn's separate decontaminated suite and this study's frozen task manifest determine our 14-task population. The OpenReview page presented a browser challenge. |
| `kim2024hil` | [HIL](https://aclanthology.org/2024.naacl-long.437/), abstract and publication metadata | Hybrid isotropic/anisotropic representations in ColBERT; not a rank-only explanation and not an architecture evaluated here. |
| `takeshita2025dimensions` | [EMNLP paper](https://aclanthology.org/2025.emnlp-main.1410/) and [PDF](https://aclanthology.org/2025.emnlp-main.1410.pdf), Sections 2--4 | Random-removal robustness and individual coordinate ablations; isotropy/correlation/outlier comparisons. Its single-coordinate study uses downstream performance, while our primary readout is task-mean eight-candidate cosine margin. Neither should silently substitute for the other. |

No exhaustive novelty search was performed. In particular, the scale-invariance
identity in the separate cosine analysis must not be advertised as a novel
optimization theorem or a property unique to Muon.

## Deferred corrections

1. Correct the Muon contributor's family/given-name order in the bibliography.
2. Replace the unsupported prevalence claim in the introduction with the
   directly evidenced fact that DenseOn's published adaptation recipe uses
   AdamW. Retain the broader scientific motivation without a field-wide count.
3. Explain that all-rate averaging and validation-selected comparisons answer
   different questions. Neither estimand dominates the other by definition.
   The original frozen grids, selection and statistical decisions remain intact.
4. Disclose that the selected AdamW rate is the upper tested boundary; the
   experiment does not establish where AdamW's optimum lies. This limits claims
   about exhaustive tuning, not the validity of the completed pinned-grid result.
5. Review the actual cosine-deletion decomposition before deciding whether it
   clarifies the positive degrading-mass association. Do not insert the new
   interpretation as an optimizer mechanism or hide its post-result status.
6. State explicitly that primary AdamW uses its grid rate for both hidden and
   auxiliary parameters, while Muon-family auxiliary AdamW remains at 3e-6.
   The primary comparison concerns complete optimizer recipes, not an isolated
   hidden-update-rule substitution. This is directly visible in the original
   `build_optimizer` routing, lines 257--336 of the primary source checkout.
   The separate continuation already fixes a common auxiliary rule/rate and
   calibrates initial hidden-update scale; it does not retrospectively remove
   differences between the source training recipes.

The pending continuation results still determine their own scientific conclusion.
Do not pre-author a favorable state/operator/interaction result. Nothing in this
review changes source-state identity, task weighting, marginal/simultaneous
intervals, primary-seed coverage, original evidence or final paper requirements.
