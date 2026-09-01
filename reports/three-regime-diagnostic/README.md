# Training, shortlist validation, and full-corpus retrieval disagree

This is a **post-hoc descriptive diagnostic**. The contrast is the query-disjoint validation-selected `3e-3` run minus the within-optimizer discovery-BEIR oracle at `3e-4`.

| Optimizer | Δ trailing train loss | Δ validation loss | Δ validation margin | Δ BEIR nDCG@10 | Three-regime reversal |
| --- | ---: | ---: | ---: | ---: | --- |
| muon | +0.1210 | -0.3696 | +0.0192 | -0.0315 | yes |
| normuon | +0.1068 | -0.4268 | +0.0204 | -0.0300 | yes |

For both Muon and NorMuon, the larger dose fits the sampled training tuples less well, generalizes better to the held-out eight-way shortlist, and retrieves worse against the complete corpus. This rules out simple training-set memorization as a sufficient explanation, but it does not identify missing-candidate coverage causally.

> Claim boundary: This diagnostic was designed after every discovery loss, validation metric, discovery BEIR score, and confirmatory final-BEIR result was visible. It is a descriptive post-hoc decomposition of the already observed shortlist-corpus gap. It cannot alter recipe selection, confirmatory optimizer inference, or establish that candidate coverage causes the reversal; the separately frozen candidate-breadth intervention is required for that mechanism test.
