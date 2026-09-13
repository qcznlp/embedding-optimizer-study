# Closest-work and contribution review

Reviewed **2026-09-13** while the original full-corpus continuation evaluations
remain live. This is a targeted primary-literature review, not an exhaustive
priority search, a new experiment, or an installed manuscript change. It extends
the earlier fourteen-reference citation check rather than repeating it.

## Material omissions found

The current bibliography omits six directly relevant works. A separate
[bibliography candidate](references-additions.bib) and
[replacement Related Work candidate](related-work-candidate.tex) are provided.
Neither has been installed into the authoritative paper or live document inputs.

| Closest prior evidence | Consequence for this paper |
| --- | --- |
| Wang et al. study Muon through transformer associative-memory components and tail-frequency learning, connecting weight spectra with functional outcomes. [Author manuscript](https://arxiv.org/html/2509.26030v2) | A spectrum-to-learning story already exists. Our layer/rank maps are not by themselves a new explanation; we test transfer to dense-retriever adaptation, not their frequency mechanism. |
| Vasudeva et al. analyze spectral descent under imbalance and compare practical optimizers; their balanced-loss advantage survives a normalized-GD comparator. [Author manuscript](https://arxiv.org/html/2510.22980v3) | Do not reduce prior work to convergence speed or one-step algebra. Our negative predictive controls do not refute their assumptions or class-imbalance results. |
| Dragutinović and Ranganath show that spectral descent can lose GD's simplicity bias and construct shared-representation/spurious-feature disadvantages. [Author manuscript](https://arxiv.org/html/2603.00742v1) | “Broader spectral learning need not be better” is also prior work. Our contribution must be the measured retrieval-specific distinction, not this general warning. Their GD comparison is not a theorem that AdamW wins our task. |
| Wu et al. learn query-conditioned dimension selection from positive/negative embedding signals. [ACL paper](https://aclanthology.org/2026.acl-long.849/) and [method](https://arxiv.org/html/2602.03306v1) | Relevance-sensitive dimension importance is not new. Their query-side selector is different from our paired coordinate deletion, renormalization and task-mean trajectory measurement. We propose no learned selector. |
| Park et al. combine Muon with normalization and embedding-projection changes for outlier-safe pretraining. [ACL paper](https://aclanthology.org/2025.acl-long.618/) | Absence of privileged-coordinate behavior is not our discovery. Their multi-component intervention must not be relabelled as a Muon-only effect or our output-rotation control. |
| Wen et al. control weight/update norms and analyze radial versus angular dynamics for scale-invariant matrix blocks. [Author manuscript](https://arxiv.org/html/2606.16899v1) | Equal numerical LR, equal Frobenius update size, equal angular movement and equal functional perturbation are different controls. Our fixed-probe initial calibration does not match subsequent angular dynamics. |

The existing but previously unchecked `singh2026basis` entry was also inspected:
[the author abstract](https://arxiv.org/abs/2608.05136) separates parameter-gauge
equivariance from coordinate-wise optimization and explicitly says equivariance
alone is insufficient for low-rank recovery. This is related background, not an
explanation of our post-encoding Haar-rotation measurements. We rotate final
query/document vectors; we do not retrain gauge-transformed model parameters.
The candidate Related Work makes this distinction without asserting a new
invariance result.

## What is, and is not, distinct here

The defensible question is **whether an optimizer-associated weight-space
difference actually accounts for a retrieval difference in a controlled
adaptation problem**. The current evidence separates three things that should
not be collapsed: recognizable weight trajectories, native-coordinate
sensitivities, and predictive information beyond the recipe.

1. The complete primary comparison retains twelve recipes, sixty states and
   all fourteen retrieval tasks. This is an empirical comparison of the pinned
   rate grids, not a universal optimizer winner or a first-use-of-Muon claim.
2. A native-coordinate Muon helpful-participation increase is present, but
   changes sign under two of three common rotations. This demonstrates a limit
   of that particular dimension-use interpretation; it does not demonstrate
   that Muon has no functional advantage.
3. Some measurements predict held-dose retrieval beyond the original additive
   comparator, yet none passes all four recipe comparators. This supports a
   recipe-dependence caution, not equivalence or proof that geometry is
   irrelevant. The richer comparisons remain explicitly post-result.
4. The complete state-by-reset-operator continuation will answer a different
   conditional question about the fixed source pair. Its endpoint effects
   remain **pending**, not inferred from incomplete tasks, training loss,
   shortlist probes, or the primary selected cells. It cannot establish a
   geometric mediator because no geometric feature is independently manipulated.

These statements follow the actual
[functional readout](../../dense-v3-functional-inference-v1/README.md),
[recipe sensitivity](../../dense-v3-functional-sensitivity-v1/README.md),
and the existing complete primary development paper. No prior-paper result is
used as an observed value in this study. One primary seed, one backbone and
bounded LR ranges remain substantive limits on generality; three continuation
order seeds do not repair primary-source replication. This review does not
declare the present evidence sufficient for NAACL acceptance.

## Important distinction in the coordinate score

Wu et al.'s oracle uses the same algebraic product form
`q_j * (p_j - n_j)`, with a weighted positive centroid and averaged sampled
negatives. Our full-margin identity uses a fixed strongest competitor, while
our actual attribution deletes the coordinate from both vectors, renormalizes,
and allows the strongest competitor to change. Thus the shared product is
prior art, but their score is not identical to our finite deletion statistic.
Our separate cosine-sensitivity result is measurement interpretation, not a
new optimizer-specific theorem. The ACL camera-ready Section 3.1 and Figure 1
retain this construction. [Wu et al., paper](https://aclanthology.org/2026.acl-long.849.pdf)

## Search and scope discipline

The targeted search covered Muon with dense retrieval, sentence embeddings,
contrastive learning, effective rank, associative memory, imbalance and
privileged bases; exact-title follow-ups resolved the papers above. Only
author manuscripts and official proceedings support these conclusions.
OpenReview's live Wang page presented a browser challenge; the public arXiv
manuscript supplied the actual reading. A first ACL author-page click failed;
the direct linked paper was then resolved without a content-access bypass.
No source code or third-party result was executed, and no new library installed.

Additional scope checks found Muon already used in CT/text representation
learning with retrieval evaluation ([SigVLP](https://arxiv.org/abs/2602.21735));
this alone rules out a broad first-use-in-embedding claim, not a narrowly defined
DenseOn comparison. Separate softmax associative-memory theory also exists
([Li et al.](https://arxiv.org/abs/2602.05725)). IMRNNs instead concerns inference
adapters ([official abstract](https://aclanthology.org/2026.findings-eacl.333/));
it is not a matched optimizer experiment. These three were abstract-level scope
checks, not full-method reviews or proposed additional experiment baselines.

The search also surfaced this project's historical public pages; they were
excluded as independent literature and do not prove that current local work
is published. Not finding an exact duplicate does not establish priority.

## Integration status

The new six-entry bibliography and complete Related Work text are **deferred
candidates only**. They preserve the original cited work, add the closest
missing comparisons, and avoid fabricated favorable continuation findings.
The original native document and combined replay must finish with their bound
inputs unchanged. After that, combine this candidate with the prior
[five specific prose fixes](../dense-v3-cosine-sensitivity-v1/deferred-prose-candidate.json),
then run a new full numerical/document build and visual review. Neither wording
file is a new scientific protocol, a final-paper claim or a release gate waiver.
No additional training is requested by this review.

## Actual candidate check and handoff

CPU session **24545**, terminal **2c1ef8**, exited zero at **00:26:20 UTC**.
All four actual pdflatex/BibTeX commands exited zero. Six new bibliography keys
close with the existing file to resolve all **19** citations in the candidate;
nineteen entries render. The two-page
[standalone preview](citation-check/snippet.pdf) was visually inspected, with
four embedded Type-1 fonts and no overfull boxes or undefined citations.
This checks the candidate prose and bibliography only, not complete-paper page
fit, all scientific claims, complete numerical reconstruction or publication.
The original main, development main and bibliography hashes are unchanged.
The initial file-patch command had a patch-syntax refusal before any write;
the corrected authored files are the versions actually compiled. The two
protected manuscript versions were not modified or rebuilt.

The [input record](actual/citation-inputs.json),
[actual completion](actual/citation-completed.json), full compiler inputs/logs,
both page images and pre-edit handoffs are retained. No new source release,
GitHub write, HF write, scientific source change, or numerical recomputation
occurred. Existing actual publication sources remain bound unchanged.

At **00:28:11 UTC**, the exact BEIR observer reports **10/168** tasks, both
coordinators live/S and eight exact workers live/R, without failure. At
**00:27:10 UTC**, all six downstream CPU waiters are live/S without actual
collectors, complete native paper/replay, outcome upload or GPU-resume ranks.
All observations are in `actual/*-final.json`. They are verified waits, not
new completed experiments or an estimate of remaining wall-clock percentage.
Continue those original entries; do not duplicate them or repeat the completed
literature/citation check as missing work. This goal turn is **PROGRESS** in
scientific positioning, with verified waiting for the remaining full outcomes.
