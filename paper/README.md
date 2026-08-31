# NAACL manuscript

This directory contains the result-safe DenseOn manuscript for the AdamW/Muon/NorMuon optimizer
study. It uses the official ACL style files pinned to commit
`d5adc823ff0f80f98c80405ca0ab66c68e684409` of
[`acl-org/acl-style-files`](https://github.com/acl-org/acl-style-files). The style files are fetched
into an ignored local directory so the repository does not silently fork the conference template.

Build the review-format PDF with:

```bash
cd paper
make
```

## Scope amendment

The project originally included DenseOn and LateOn. After the original discovery runs and some
exploratory mechanism outputs were visible, the user directing the project requested that future
work focus on DenseOn because LateOn was substantially slower and less central to the intended paper
audience. This is a **user-directed, post-hoc scope amendment**, not a preregistered exclusion.

The main paper therefore makes primary, causal, and confirmatory claims about DenseOn only. LateOn
artifacts remain in the repository as historical exploratory material, but they are not pooled,
treated as replication, used to estimate architecture interactions, or allowed to determine headline
wording. The authoritative dated record is `../configs/dense_scope_amendment.json`; the manuscript
plan is `../docs/naacl-dense-paper-plan.md`. The original
`../docs/naacl-paper-plan.md` remains byte-for-byte frozen as a historical two-family protocol
artifact.

Legacy manifests outside the manuscript may still contain two model panels for provenance. Every
generated manuscript table and placeholder is strictly DenseOn-only. The final paper audit must
reject any headline, confidence interval, or generated table that includes LateOn.

## Result safety

`results.tex` is the only checked-in source of final numerical result macros. A
`\ResultPending{...}` marker denotes an unresolved evidence gate and renders visibly in red. The
paper is not submission-ready while any marker remains.

The central result sequence is deliberately gated:

1. same-state, Frobenius-matched Muon-family steps can be worse than AdamW on mean immediate margin;
2. the DenseOn discovery trajectories can nevertheless have better final BEIR point estimates;
3. symmetric cross-tails show redistribution of fragile queries rather than uniform dominance;
4. a post-hoc cosine synthesis motivates optimizer-induced state feedback but is not causal;
5. three-seed shared-start branches test whether the advantage accumulates;
6. hybrid AdamW tests whether parameter routing is sufficient; and
7. three new negative-sampling seeds determine the final retrieval claim.

Exact state-feedback cosines, shared-start outcomes, hybrid results, confirmatory intervals, and the
temporal and dose/band causal-chain estimates must enter through audited macros or generated tables.
A supported/negative verdict alone is insufficient: the paper must display the numerical predictor
and negative-control errors, treatment-coefficient changes, anchor support counts, and forward-bridge
errors that determine each verdict. Do not copy them directly into prose. Spectral flattening and row
adaptation are implementation fingerprints, not standalone paper contributions.

The claim protocol and its content-hashed amendments bind experiment and intervention protocols to
headline decision rules. If a bound source receives a factual documentation correction, the original
freeze context and the evidence visible at amendment time must remain recorded. An unrecorded source
change is a hard audit failure.

Run `embed-optim-audit-paper` during drafting. It reports unresolved evidence without hiding it.
`embed-optim-audit-paper --strict` is the final submission gate. After every DenseOn evidence report
exists, run `embed-optim-render-paper-results`; it replaces only authorized headline macros and table
files, writes `reports/paper-results.manifest.json`, and binds rendered bytes to the claim protocol,
evidence manifests, and source tables.

`make -C paper` invokes the renderer with `--if-ready`: incomplete experiments keep audited red
placeholders, while complete evidence can be rendered before LaTeX compilation. This preserves a
buildable developer draft. `make -C paper release` instead cleans the build, runs the renderer without
`--if-ready`, and builds the PDF only from complete evidence. The release finalizer then repeats the
strict paper audit after the PDF build and before constructing either distribution.

The pending tables are topology-faithful: they reserve the same main/appendix float labels and row
cardinalities as the final Dense-only renderer. The pending headline and conclusion macros are also
final-shaped layout fixtures: their prose and numeric tokens reserve the deterministic renderer's
full result footprint rather than using abbreviated status text. Main text retains six headline
floats; systems, per-task, representation, basis, tail, and full causal diagnostics are placed in the
appendix. Every developer build runs the layout gate. It audits every classified float label, not
only the conclusion page, so a deferred main-text float after the audited endpoint or on page 9
cannot be hidden by an end label that remained within the limit.

## DenseOn evidence contract

The final renderer/auditor may read legacy two-model manifests, but each primary gate must select and
verify the DenseOn subset explicitly.

| Claim family | Authoritative evidence | DenseOn final gate |
|---|---|---|
| Discovery training and systems behavior | canonical Trainer/W&B histories, completion records, and manifests under `reports/training-dynamics/` | 12 DenseOn runs and 60 checkpoints; every history, terminal record, systems summary, and source-bound figure passes audit |
| Discovery retrieval behavior | [`reports/dense-discovery/coverage.json`](../reports/dense-discovery/coverage.json), strict aggregate tables, and plot sidecars | exactly 840 DenseOn checkpoint--task cells across 14 decontaminated-BEIR tasks |
| Discovery time-to-quality | `configs/retrieval_dynamics_protocol.json` and `reports/retrieval-dynamics-dense/summary_manifest.json` | 60 DenseOn checkpoint means and all 840 source task files pass hashes; right-censored AdamW-median rule and post-hoc timing are disclosed |
| Integrated weight trajectories | `reports/weight-space/summary_manifest.json` | 12 DenseOn runs and 60 checkpoints with verified model inputs |
| Common-state update geometry | `reports/common-state/summary_manifest.json` and exact-spectrum manifest | every frozen DenseOn anchor, gradient replay, transform, and spectrum passes Cartesian and source-hash audits |
| Immediate causal intervention | `reports/functional-intervention/manifest.json` | every DenseOn anchor and paired query record passes scale, sign, pairing, and source audits |
| Post-hoc state-feedback synthesis | common-state cosine summary and `reports/local-global-reversal/summary_manifest.json` | trajectory-conditioned AdamW--Muon, Muon--NorMuon, and terminal-gradient alignments are source-bound and labeled post hoc |
| Post-hoc symmetric tail diagnostic | `configs/tail_stability_analysis.json` and `reports/tail-stability/summary_manifest.json` | fixed DenseOn quantiles and Adam/challenger cross-tails are labeled post hoc; no robustness claim without the accumulated gate |
| Spectrum/basis attribution | `configs/spectral_transplant_intervention.json` and `reports/spectral-transplant/summary_manifest.json` | complete DenseOn spectrum, basis, interaction, path, and band cells; this fixed-state tier is insufficient for long-horizon causality |
| Accumulated causal branch | `reports/short-branch/summary_manifest.json` | 9 DenseOn runs and 45 checkpoints on both frozen probes; three-seed joint loss-p95/unseen-margin-p05 endpoint |
| Temporal causal-chain bridge | `reports/temporal-short-branch/summary_manifest.json`, `paired_contrasts.csv`, `loso_predictions.csv`, and `estimates.csv` in that directory | every frozen predictor and norm control is reported across held-out seeds; numerical RMSE changes and optimizer-coefficient changes are displayed even when the bridge is negative |
| Dose/band causal-chain bridge | `reports/dose-band/summary_manifest.json`, `reports/dose-band/anchor_tests.csv`, and `reports/dose-band/heldout_predictions.csv` | all 10 frozen anchors and 84 forward-retrieval rows pass audit; numerical dose, band, basis-control, and held-out RMSE results are displayed rather than reduced to a verdict |
| Routing fairness | `reports/hybrid-adamw/summary_manifest.json` | all 4 DenseOn hybrid-AdamW learning-rate runs and 56 final BEIR cells |
| Confirmatory retrieval claims | `reports/confirmatory/summary_manifest.json` | 3 seeds × 3 optimizers = 9 DenseOn runs and 126 final BEIR cells; two primary contrasts receive nominal intervals and the original six-comparison Bonferroni familywise intervals |
| Final outcome rendering | `reports/outcome-summary.manifest.json` and `reports/paper-results.manifest.json` | all DenseOn tables/headlines are source-hashed; no unresolved marker or LateOn contribution to primary inference |

Checkpoint correlations are observational. Causal language is reserved for fixed-state interventions
and shared-start branches. The exploratory training view is never pooled with the three confirmation
views as though all four were prospectively sampled.

## Historical LateOn artifacts

Do not delete or rewrite LateOn logs to make the narrowed study look prospective. Preserve their
hashes and provenance, label them historical/exploratory, and keep every LateOn table in the
repository archive outside the manuscript. The paper appendix may describe and link to that archive,
but every generated appendix table remains strictly DenseOn-only and the manuscript must not make
MaxSim, token-level, cross-architecture, or general late-interaction claims.
