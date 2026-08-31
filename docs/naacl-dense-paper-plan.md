# From Local Update Disadvantage to Global Retrieval Differences

## Scope and disclosure

The NAACL paper is now a **DenseOn-only** study of AdamW, Muon, and NorMuon. The project originally
included DenseOn and LateOn. After the original discovery runs and some exploratory mechanism
outputs were visible, the user directing the project requested that subsequent work focus on dense
retrieval because LateOn was substantially slower and less central to the intended audience.

This is a **user-directed, post-hoc scope amendment**, not a preregistered scientific exclusion.
Consequently:

- primary, causal, and confirmatory claims concern DenseOn only;
- LateOn is not pooled with DenseOn, treated as a replication, or used to estimate an architecture
  interaction;
- no MaxSim, token-utilization, or late-interaction mechanism appears in the main argument;
- completed LateOn configurations, checkpoints, logs, and tables remain auditable as historical
  exploratory material;
- the amendment is disclosed in the abstract, methods, limitations, paper README, and appendix; and
- results visible before the amendment cannot be described as preregistered evidence for the new
  scope.

The dated, content-hashed amendment in `configs/dense_scope_amendment.json` is the authoritative
scope record. It narrows future inference; it does not erase prior work.

## One-sentence thesis

> In the one-seed discovery grid, Muon-family matched steps are locally weaker than AdamW while their
> four-learning-rate median BEIR is higher. Optimizer-induced state feedback and tail redistribution
> are candidate explanations, but whether validation-frozen recipes improve retrieval on new seeds
> remains unresolved.

This is the paper's current hypothesis, not an unconditional optimizer claim. The shared-start
endpoint decides the accumulation account, and the validation-frozen three-seed BEIR comparison
decides the recipe outcome. No discovery-oracle result can upgrade either conclusion.

## Why this is new

Muon spectral equalization and NorMuon row-wise adaptation are properties of the optimizers. Showing
them again during embedding training is implementation validation, not the contribution. The
retrieval-specific novelty is the conjunction of seven observations and tests:

1. **Same-state local disadvantage.** Under common weights, common gradient histories, and
   Frobenius-matched update budgets, Muon-family directions need not beat AdamW in mean immediate
   query-margin improvement.
2. **Final discovery gain.** When the discovery BEIR suite itself selects one learning rate per
   optimizer, Muon-family trajectories finish slightly above the likewise BEIR-selected AdamW point
   estimate. This is an exploratory same-suite oracle comparison, not the validation-selected result:
   the frozen query-disjoint validation rule instead selects 3e-3 for both Muon variants, whose
   discovery-BEIR scores are below AdamW.
3. **Tail redistribution.** Muon/NorMuon can improve AdamW's worst-query set while AdamW performs
   better on each challenger's own worst set; low tail overlap means the operator changes which
   queries are fragile rather than uniformly shrinking a shared tail.
4. **Optimizer-induced state feedback.** In the post-hoc common-state synthesis, AdamW--Muon update
   cosine falls from 0.537 at the pretrained anchor to 0.490 along the AdamW trajectory and about
   0.463 along the Muon trajectory, while Muon--NorMuon remains about 0.971. At final anchors,
   Muon/NorMuon align less with the same terminal gradient (median 0.258/0.244) than AdamW (0.401).
   This is descriptive evidence that committing an optimizer changes the future states and gradients
   on which later updates act; it is not yet a causal accumulation result.
5. **Accumulated adaptation test.** Three-seed shared-start branches ask whether repeated updates turn
   the local redistribution into simultaneous query-disjoint loss-p95 and unseen-margin-p05 gains.
6. **Fair operator attribution.** Hybrid AdamW matches Muon's parameter routing and auxiliary rate;
   three independently resampled training views test the validation-frozen BEIR contrasts on new
   seeds.
7. **A falsifiable spectrum-to-retrieval bridge.** Before any shared-start or spectral-transplant
   outcome exists, freeze two tests that can reject the proposed explanation: early
   adjacent-update spectra must predict the final branch tail across held-out seeds beyond optimizer
   labels and norm controls, while a fixed-state spectrum transplant must show a dose response,
   tail-band localization, basis-swap specificity, and held-out prediction of the next same-task
   BEIR change. A familiar Muon spectrum without these downstream tests is explicitly a null
   mechanism result.

The local--global reversal rules out the easy explanation that Muon wins simply because its next
step descends the retrieval objective more effectively. It makes the changing gradient sequence the
scientific object.

## Narrative in five beats

1. **Optimizer switching is risky.** DenseOn begins from an Adam-pretrained solution. Matrix-aware
   conditioning may spread adaptation usefully, or it may disrupt features that support zero-shot
   rankings.
2. **Discovery establishes the tension.** Run all three optimizers over four learning rates and five
   checkpoints on the same 500K-example ledger; report loss, throughput, state memory,
   time-to-quality, and decontaminated BEIR without hiding unstable cells.
3. **A same-state intervention contradicts the naive story.** Muon-family directions have their
   expected geometric fingerprints but do not provide better mean immediate margin improvement at a
   matched step norm.
4. **The query distribution and direction drift suggest state feedback.** The adverse tail is
   redistributed rather than uniformly suppressed, AdamW--Muon direction similarity falls after the
   shared initialization, and Muon/NorMuon remain mutually similar while aligning less with the same
   terminal gradient than AdamW. Repeated updates can therefore change later gradients and the
   identity of hard queries even when the first mean step is weaker. These post-hoc observations
   motivate, but do not prove, optimizer-induced state feedback.
5. **Controls decide the claim.** Shared-start branches test accumulation, the frozen temporal and
   dose/band bridges test whether spectrum actually predicts downstream retrieval behavior, hybrid
   AdamW removes a routing confound, and three new negative-sampling seeds determine the final BEIR
   wording.

## Research questions

### RQ1: Does optimizer choice change dense-retrieval learning dynamics?

Compare AdamW, Muon, and NorMuon across four learning rates and five DenseOn checkpoints. Report
training loss, validation margin, useful wall time, GPU-hours, optimizer-state size, failures,
time-to-quality, and all 14 decontaminated BEIR tasks.

The discovery grid is exploratory. Best BEIR cells cannot support the final optimizer claim.

### RQ2: What update does each optimizer prescribe at the same state?

Freeze DenseOn weights at prespecified anchors and replay the same ordered gradient history through
each optimizer without committing it. Measure update norm, direction angle, stable rank, condition,
row-energy concentration, and layer allocation before and after global or per-tensor scale matching.

Expected spectral and row fingerprints validate the implementations. They are not evidence of
better retrieval unless a controlled functional effect follows.

### RQ3: Why can the local and global rankings disagree?

Apply each same-state direction to the same query batches at matched norms. Report the mean and fixed
loss/margin quantiles, then perform a symmetric cross-tail comparison: every operator is evaluated on
AdamW's worst set and on its own worst set.

The observed DenseOn pattern is tail redistribution. The working explanation is that committing
different first steps changes subsequent gradients and optimizer history, so accumulated training
need not preserve the one-step ordering.

### RQ4: Does the candidate effect survive accumulation?

Branch AdamW, Muon, and NorMuon from the same 60% AdamW checkpoint, match the initial global relative
update budget, train on the same 50K groups under three order seeds, and score all five stages on
query-disjoint and unseen probes.

The frozen tail endpoint requires the three-seed median to improve both query-disjoint loss p95 and
unseen margin p05. Mean-only improvement is insufficient for a robustness claim.

### RQ5: Is the effect the matrix rule or the routing scheme?

Compare native AdamW with hybrid AdamW, which adopts the Muon-family hidden/auxiliary parameter split
and auxiliary learning rate. Then compare hybrid AdamW with Muon under the matched routing scheme.

If hybrid AdamW closes the gap, routing is sufficient; if Muon remains better, the matrix transform
has evidence beyond the routing confound.

### RQ6: Does the DenseOn recipe generalize across new training views?

Select one learning rate per optimizer using only the frozen non-BEIR validation rule. Retrain on
three independently resampled negative views that preserve query and positive identities. Evaluate
final checkpoints on strict decontaminated BEIR and use a seed-by-task hierarchical bootstrap.

No confirmatory result may be stated until all runs, task cells, hashes, and familywise intervals are
complete.

### RQ7: Does a specific spectral behavior explain any accumulated retrieval effect?

Use all five checkpoints from the nine randomized shared-start branches. Extract exact spectra of
each adjacent hidden-matrix displacement, with stage 1 measured from the common start. The primary
early predictor is tail singular-value energy averaged over stages 1 and 2; stable rank, entropy
rank, head/middle energy, and row-norm variation are fully reported secondaries, while total update
and weight norms are negative controls. Test stage-5 loss p95 and unseen-margin p05 using
leave-one-seed-out prediction against an optimizer-label-only baseline.

Separately, interpolate the Muon spectrum into the AdamW basis at fixed doses, transplant frozen
head/middle/tail bands, and swap basis versus spectrum. Require the prespecified local response at
at least eight of ten anchors, then ask whether task-level immediate effects improve held-out-run
prediction of the next same-task BEIR change. This is called **falsifiable causal-chain
triangulation**, not formal causal mediation. Failure of the held-out bridge means that spectral
flattening does not explain the retrieval result, even if its local optimizer fingerprint is strong.

## Identification strategy

Keep four spaces separate:

| Space | Controlled object | Measurements | Valid inference |
|---|---|---|---|
| Operator/update | Same weights and ordered gradient history | update angle, scale, stable rank, condition, row concentration | optimizer fingerprint |
| Immediate function | Same queries after one matched virtual step | loss/margin mean and quantiles, cross-tail membership | local directional effect |
| Accumulated trajectory | Shared start, data, schedule, and initial update budget | path, probe loss/margins, drift, representation geometry | effect of repeated optimizer-specific adaptation |
| Retrieval outcome | Validation-frozen final models | task nDCG@10, seed/task uncertainty, time-to-quality | useful, neutral, or harmful final behavior |

The evidence ladder is:

1. **Descriptive:** complete discovery curves and checkpoint geometry.
2. **Locally causal:** common-state, scale-matched virtual interventions.
3. **Accumulated causal:** shared-start three-seed branches and routing-matched AdamW.
4. **Causal-chain stress test:** frozen early-to-late held-out prediction and fixed-state
   dose/band/basis intervention, with norm and basis negative controls.
5. **Confirmatory outcome:** validation-frozen three-seed BEIR comparison.

Do not call checkpoint correlations a causal mediation analysis. The defensible phrase is
“a mechanistic chain supported by interventions.”

## Experimental backbone

### DenseOn discovery

- Base checkpoint: `lightonai/DenseOn-unsupervised` at the pinned repository revision.
- Training view: 500,000 deterministic queries with one positive and seven explicit sampled hard
  negatives.
- No in-batch or cross-device negatives.
- Context limit: 8,192 tokens, following the DenseOn recipe.
- Objective: cosine InfoNCE at temperature 0.02.
- Optimizers: AdamW, Muon, and NorMuon.
- Learning rates: four per optimizer; all are reported.
- Training: one epoch, 3,907 optimizer steps, five retained stages.
- Evaluation: 14-task decontaminated BEIR, nDCG@10.
- Discovery scope: 12 DenseOn runs, 60 checkpoints, and 840 checkpoint--task cells.

The original dual-model aggregate has 24 runs, 120 checkpoints, and 1,680 cells. Those larger totals
may remain in legacy manifests, but must never be used to describe the Dense-only primary sample.

### Recipe selection

Use the frozen 4,096-query validation set, not BEIR:

1. minimize eight-way validation loss;
2. maximize positive margin as a tie-breaker; and
3. prefer the smaller learning rate as the final tie-breaker.

Discovery BEIR oracles remain descriptive and should be shown only to quantify selection regret.

### Confirmation

- Three new negative-resampling seeds.
- Same query and positive identities across optimizers within each seed.
- AdamW--Muon and AdamW--NorMuon are the two primary contrasts.
- Deterministic 20,000-draw hierarchical bootstrap over seeds and tasks.
- Nominal and familywise 95% intervals; the latter retain the original six-comparison Bonferroni
  family frozen before the post-hoc Dense-only scope amendment.
- Final language is positive, negative, or inconclusive according to the frozen familywise rule.

### Shared-start accumulation

- Start: the same 60% AdamW DenseOn checkpoint.
- Budget: learning rates calibrated to a common initial global
  `||ΔW||_F / ||W||_F` target.
- Data: the same 50K groups under three order seeds.
- Outputs: all five stages on query-disjoint and unseen probes.
- Primary tail endpoint: joint loss-p95 decrease and unseen-margin-p05 increase under the three-seed
  median.

### Routing control

Hybrid AdamW must use exactly the Muon-family parameter partition and auxiliary AdamW learning rate.
The main contrasts are:

1. native AdamW versus hybrid AdamW: routing and auxiliary-rate effect;
2. hybrid AdamW versus Muon: orthogonalized hidden-matrix rule under matched routing; and
3. Muon versus NorMuon: additional row-wise adaptation.

### Falsifiable spectrum-to-retrieval bridge

The authoritative lock is `configs/causal_chain_analysis.json`. It was written after exploratory
geometry and tail redistribution were visible, but before any randomized branch,
spectral-transplant, or confirmatory outcome existed. It fixes:

- 45 adjacent-displacement spectra from nine runs and five checkpoints;
- stages 1--2 as predictors and stage 5 as the outcome;
- tail energy as the only primary spectral mediator, with all secondary and control metrics
  reported;
- leave-one-seed-out prediction versus an optimizer-label baseline;
- dose values 0/.25/.5/.75/1, the tail band, the basis-swap negative control, and an 8/10 anchor
  threshold; and
- 84 task-aligned forward observations from stage 1 to 2 and stage 3 to 4, evaluated by
  leave-one-source-run-out prediction.

The analysis may finish with `supported=false`; that is a complete, claimable negative mechanism
result. Missing coverage is instead `pending-not-claimable` and blocks finalization.

## Result-safe claims

| Claim | Current status | Required final evidence |
|---|---|---|
| Muon-family directions have distinct spectra/row allocation | Completed operator fingerprint | audited common-state matrix |
| Muon-family matched steps are weaker on mean immediate margin | Completed exploratory/local result | audited fixed-state intervention; wording remains local |
| DenseOn exhibits worst-tail redistribution | Completed post-hoc diagnostic | audited symmetric cross-tail table; label post hoc |
| Trajectory-dependent direction drift supports optimizer-induced state feedback | Completed post-hoc synthesis | audited common-state cosine summary; causal wording remains prohibited |
| Muon/NorMuon have higher one-seed four-LR-median and same-suite BEIR-selected discovery point estimates | Completed exploratory observation | disclose both aggregation and test selection; neither is validation-selected |
| Tail redistribution becomes accumulated robustness | Pending | frozen three-seed shared-start joint endpoint |
| Matrix transform, not routing, causes the accumulated effect | Pending | complete hybrid AdamW control and shared-start comparison |
| Muon or NorMuon is a better DenseOn recipe | Pending | complete three-seed strict BEIR matrix and familywise interval |
| Early spectral redistribution predicts accumulated tail behavior | Pending/conditional | frozen 45-checkpoint temporal analysis must beat optimizer labels and both norm controls across held-out seeds |
| A singular-value component explains later retrieval change | Pending/conditional | frozen dose, tail-band, basis-control, and held-out same-task BEIR bridge must all pass; otherwise report a local-only or negative mechanism result |

Do not replace any pending claim with a plausible qualitative sentence. Retain `\ResultPending`
macros until the corresponding strict renderer succeeds.

## Decision rules for the final story

### Strong positive story

Use only if all of the following hold:

- one Muon-family optimizer has a positive familywise three-seed BEIR interval versus AdamW;
- hybrid AdamW does not explain the full gain;
- the shared-start branch passes its joint accumulated tail endpoint;
- the temporal and dose/band bridges pass their frozen negative controls and held-out
  predictions; and
- the local matched-step mean remains no better than AdamW.

Then the headline is a genuine local--global reversal explained by accumulated adaptation, with tail
redistribution as the query-level bridge.

### Retrieval gain without mechanism closure

If confirmation is positive but the shared-start tail endpoint or spectral attribution fails, report
the optimizer advantage but state that the tested tail/spectral mechanism does not explain it.

### Mechanism without recipe advantage

If the shared-start effect is reproducible but the BEIR interval is inconclusive, report a robust
trajectory difference without recommending Muon as a generally better retriever optimizer.

### Routing explanation

If hybrid AdamW matches Muon, the paper becomes a fairness result: much of the apparent optimizer
gain arises from hidden/auxiliary routing rather than orthogonalization.

### Negative or inconclusive result

If Muon-family confirmation is negative or inconclusive, retain the local--global and tail analyses
as constraints on optimizer explanations. Do not select a favorable discovery cell or LateOn result
to rescue the headline.

## Figures and tables

### Main figures

1. **DenseOn learning dynamics:** nDCG@10 and validation loss at 20/40/60/80/100%, all learning
   rates, with useful wall-time and censoring markers.
2. **Local--global reversal:** same-state matched-step mean margin on the left; one-seed medians over
   the four fixed learning-rate sweep points for unseen margin and final BEIR on the right.
3. **State feedback and tail redistribution:** trajectory-conditioned update cosines plus Adam-selected
   and challenger-selected cross-tail effects with set overlap.
4. **Accumulation and mechanism stress test:** three-seed shared-start loss p95/margin p05
   trajectories, early-spectrum held-out predictions, and the fixed dose/band/basis decision.

### Main tables

1. Discovery final nDCG@10, time-to-quality, throughput, memory, and instability.
2. Same-state update fingerprints and immediate functional effects.
3. Hybrid-routing and shared-start controls.
4. Validation-frozen three-seed BEIR contrasts with nominal and familywise intervals.

### Appendix

- all DenseOn learning-rate and per-task cells;
- layer/anchor common-state details;
- exact-spectrum and spectrum/basis transplant results;
- selection regret and post-hoc loss--BEIR diagnostics;
- complete provenance and failure records; and
- clearly labeled historical LateOn exploration.

Do not place token-level or MaxSim figures in either the main DenseOn argument or its mechanism
summary. Historical LateOn artifacts may be linked for audit without being interpreted as evidence
for the DenseOn thesis.

## Manuscript outline

1. **Introduction:** optimizer-switch tension and the local--global reversal.
2. **Background:** AdamW, Muon, NorMuon; local direction versus accumulated trajectory; tail
   redistribution.
3. **Controlled Dense Retrieval Study:** scope amendment, DenseOn data/objective, discovery,
   selection, confirmation.
4. **Retrieval Outcomes and Training Dynamics:** full DenseOn discovery and systems evidence.
5. **From Same-State Updates to Query Effects:** operator fingerprints, local disadvantage, symmetric
   tails.
6. **Does the Effect Accumulate, and Why?:** shared-start branches, temporal held-out bridge,
   spectrum dose/band intervention, and hybrid AdamW.
7. **Three-Seed Confirmation:** strict BEIR intervals and frozen wording.
8. **Discussion:** result-contingent interpretation and practical recommendation.
9. **Limitations and Reproducibility:** single-model scope and explicit post-hoc amendment.
10. **Appendix:** claim firewall, full DenseOn matrix, historical LateOn archive.

## Reviewer-facing claim firewall

| Reviewer concern | Required answer |
|---|---|
| Was the scope narrowed after seeing results? | Yes. State this plainly, preserve LateOn artifacts, and exclude them from primary inference. |
| Were learning rates selected on BEIR? | No for confirmation. The frozen validation rule selects recipes; discovery BEIR remains exploratory. |
| Is Muon's known spectral flattening the claimed novelty? | No. The contribution is the local--global reversal and its accumulated retrieval test. |
| How can spectrum explain retrieval rather than merely restate Muon? | It must beat optimizer labels and norm controls across held-out branch seeds, then pass frozen dose, band, basis-swap, and next-task BEIR prediction tests. Otherwise the paper explicitly rejects the spectral explanation. |
| Could routing explain the result? | Hybrid AdamW matches the hidden/auxiliary partition and auxiliary rate. |
| Does a one-step intervention establish final behavior? | No. It establishes only the local direction; shared-start branches test accumulation. |
| Does “tail improvement” use a favorable tail chosen after inspection? | Report fixed quantiles and symmetric Adam/challenger cross-tails; reserve robustness language for the frozen branch endpoint. |
| Are three seeds enough? | Report paired seed/task uncertainty and acknowledge that small effects may remain inconclusive. |
| Can LateOn be treated as a second replication? | No. It is historical exploratory material after a post-hoc scope amendment. |

## Completion checklist

- [x] Rewrite title, abstract, and contributions for DenseOn only.
- [x] Remove MaxSim/token-level claims from the main argument.
- [x] Disclose the user-directed post-hoc scope amendment.
- [x] Preserve local disadvantage, final discovery gain, tail redistribution, accumulation, hybrid
      routing, and three-seed confirmation as the central sequence.
- [ ] Complete and audit all DenseOn hybrid AdamW runs and strict BEIR cells.
- [ ] Complete and audit all DenseOn three-seed confirmatory runs and strict BEIR cells.
- [ ] Complete and audit all DenseOn shared-start branches and frozen tail endpoints.
- [ ] Complete the DenseOn spectrum/basis transplant or label the mechanism unresolved.
- [ ] Complete and audit the 45-checkpoint temporal spectrum predictor and its held-out-seed
      negative controls.
- [ ] Complete and audit the frozen dose/band/basis and held-out-run BEIR bridge; preserve a failed
      hypothesis as a claimable negative result.
- [ ] Render result macros and tables only after every bound evidence gate passes.
- [ ] Ensure the final abstract uses familywise confirmatory wording, not discovery point estimates.
- [ ] Retain a visible historical LateOn appendix/archive and the scope-amendment hash.
- [ ] Run strict paper audit and build the final review PDF.
