# Better Retrievers from Worse First Steps: paper story and evidence map

## The paper in one sentence

Muon reaches better DenseOn retrievers across a coherent historical learning-rate region even though
its norm-matched immediate step is weaker than AdamW's: the paper explains this local-to-global
reversal as an advantage that emerges along the optimization trajectory, rejects the familiar
spectral account, and uses the clean independently padded replication to determine whether the
advantage survives an execution-invariant comparison.

This is the authorial spine. The title, abstract, first page, result order, and conclusion must all
state the same progression: **Muon works; its first step is worse; the advantage emerges through
accumulation; the obvious spectral explanation fails; and a clean selector is required to find the
useful path.** The paper is not organized as a list of experiments or research questions. Every
main-text result must advance one link in that progression or protect it from a selection-path
artifact.

## The five-beat story

### 1. Muon works

Lead with the positive retrieval result rather than the optimizer implementation:

- best final nDCG@10: AdamW 0.5899, Muon 0.5923, NorMuon 0.5934;
- best-run task wins/losses against AdamW: Muon 10/4, NorMuon 11/3;
- four-rate final medians: 0.5858, 0.5901, and 0.5910;
- rates reaching the frozen AdamW reference: 2/4, 3/4, and 3/4; and
- fastest observed time to that reference: about 1.41 hours for AdamW versus 0.75 hours for each
  Muon-family optimizer.

The speed claim must remain precise. Muon and NorMuon have only 0.95x and 0.93x AdamW throughput on
this stack. They can reach quality sooner at a good rate; they do not execute each step faster.

These historical results are exploratory because BEIR reveals the best rate and training used the
packed path. They are nevertheless coherent evidence across rate, checkpoint, task, and
time-to-quality views, not a single favorable cell.

### 2. The obvious mechanism is wrong

If Muon simply provides a better descent direction, then a norm-matched step from identical weights
should improve the retrieval proxy more. It does not:

- matched-step mean margin gains are AdamW 0.0009, Muon 0.0006, and NorMuon 0.0005;
- Muon/NorMuon have recognizable spectral and row-allocation signatures, but those are operator
  fingerprints; and
- the optimizers select different adverse query tails rather than uniformly improving one shared
  set.

This creates the paper's main scientific puzzle: how does a locally weaker update lead to a better
full trajectory?

### 3. Repeated updates change the problem being optimized

The working answer is an **optimizer-induced trajectory effect**. Committing an update changes later
weights, gradients, and optimizer state, so a one-step ordering need not persist.

Evidence that belongs in the main text:

- AdamW--Muon same-gradient update cosine falls from 0.537 at the pretrained anchor toward 0.463
  along the Muon trajectory, while Muon--NorMuon remains near 0.971;
- at terminal anchors, Muon/NorMuon align less with the same terminal gradient than AdamW;
- in three shared-start seeds, Muon finishes with a positive unseen-margin contrast every time; and
- Muon-family checkpoints move about twice as far in hidden-weight space for nearly the same
  fixed-probe score drift.

The cosine evidence is post hoc and descriptive. The shared-start intervention establishes an
accumulated functional effect, but its frozen joint tail endpoint is mixed. The paper may say that
the benefit emerges along the trajectory; it may not claim formal mediation through state feedback.

### 4. The familiar spectral story fails

Spectral flattening is not the novelty. The study tries to make it explanatory and fails:

- early tail spectral energy does not improve leave-one-seed-out prediction of final loss p95 or
  unseen margin p05;
- interpolating Muon singular values into the AdamW basis shows neither the required dose response
  nor tail-band localization;
- spectrum features do not beat basis controls in held-run prediction of the next BEIR checkpoint;
  and
- routing-matched hybrid AdamW changes mean retrieval by only +0.000077, so parameter grouping is not
  a sufficient recipe explanation.

This negative mechanism result advances the story: Muon's retrieval behavior cannot be reduced to
the most visually distinctive property of its update. The corrected nine-feature bridge is the only
route by which a new geometry feature may enter the conclusion. It must lower pooled held-out RMSE
and improve at least three of four leave-dose-index-out folds beyond optimizer, stage, and dose.

### 5. Selecting Muon is a separate problem from Muon being good

The historical eight-way validator chooses 3e-3 for Muon and NorMuon, ten times the retrieval-optimal
rate. Three new training views confirm that those selected recipes lose about 0.03 nDCG@10 to
AdamW. This is not the main optimizer result; it is a model-selection failure.

The candidate-breadth audit first tests whether seven negatives are too narrow. Its prerequisite
fails: independent padded width-7 scoring cannot reproduce the packed validator. A two-example
control changes a packed cosine score by as much as 0.211914, versus 0.001953 with forced padding.
On the padded path, the high-dose advantage is absent before widening from 7 to 2,048 candidates.

The consequence is specific:

- do not describe the negative three-seed selected-recipe result as evidence that Muon is bad;
- do describe it as evidence that a batch-dependent selector can miss Muon's useful region;
- do not pool historical packed and corrected padded executions; and
- let the corrected 12-run matrix govern the final optimizer recommendation.

## Final paper structure

1. **Introduction:** the positive Muon result and the locally-weaker/globally-better paradox.
2. **Why retrieval is different:** shortlist training, corpus ranking, and the four-link evidence
   chain.
3. **Controlled study:** only the design needed to follow the argument.
4. **Muon reaches better retrievers---and validation misses them:** result, task breadth,
   time-to-quality, then the selection reversal.
5. **Why model selection hides the useful Muon regime:** candidate-breadth falsification and packed
   execution audit, ending in the clean replication.
6. **Corrected independently padded replication:** generated primary all-rate result, secondary
   selected-recipe result, five-stage dynamics, and the compact systems summary. The three-contrast
   all-rate table stays in the main narrative; the same source-bound all-rate finding is injected
   into the abstract and Conclusion; the complete nine-feature bridge and execution-path
   sensitivity tables are reported in the appendix.
7. **A trajectory effect, but not a spectral explanation:** same-state step, shared-start
   accumulation, weight/function distance, failed spectral chain, and corrected predictive bridge.
8. **Discussion:** optimizer quality versus optimizer selection; time-to-quality versus throughput;
   geometry fingerprints versus explanations.
9. **Conclusion:** the source-bound corrected optimizer verdict first, then one mechanism statement
   and one selection statement. The complete historical packed-selector claim is an audited
   appendix record, not the main conclusion.

Protocol inventories, source hashes, scope history, full numerical diagnostics, all learning-rate
cells, all per-task cells, and LateOn provenance belong in the appendix or repository. They should
not interrupt the argument.

## Corrected result branch points

The narrative spine stays fixed, but the final wording follows the independently padded matrix:

- **Muon positive:** the historical advantage replicates; headline Muon's all-rate robustness and
  ask which frozen geometry feature, if any, predicts the gain.
- **Muon inconclusive:** historical evidence remains promising but execution-sensitive; emphasize
  that the effect size is smaller than the current design can resolve.
- **Muon negative:** the historical advantage was specific to packed training; the contribution
  becomes a strong execution-path result, while retaining the local-to-global trajectory analysis as
  a bounded historical mechanism study.

No branch may be chosen before all 12 runs, 60 checkpoints, 840 BEIR task units, validation outputs,
geometry rows, and source-bound audits are complete.

## Prospective state-by-operator factorial

The paper needs a positive explanation only if it survives a direct attempt to separate the state
reached by an optimizer from the operator applied next. The historical fixed-state interventions
contain a useful but post-hoc crossover: on AdamW trajectory anchors, a matched Muon step often
looks better than AdamW, while on Muon trajectory anchors that ordering often reverses. This rejects
a state-invariant ranking of isolated directions, but it is not yet a mechanism result.

The corrected follow-up is frozen in
`configs/dense_no_packing_state_operator_factorial_protocol.json`. At the 60% checkpoints of the
historically retrieval-optimal AdamW and Muon rates, it crosses two weight states with two reset
continuation operators on the same 50K branch view and three fixed order seeds. The fixed
calibration-probe hidden update is scale matched in every cell; this does not assert that the first
training batch produces an identical realized update. Final full-corpus BEIR yields three
predeclared contrasts:
the carried weight-state effect, the continuation-operator effect, and their interaction.

Its exact calibration, reset-continuation training, padded probe, final-BEIR, and two-way bootstrap
implementation is independently source-bound in
`configs/dense_no_packing_state_operator_factorial_implementation_protocol.json`; execution and
interpretation commands are documented in `docs/state-operator-factorial.md`.

This factorial decides what the mechanism section is allowed to say:

- a weight-state effect means Muon reaches weights whose advantage survives an optimizer reset;
- an operator effect means Muon's transform helps from both source states;
- a positive interaction means the Muon state and Muon continuation reinforce one another, which
  is the direct evidence needed for a closed-loop state-feedback account; and
- no stable contrast means the paper keeps the positive retrieval result but makes no positive
  mechanism claim.

NorMuon is intentionally excluded from this factorial. It remains a secondary optimizer ablation;
the causal story being tested is the AdamW--Muon comparison that anchors the paper.

## Claim discipline

Use these distinctions consistently:

| Observation | Permitted wording | Forbidden shortcut |
|---|---|---|
| Better best/median historical BEIR | promising coherent Muon region | universal Muon superiority |
| Faster first passage at one good rate | lower observed time-to-quality | faster optimizer implementation |
| Weaker same-state mean step, better final run | trajectory-level emergence | formal mediation by state feedback |
| Distinct spectrum/row statistics | optimizer fingerprint | retrieval mechanism |
| Failed spectral prediction/intervention | tested spectral account rejected | no geometric mechanism can exist |
| Packed selection failure | validator misses useful region | Muon intrinsically overfits seven negatives |
| Historical/corrected difference | execution-path sensitivity | randomized causal effect of packing |
| Corrected geometry prediction | candidate predictive bridge | causal mediation |

## Scope disclosure

The active paper is DenseOn-only. LateOn was removed by a user-directed, post-hoc scope amendment
after some exploratory outputs were visible because it was much slower and less central to the
intended audience. LateOn is not pooled with DenseOn, used as replication, or allowed to determine
headline wording. Its complete provenance remains auditable. The dated authoritative record is
`configs/dense_scope_amendment.json`.

## Finalization checklist

- Replace the outcome-neutral corrected paragraph only through the source-bound publication
  renderer after all corrected audits pass.
- Verify that the renderer has inserted the same all-rate Muon-versus-AdamW and
  NorMuon-versus-AdamW finding into both the abstract and main Conclusion.
- Confirm that main-text figures show the positive retrieval surface and the local-to-global
  paradox, not a wall of protocol tables.
- Keep only one compact causal-chain decision table in the main paper; full estimates remain in the
  appendix.
- Rebuild the PDF and verify the eight-page main-text boundary, float topology, citations, source
  hashes, and absence of pending result macros or Type 3 fonts.
