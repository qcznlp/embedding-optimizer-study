# NAACL paper plan: optimizer-induced weight states in dense retrieval

Status: active authorial plan. The authoritative manuscript is `paper/main.tex`.

## Thesis

The paper does not ask whether Muon has a recognizable update spectrum. Orthogonalization makes
that unsurprising. It asks what retriever Muon creates from a pretrained DenseOn state, how that
state differs from the one AdamW creates, and how the reached states respond to further optimization.

The central distinction is **weight-space spread versus retrieval-useful dimension allocation**.
The exact coordinate sum for a cosine margin motivates this distinction; coordinate deletion with
renormalization measures something different and is not an additive decomposition or new theorem.
The paper should develop this question through the results, rather than narrate verification gates.

The explanatory chain is:

```text
optimizer rule
    → closed-loop trajectory through weight space
    → reached weights
    → embedding geometry and dimension utility
    → full-corpus retrieval
```

A scientific explanation must cross this chain. An intrinsic update property, a one-step proxy, or
an optimizer-separating visualization is insufficient by itself.

## Central questions

1. Across a predeclared learning-rate surface, do AdamW, Muon, and NorMuon produce different
   DenseOn retrieval outcomes and learning dynamics?
2. How do their reached weight states differ in displacement, effective rank, row allocation, and
   subspace?
3. Do those states use the 768 embedding dimensions differently—not only in covariance rank, but
   in helpful, redundant, and degrading contributions to retrieval?
4. Which weight- or representation-space measurements predict full-corpus retrieval when an entire
   learning-rate dose is unseen?
5. After a fixed continuation horizon, does a source-state advantage persist on average, and does
   the relative continuation preference depend on the reached state?

## Main experiment

- Model: `lightonai/DenseOn-unsupervised`, pinned revision.
- Data: one deterministic 500,000-query view.
- Contrastive group: one positive and seven seeded hard negatives.
- In-batch and cross-device negatives: disabled.
- Maximum query/document length: 8,192.
- Objective: cosine InfoNCE, temperature 0.02.
- Schedule: one epoch, global batch 128, five retained stages.
- Optimizers: AdamW, Muon, NorMuon.
- Surface: four predeclared learning rates per optimizer.
- Evaluation: exact nDCG@10 on 14 pinned decontaminated BEIR tasks.
- Compute: two disjoint four-GPU pools.

The primary estimand averages all four learning-rate cells within optimizer at the final stage. It
is a statement about the tested surface, not the best observed cell. Three pairwise optimizer
contrasts receive common-resample simultaneous max-T task-level intervals. A validation-selected
recipe and all five trajectory stages are secondary, fully reported views.

## Weight-state analysis

At each retained checkpoint, measure:

- saved-checkpoint segment magnitude as the joint displacement Frobenius norm divided by the
  joint current hidden-weight norm, not an average of matrix-wise ratios;
- cumulative displacement from the pretrained state;
- stable- and entropy-effective-rank fractions;
- row-norm coefficient of variation and top-1% row-energy share;
- rank-16 left and right subspace overlap over all unordered run pairs at matched stage, with
  optimizer-pair summaries averaging all rate pairs equally.

The retained-checkpoint segment is not called a per-step optimizer update. These measurements can
describe how trajectories diverge, but they explain retrieval only if they add out-of-dose
predictive value.

Retain the original approximate measurements and the separately defined full-spectrum sensitivity.
Full-spectrum entropy rank and renormalized truncated-spectrum entropy rank have different
definitions, not merely different numerical precision. The full-spectrum rank summaries are
parameter-weighted over nonzero matrices; make that denominator explicit. Neither branch replaces
the other or silently selects a more favorable descriptor.

## Functional dimension-utilization analysis

The dimension analysis follows the functional distinction in
[Takeshita et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1410/) while adding a
change-of-basis control.

For every checkpoint on a frozen 224-query, 14-task probe:

- compute query and document covariance stable/effective rank;
- remove 20 shared random coordinate sets at each of 10%, 25%, 50%, and 75%;
- delete each coordinate in turn and recompute shortlist nDCG and margin;
- use task-mean margin deletion effects to compute helpful mass share, normalized helpful
  participation, and degrading mass; analogous nDCG attributions remain descriptive;
- repeat endpoint attribution after three shared Haar rotations that preserve full-vector cosine
  scores.

For each coordinate, average the deletion effect within task before separating its sign. If H is
total helpful mass and B is total degrading mass, the primary features are H/(H+B),
H²/(768 × sum of squared helpful effects), and B. Zero denominators are zero by definition.
Compute these nonlinear summaries within task before averaging tasks. The final-stage inference
uses all four rates and one nine-contrast fixed-SE max-T family with 50,000 common task resamples.
These are the existing source/protocol definitions, not a newly selected metric or analysis.

The paper never equates effective rank with useful dimensionality. High random-removal retention
can mean redundancy, distributed coding, or cancellation. Native-coordinate effects are explicitly
basis dependent. A claim that Muon uses coordinates more constructively requires simultaneous
support for more helpful mass share, broader helpful participation, and less degrading mass. A
direction stable across the three tested rotations is a robustness result, not proof of invariance
under arbitrary bases or greater dimensional capacity. The factorial does not intervene on a
particular geometric feature and cannot by itself establish that feature's causal role.

## Retrieval bridge

Every candidate feature is added separately to a baseline with:

- optimizer identity;
- checkpoint stage;
- centered log10 learning-rate dose within optimizer.

Four folds each hold out one ordered dose index for all optimizers and stages. Support requires
lower pooled held-out RMSE and improvement in at least three of four folds. All frozen features are
reported, including unsupported and unidentified comparisons. Predictive support is not causal
mediation, and held-out-dose performance is not held-out-task generalization.

## Conditional continuation experiment

Take one 60% AdamW-reached state and one matched-stage Muon-reached state. Reset all optimizer
history and cross each source state with AdamW and Muon continuation:

| Source state | Reset AdamW | Reset Muon |
|---|---:|---:|
| AdamW-reached | A→A | A→M |
| Muon-reached | M→A | M→M |

The hidden update-to-weight ratio is matched on the fixed calibration probe, not on the first
realized shuffled-batch update or every subsequent step. All cells use the same
50,000-query horizon under three fixed data-order seeds. The factorial estimates:

- source-state effect: average post-continuation difference between the two fixed reached states;
- continuation-operator effect: average endpoint difference between reset Muon and reset AdamW;
- interaction: whether the continuation effect depends on which state was reached.

These are contrasts of final scores after 50K continuation, not gains relative to the starting
checkpoints or a decomposition of the primary 500K result. For endpoint means AA, AM, MA and MM,
the diagonal difference MM-AA equals the state plus operator contrasts; the interaction is not a
third additive component. An averaged benefit need not hold within both states or operators, and
positive interaction may mean less harm rather than benefit. Separate marginal intervals are not
simultaneous coverage; an inconclusive interval establishes neither equivalence nor dominance of
another contrast.

The [scientific claim review](../reports/paper-review/factorial-claims-v1/README.md) preserves
counterexamples to the older wording and the original candidate. Its narrower interpretation is
now integrated in the isolated manuscript/renderer under a separate
[publication-only amendment](../configs/dense_no_packing_state_operator_claim_wording_amendment.json).
The numerical design is unchanged, and no runtime deployment has occurred. The experiment remains
useful for conditional continuation responses, not proof of feature mediation.

## Main-paper structure

1. **Introduction:** optimizer choice as weight-state selection; explanatory chain and hypotheses.
2. **From updates to retrieval states:** why an operator fingerprint is not an outcome.
3. **Related work:** dense retrieval, matrix optimizers, representation dimensionality.
4. **Experimental design:** one controlled DenseOn surface and frozen inference.
5. **Optimizer effects on dense retrieval:** primary contrasts and five-stage dynamics.
6. **How the optimizers reshape the retriever:** weight trajectories, dimension utility, and
   out-of-dose bridges.
7. **How reached states respond to further optimization:** conditional crossed continuation.
8. **Discussion and conclusion:** result-contingent interpretation, scope, and transfer implications.

Implementation debugging and invalid exploratory runs are excluded from the paper. They remain in
the repository's engineering provenance but are not scientific results, contributions, limitations,
or narrative transitions.

## Result-contingent interpretation

- **Muon retrieval gain + source-state effect:** the fixed Muon source has an averaged advantage
  after continuation; this need not hold under each continuation rule.
- **Muon retrieval gain + operator effect:** Muon has an averaged continuation benefit for the
  fixed pair, not necessarily a benefit within both states.
- **Muon retrieval gain + interaction:** relative continuation preference depends on source state;
  the sign alone does not prove beneficial co-adaptation or explain the primary gain.
- **Muon retrieval gain + predictive dimension feature:** representation utility supplies a
  testable bridge from weight trajectory to corpus ranking.
- **Muon retrieval gain + no supported bridge/factorial effect:** report the gain and which
  explanations remain unsupported; lack of support alone does not falsify every mechanism or
  establish equivalent states.
- **No established Muon retrieval gain:** distinguish an estimated disadvantage from inconclusive
  evidence in the fixed primary comparison. Weight-space differences remain descriptive unless
  independently linked to retrieval; do not infer a universal negative result about Muon.

## Claim firewall

| Observation | Supports | Does not support alone |
|---|---|---|
| flatter Muon update spectrum | operator identity and implementation check | better retrieval |
| larger cumulative displacement | a different optimization path | more useful representation |
| higher effective rank | broader covariance support | constructive coordinate use |
| better random-removal retention | robustness to coordinate deletion | basis-independent dimension use |
| out-of-dose prediction gain | candidate retrieval bridge | causal mediation |
| source-state factorial effect | averaged post-continuation difference for the fixed pair | benefit under both operators, or decomposition of the primary gain |
| state-by-operator interaction | a difference in relative continuation responses | benefit in both states or full-trajectory co-adaptation |

## Submission gates

The paper is submission-ready only when:

- all 12 primary runs and 60 checkpoints are complete and remotely audited;
- all 840 checkpoint-task BEIR units are present;
- corrected weight-state and dimension exports pass source hashes;
- all dimension contrasts, rotation controls, and retrieval bridges are rendered;
- all 12 factorial branches, 60 probe checkpoints, and 168 factorial BEIR units are complete;
- the paper has no pending result macro, exceeds neither the 200-word abstract limit nor the
  eight-page main-text limit, and contains no unsupported causal language;
- a clean clone reproduces every manuscript table from the portable evidence closure.
