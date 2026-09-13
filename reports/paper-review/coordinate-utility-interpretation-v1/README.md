# Coordinate utility is a sensitivity, not a retrieval gain

This is a **mathematical interpretation check using constructed vectors**, not a DenseOn result,
an optimizer comparison, a new primary metric, or a reason to retrain. The frozen analysis and
manuscript findings are unchanged. Its purpose is to prevent a later coordinate-utility result
from being mistaken for a demonstrated retrieval improvement or feature-level causal explanation.

## Exact identities

For unit query and document vectors, write their full cosine as $s=q^T d$. Provided the retained
vectors are nonzero, deleting coordinate $j$ and renormalizing gives

$$s_{-j}=\frac{s-q_jd_j}{\sqrt{(1-q_j^2)(1-d_j^2)}}.$$

This is a mathematical identity, **not a proposal to replace the literal-deletion kernel** with
total-minus-component arithmetic. It shows why a deletion effect is not just minus the removed
coordinate's additive score contribution: query and document normalization also changes.

There is a second exact term in a hardest-negative margin. Fix one query and let $k^*$ be a
hardest negative before deletion. Let $f_j$ be the margin change after deletion if this original
negative were kept fixed. The actual readout recomputes the maximum over all seven negatives:

$$a_j=f_j-r_j,\qquad r_j=\max_{k\in\mathcal N}s_{k,-j}-s_{k^*,-j}\geq0.$$

Task averaging preserves this equality: $a_{tj}=\bar f_{tj}-\bar r_{tj}$ with
$\bar r_{tj}\geq0$. Only **after** this average does the original analysis separate signs.
Relative to the hypothetical fixed-negative readout, reselecting the hardest negative can only
increase helpful mass and decrease degrading mass; helpful mass share cannot decrease under the
original zero-total convention. Helpful participation has no monotonic implication from this
component-wise argument. These statements concern the two readouts of one fixed representation,
not a general ordering between optimizers. The hypothetical readout is not added to the primary
feature set or used to choose a preferred result.

For task-mean effects $a_{tj}$, $H_t=\sum_j\max(-a_{tj},0)$ and
$B_t=\sum_j\max(a_{tj},0)$. When their sum is positive,

$$\frac{H_t}{H_t+B_t}=\frac12\left(1-\frac{\sum_j a_{tj}}{\sum_j|a_{tj}|}\right).$$

Thus helpful share describes the signed balance of deletion effects. It is not a count of
independent semantic dimensions. Nothing here changes the original meaning of native-coordinate
sensitivity or establishes that it cannot predict retrieval empirically.

## A dense 768-dimensional counterexample

The [fixed dense witness](dense-counterexample.json) changes only the orientation of a
lower-ranked competing document. The query, positive, original hardest negative, and other five
negative vectors remain identical. The vectors below give their first four coordinates; **every
one of their remaining 764 coordinates equals 1**:

```text
query:                    ( 1000,  1000,  1000,  1000)
positive:                 ( 2000, -1000,     0,  2000)
original hardest negative:( 1000, -3000, -1000,  3000)
other competitor, A:      (    0, -3000, -2000,  3000)
other competitor, B:      (-3000,     0,  3000, -2000)
each of five others:      (-1000, -1000, -1000, -1000)
```

The changed competitor's first four entries are a permutation. Since the query's first four
entries are equal, its dot product with the query and its norm are exactly unchanged. Therefore
**all eight full-vector cosine scores are identical**, not merely rounded nDCG or a coarse rank.
The joint query/document span has exact rational rank **5 in both constructions**.

| Readout | Construction A | Construction B |
| --- | ---: | ---: |
| Full-vector shortlist margin | 0.4999729458 | 0.4999729458 |
| Full-vector shortlist nDCG | 1 | 1 |
| Helpful mass share | 0.5689688194 | 0.9999700909 |
| Normalized helpful participation | 0.0031126954 | 0.0040149355 |
| Degrading mass | 0.2804130035 | 0.0000270518 |

All three raw feature values move in their desired directions, although the full scores and span
rank do not change. The hypothetical fixed-original-negative effects are identical between A and
B. The difference is accounted for by nonnegative hardest-negative switching penalties at two
coordinates. This demonstrates a logical boundary, not a frequency estimate or a Muon effect.

The query has 768 nonzero coordinates; every document has at least 767. Every single-coordinate
deletion is valid, and all **80 original shared random-removal masks** produce finite cosine
scores in both cases. This dense version is not relying on undefined zero-vector deletions.
It still is one constructed shortlist, not the 224-query probe or a complete primary analysis.

## What was actually checked

- The original [small-integer search](source/review.py) found its first witness at trial 43 under
  seed 20260910. It was explicitly searching for a counterexample, not estimating prevalence.
  Its zero-tail construction and original [result](counterexample.json) remain unchanged.
- The [dense check](source/dense_review.py) uses the fixed witness above, 80-digit Decimal cosines
  from exact integer products, exact rational rank, and the **unchanged** production
  `leave_one_out_metrics` and `_attribution_summary` at all 768 coordinates.
- Per-coordinate agreement with the Decimal reference satisfies the unchanged diagnostic
  absolute bound of $10^{-12}$; maximum observed error is about $1.13\times10^{-14}$.
  Propagating that bound through the sums and nonlinear ratios produces disjoint deterministic
  bounds for all three feature orderings. These bounds are **not statistical confidence intervals**.
- Both archived script entrypoints were actually rerun. All non-timestamp output fields reproduce
  exactly on this host. Replays are not independent experiments or training seeds.

The first dense check incorrectly required a 768-term aggregate to retain the same $10^{-12}$
absolute error bound as one coordinate; it exited 1. The [initial source](source/dense_review.initial.py),
[measured differences](initial-aggregate-diagnosis.json), and original failure are preserved.
Aggregate discrepancies reach about $6.1\times10^{-12}$ while individual errors remain below
$1.13\times10^{-14}$. The follow-up retains those failed flags and the original coordinate bound,
and explicitly propagates the bound through each feature. No production arithmetic, original
scientific tolerance, or statistical decision rule was changed to obtain acceptance.

No model, dataset, training state, or GPU was loaded. No task bootstrap, held-out-dose fit, rotation
inference, or complete-probe publication was run. The JSON explicitly leaves the formal scientific
claim rule and scientific completion false. This is not an assertion that the complete scientific
claim rule passed on the constructed example.

## Consequence for the paper and next work

Retain the predeclared dimension metrics and simultaneous inference. If they favor an optimizer,
describe the measured native-coordinate sensitivities first. Do not infer increased independent
dimensional capacity, improved full-vector ranking, or causal explanation of a retrieval gain from
those three metrics alone. The full-corpus comparison and held-out-dose bridge remain necessary;
even predictive support is not mediation. No additional experiment or feature is made a completion
prerequisite by this note, and no result is selected or discarded because of it.

The [manuscript-framing milestone](../dense-v3-retrieval-usefulness-v1/README.md), all its artifacts,
and its manuscript bytes are preserved. Only the [current handoff](../../../CURRENT_EXPERIMENT.md)
adds this interpretation note and a dated progress update; its previous version is under [before/](before/).
See [verification](verification.json), [commands](commands.json), and separately timestamped
[observations](observations.json). Runtime restrictions in [AGENTS.md](../../../AGENTS.md) are unchanged.

Run either archived check on this host with hidden GPUs and the exact audited source:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src /usr/bin/python -B \
  /root/embedding-optimizer-story-refactor/reports/paper-review/coordinate-utility-interpretation-v1/source/dense_review.py
```

Use `source/review.py` instead for the original finite search. These are CPU interpretation checks,
not formal experiment launchers or a source release.
