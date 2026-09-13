# Do functional predictors survive recipe controls?

Post-result exploratory analysis of all four functional measurements on the
same complete 60-checkpoint panel. It extends the already reported weight-space
sensitivity with exactly the same four recipe comparators and two displacement-
conditioned variants. No predictor, rate, stage or fold is selected or omitted.

## Finding

**No functional predictor passes all four recipe comparators.** The apparently
strong degrading-attribution signal under the declared baseline is comparator-
dependent, just as the earlier displacement signal was. These results do not
provide a robust explanation of an optimizer's retrieval advantage.

The table reports whether adding each measurement reduces pooled held-dose RMSE
and improves at least three of four folds. All 96 fits are resolved. “No” is a
failure of that predictive criterion, not evidence of equivalence.

| Comparator | Helpful share | Helpful participation | Degrading mass | 50% removal retention |
| --- | --- | --- | --- | --- |
| B0: optimizer + stage + log rate | No | No | Yes, 4/4 | No |
| B0 + cumulative displacement | No | No | Yes, 3/4 | No |
| B1: B0 + optimizer-specific raw rate | No | No | No | Yes, 3/4 |
| B2: optimizer × stage + optimizer-specific log/raw rate | No | No | No | No |
| B3: B2 + optimizer-specific nominal schedule budget | No | Yes, 4/4 | No | No |
| B3 + cumulative displacement | No | No | No | No |

The degrading-mass predictor reduces B0 RMSE from **2.90535 to 1.89844** nDCG@10
points. With displacement already included, it reduces **1.00397 to 0.90680**
(3/4 folds). With raw-rate controls, its pooled improvement is much smaller,
**0.65174 to 0.62318**, and only 2/4 folds improve. Under B2 and B3 it worsens
pooled prediction. B0's positive residual association therefore cannot be
presented as a recipe-robust mechanism.

The other isolated flags must also be retained: 50% removal retention improves
B1 (**0.65174 to 0.59980**, 3/4), while helpful participation improves B3
(**1.43879 to 1.42201**, 4/4). Neither passes all comparators; the latter does
not pass after adding displacement. We do not choose a preferred baseline from
these results or interpret the diagnostic flags as significance tests.

## Design and complete evidence

The same four folds hold out one learning-rate dose across every optimizer and
stage: 45 train / 15 test observations. Outcomes are the fourteen-task macro
full-corpus nDCG@10 scores. Predictor measurements use the fixed eight-candidate
probe, not a separate compressed full-corpus benchmark. Baseline widths are
8/11/21/24 before the optional displacement column. The schedule covariate is
nominal accumulated LR, not an observed update norm.

- [All 24 summaries](actual/summaries.csv), [96 folds](actual/folds.csv), and
  [1,440 exact held-out predictions](actual/predictions.csv).
- [Unrounded calculations and design matrices](actual/result.json).
- [Independent verification](actual/independent_verification.json): all designs
  rebuilt, the nominal schedule independently summed, and all predictions and
  decisions checked with full SymPy rational OLS rather than the producer's FWL.
- [Original functional results](../dense-v3-functional-inference-v1/README.md)
  remain unchanged. All 240 original B0 predictions are exactly reproduced.

This is an exploratory robustness qualification of the same measurements, not
an independent training replication, a revised primary optimizer comparison,
causal mediation, or generalization to unseen tasks. Crossed continuation
outcomes and final manuscript integration remain pending.

## Numerical replay

From this directory, with the study's pinned Python/NumPy/SymPy environment:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B source/functional_sensitivity.py \
  --parent inputs/functional \
  --designs source-parent/sensitivity.py \
  --arithmetic source-parent/bridge_exact_arithmetic.py \
  --output /tmp/functional-sensitivity-NEW
```

The output path must not exist. Inputs and the two unchanged numerical parents
are content-bound. Replay is numerical reconstruction, not a fresh validation
of all original model checkpoints. Source copies are local WIP, not a GitHub
release. Immutable result identities and copies are listed in `verification.json`.
