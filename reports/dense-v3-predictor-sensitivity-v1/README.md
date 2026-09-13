# Recipe controls change the interpretation of geometric prediction

This **post-result exploratory sensitivity** tests the two limitations identified
in the [locked weight-to-retrieval readout](../dense-v3-weight-retrieval-v1/README.md):
nonlinear recipe effects omitted by the baseline, and whether other geometric
measurements add information conditional on cumulative displacement. It preserves
the original fourteen features, sixty states, four held-dose folds and every
original result. It is not a new preregistration or a replacement primary analysis.

The first actual calculation finished **2026-09-12 12:33:52 UTC**. Independent
design reconstruction and full-matrix rational OLS finished **12:38:50 UTC**:
all **5,040 prediction rows, 336 fold comparisons and 84 pooled decisions** agree.
A copied-source/data replay finished **12:37:53 UTC**, reproducing all seven
numerical/text outputs byte-for-byte. Original B0 predictions also reproduce all
840 predictions in the preceding locked readout exactly.

## Main observation: the comparator matters

All values below are pooled held-dose RMSE in nDCG@10 points; lower is better.
The final two columns are separate one-feature additions, not a joint model.

| Recipe-only comparator | Parameters | Comparator alone | + cumulative displacement | + full-spectrum segment entropy |
| --- | ---: | ---: | ---: | ---: |
| B0: original additive baseline | 8 | 2.905349 | 1.003975 (4/4 improved) | 2.347064 (3/4) |
| B1: B0 + optimizer-specific raw LR | 11 | 0.651744 | 0.795009 (1/4) | 0.689136 (2/4) |
| B2: optimizer-specific stages and log/raw LR | 21 | 1.390538 | 1.407186 (2/4) | 1.294010 (4/4) |
| B3: B2 + nominal cumulative LR budget | 24 | 1.438786 | 1.420297 (4/4) | 1.356213 (4/4) |

The large displacement gain under B0 does **not** persist as a comparably large
gain under these recipe-only alternatives. B1, which adds only three raw-rate
columns and reads no model weights, has lower observed error than B0 plus
displacement; adding displacement to B1 makes prediction worse. B3 retains a
small positive increment, so the result is not that displacement never matters.

The conclusion should be **comparator sensitivity**, not “displacement explains
Muon.” Nor can the failure of stable rank under B0 be generalized to every
comparator: original/exact segment stable rank passes the descriptive criterion
under B1. No one of the fourteen features passes under **all four** unconditioned
comparators. Those are deliberately all reported; B1 is not selected as a new
confirmatory benchmark because it has the lowest observed error.

## Conditional information is also not stable across comparators

For B0 and B3, cumulative displacement is first included in the comparator and
each of the same fourteen columns is added separately. Self-addition of
displacement is an exact redundant control, not another hypothesis.

- B0 + displacement has RMSE **1.003975**. Adding full-spectrum segment entropy
  increases it to **1.260228** (only 2/4 folds improve), failing the diagnostic
  criterion. Thus the original entropy result does not establish incremental
  predictive information after displacement under that comparator.
- B3 + displacement has RMSE **1.420297**. Adding the same entropy measurement
  reduces it to **1.347976**, with 4/4 improvements. The conditional conclusion
  therefore depends on the comparator too.
- Original/exact segment overlap with AdamW helps under B0 + displacement but
  not under B3 + displacement. This reinforces the need to report all features
  and choices, rather than choosing a preferred geometric story after fitting.

See the [complete 84-comparison table](actual/summary.md),
[all-feature sensitivity figure](figures-v2/comparator_sensitivity.pdf),
[all 336 folds](actual/folds.csv) and [all predictions](actual/predictions.csv).
The figure preserves every comparison and its numeric input CSV. A star means
only lower pooled MSE and improvements in at least three folds, never significance.

## What this does and does not change

The observation directly limits the scientific interpretation of the preceding
analysis. It is consistent with geometric predictors absorbing nonlinear
learning-rate or optimizer-stage patterns missing from B0, but it does not prove
that this is the sole explanation. Feature coefficients, prediction gains and
conditional associations do not identify causal or functional effects.

Richer is not automatically better: B2/B3 have worse held-dose error than B1.
Their 21/24 parameters are estimated from only 45 training rows; B3 plus
displacement has 25. Endpoint folds extrapolate from three learning rates, and
stage-specific rows share a training trajectory. These comparisons do not provide
new seeds, independent tasks, external models or a reliable model-selection study.
All design choices here were made after the first readout, so neither positive
nor negative flags should be described as confirmatory statistical findings.

Nothing here changes the completed endpoint comparison: the validation-selected
NorMuon–AdamW positive task-level interval, the other inconclusive contrasts,
the full retrieval trajectories and all checkpoints remain unchanged. It also
does not establish more retrieval-useful embedding dimensions. The actual
functional measurements and the separately bounded crossed continuation remain
unfinished; no optimizer mechanism should be substituted with these regressions.

## Exact computation and verification

The [plan](plan.md), saved before this sensitivity ran, fixes four comparators
and two displacement-conditioned versions: 84 comparisons in total. B0 is the
original eight-column design. B1 adds raw LR separately for each optimizer.
B2 uses fifteen optimizer-by-stage indicators and optimizer-specific log/raw
rate slopes. B3 adds nominal cumulative LR budget separately for each optimizer.

The nominal schedule covariate is **recipe-only**: 3907 total steps, 391 warmup
steps, zero multiplier at schedule index zero, linear warmup/decay. Its cumulative
sum is normalized to one at the final checkpoint. It is not a measured update
norm, optimizer-state displacement, true path length, weight-decay intervention
or compute-matched experiment. An independent direct rational sum checks the
closed-form schedule values and every actual design row.

The [standalone calculation](source/sensitivity.py) uses a new generic-width
exact FWL solver, reusing unchanged arithmetic helpers. It does not edit the
original eight-column solver or call a formal primary consumer. Original
binary64 features/outcomes are unchanged; deterministic recipe covariates are
converted to their exact rational binary64 values. Rank/undefined conventions,
strict MSE comparisons and all rows are retained. All 328 nonredundant actual
fits are resolved; eight self-addition folds are baseline-equivalent.

The [independent verifier](source/verify_independent.py) reconstructs all 240
recipe-design rows and sixty schedule covariates separately, then solves full
augmented rational normal equations, not the producer's FWL path. All predictions,
fold errors, pooled errors, displayed RMSEs and flags agree. This is a numerical
verification of the same data, not experimental replication.

## Reproduce from copied files

Copy this report directory, including the deliberately selected seven-file
`parent/` input/source subset. No model, GPU, project package or network is needed.
Use a new absolute output directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B /absolute/report/source/sensitivity.py \
  --parent /absolute/report/parent \
  --output /absolute/new-sensitivity-result
```

Actual runtime used NumPy 2.5.2; the independent verifier additionally used SymPy
1.14.0. The `parent/verification.json` is the unchanged external parent receipt,
not a claim that this seven-file subset contains all 226 parent archive files.
Copied-source/data reproduction was on the same physical host, not a second-host
model experiment. This report and its scripts are local WIP, not yet remote
data backup or source publication. No model, retrieval worker, frozen protocol,
manuscript or failed/stopped coordinator was changed.
