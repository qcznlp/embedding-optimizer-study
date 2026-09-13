# Which weight changes predict held-dose retrieval?

The complete actual sixty-state numerical analysis is now available. The first
calculation finished at **2026-09-12 12:15:02 UTC**. All **840 feature-specific
held-out predictions, 56 fold comparisons, 14 pooled decisions and 14 residual
associations** agree with an independent full-matrix rational OLS solver. A fresh
process using only copied source and copied data finished at **12:18:57 UTC**;
all **17 numerical/text outputs**, including twelve CSVs, reproduce byte-for-byte.

This is an actual fixed-grid predictive analysis, not the original formal
whole-primary/publication consumer. Those admission checks remain uncalled and
unchanged. It does not establish functional utility, mediation or causality.

## Result

The baseline's pooled held-dose RMSE is **2.905349 nDCG@10 points**. It includes
optimizer indicators, checkpoint-stage indicators and a within-optimizer centered
log10 learning rate. Each comparison adds exactly one predeclared feature.

The [complete generated result table](actual/summary.md) retains every feature.
The [all-feature figure](figures/held_dose_prediction.pdf) visualizes the same
fourteen comparisons; every plotted value is retained in its CSV.
The strongest observed reduction among these separately evaluated features is
for normalized cumulative weight displacement: RMSE **1.003975**, with improvement
in **all four** folds. Its residual Pearson association is **−0.913771**. Thus,
conditional on this baseline, larger displacement is associated with lower
retrieval scores; the result does **not** say moving farther improves retrieval.

Saved-segment and cumulative stable-rank fractions fail the declared predictive
criterion. Both improve **zero of four** folds, for the original measurements
and their exact counterparts. The separately defined full-spectrum saved-segment
entropy effective-rank fraction reaches RMSE **2.347064**, improving three folds;
its third held-dose fold is worse than baseline and remains in the result.
Original/exact subspace overlaps with AdamW do not pass the criterion.

The original family has **5/9** predictive flags true; the distinct exact
sensitivity family has **1/5** true. These are not significance tests or six
independent discoveries: several measurements describe related aspects of the
same sixty weight states. Every flag, null comparison and individual fold is
reported, including the high-rate configurations with poor retrieval.

## Scientific interpretation and limitations

These observations separate **optimizer fingerprints** from **retrieval
predictors**. A broad singular spectrum or stable-rank increase cannot, by itself,
explain why a particular configuration retrieves better. In this population,
displacement magnitude, segment entropy and row concentration carry information
beyond the locked baseline, while the tested stable-rank and AdamW-overlap
measurements do not meet the held-dose criterion.

Important limitations remain:

- The baseline is a specific additive linear model. Displacement can encode
  nonlinear learning-rate effects or optimizer-by-rate/stage interactions that
  this baseline omits. This is not proof of information beyond every possible
  hyperparameter-only baseline, nor does it isolate geometry from update size.
- These are fourteen separate one-feature fits, not a joint conditional model.
  They do not show that entropy adds information after displacement is included.
- The folds hold out an ordered learning-rate index across all optimizers and
  stages. They do not hold out a task, training seed, architecture or dataset.
  The edge folds extrapolate beyond the remaining observed rate range.
- The same model/data/training seed supplies all states. Four learning rates are
  not four independent seeds. The predictive flags have no declared family-wise
  significance interpretation.
- Full-spectrum entropy and top-64 renormalized entropy are different estimands.
  All five original/exact correspondences and their denominator coverage remain
  in the sensitivity tables; no original feature was overwritten.
- Functional dimension measurements and the separate crossed continuation are
  still missing. These results do not establish more retrieval-useful embedding
  dimensions or a causal mechanism for Muon/NorMuon's endpoint results.

## Inputs and calculation

Every state joins on **run identity, retained checkpoint step and the original
checkpoint seal**, not only a run name. Both geometry branches refer to the same
model-file identities and initialization. All 120 original checkpoint summary
rows equal the corresponding immutable CSV rows; both complete 660-pair tables
are retained. All 840 task scores match the accepted raw-score recovery index.

Input authentication covers **167 files**: twelve original source/protocol files,
151 immutable weight-manifest/CSV/raw-JSON files, and four accepted retrieval
index/table/verification files. The weight inputs are a deliberately selected
numerical subset of the previously fully recovered 298-file snapshot; no new
SVD, tensor loading or claim of rechecking all basis arrays is made here.

The outcome is the unchanged original summarizer's binary64 `statistics.fmean`
value. The raw recovery index also records a once-rounded exact rational macro.
Both are independently reconstructed under their original definitions; neither
is substituted for the other. Existing native geometry/score acceptance and
the new common-checkpoint join are distinct from final source-release admission.

The [local plan](plan.md) was written after scores were visible and explicitly
reuses the existing September 3/6 definitions; it is not a new preregistration.
The original functions execute as unchanged hash-bound AST nodes, without
importing a model, training code or a formal consumer. OLS uses exact rational
binary64 inputs; support compares exact MSE. Numerical-resolution and undefined
rules are unchanged. All 56 observed fits are resolved; no available-only fit,
replacement value or dropped row was needed.

The independent [verifier](source/verify_predictions.py) solves the full augmented
normal equations with SymPy 1.14.0, not the producer's Frisch–Waugh–Lovell code.
Its [receipt](independent-predictions.json) records exact agreement. This is an
independent numerical algorithm on the same data, not experimental replication.

## Reproduce locally or after copying this directory

No project package, model, GPU or network is required for this readout. Supply
absolute paths to this report directory and a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B /absolute/report/source/readout.py \
  --repository /absolute/report/source-original \
  --weight-root /absolute/report/inputs/weights \
  --outcome-root /absolute/report/inputs/outcomes \
  --index /absolute/report/inputs/raw-score-index.json \
  --output /absolute/new-weight-retrieval-readout
```

The calculation requires NumPy (actual version 2.5.2); the independent verifier
additionally requires SymPy. The report's full copied source/data replay is a
same-host relocation check, not a second-physical-host experiment. Source and
this new result remain local WIP; no new HF or GitHub publication is claimed.

Original preserved inputs are recoverable using the
[weight guide](../../docs/weight-analysis-restoration.md) and
[trajectory guide](../../docs/retrieval-trajectories-restoration.md). Do not use
historical or synthetic artifacts to fill any result. The two early adapter
failures and their source versions are retained in [attempts/](attempts/) and
[commands-first.json](commands-first.json); neither failure ran a model or
changed an original numerical definition. Engineering details are not manuscript
content. The complete paper and reproducible source release remain unfinished.
