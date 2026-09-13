# Coordinate utility and held-dose retrieval

Actual corrected DenseOn results, completed and independently checked on
2026-09-12. This is a scientific result record, not the final manuscript.
The complete population is the pretrained model plus all 60 primary checkpoints:
224 fixed queries, fourteen tasks, eight ordered candidates and 768 coordinates.

## What the results support

Muon has a small **native-coordinate helpful-participation increase** over AdamW:
**+0.00358455**, simultaneous 95% interval **[+0.00014359, +0.00702552]**.
This is the only supported contrast in the fixed nine-contrast family. The other
eight intervals include zero. Neither Muon nor NorMuon satisfies the predeclared
joint criterion requiring all three favorable coordinate-utility differences.
Failure of that criterion does not establish equivalence.

The Muon–AdamW helpful-participation difference changes sign under two of the
three shared rotations: **−0.00136554, −0.00165953, +0.00135763**. Thus the native
increase is not a coordinate-free dimension-use advantage. Rotation preserves
full cosine scores and ranks; deletion sensitivity depends on the chosen basis.

Randomly removing half the coordinates retains **99.1638% / 99.3778% / 99.2056%**
of shortlist nDCG for AdamW / Muon / NorMuon. This is a fixed eight-candidate
probe, not a compressed-embedding full-corpus benchmark.

## Which functional measurement predicts retrieval?

All four declared predictors were tested on the same complete 60-state panel.
The original nine weight-space columns remain intact. The fixed comparator
contains optimizer, stage and centered within-optimizer log learning rate.
Four held-out-dose folds each contain 45 training and 15 test states.

| Added measurement | Comparator RMSE | Augmented RMSE | Improved folds | Fixed criterion |
| --- | ---: | ---: | ---: | --- |
| Helpful mass share | 2.90535 | 2.94426 | 2/4 | Not supported |
| Helpful participation | 2.90535 | 3.33971 | 1/4 | Not supported |
| Degrading margin-attribution mass | 2.90535 | **1.89844** | **4/4** | Supported |
| Relative nDCG after 50% removal | 2.90535 | 2.87118 | 2/4 | Not supported |

RMSE is expressed in nDCG@10 points, on a 0–100 scale. Decisions use exact
unrounded MSE comparisons, not the rounded values in this table.

The supported predictor has a **positive**, not negative, residual association
with retrieval: Pearson **0.81476**, Spearman **0.86079**. Larger mass of
coordinate deletions that improve probe margin is associated with better
full-corpus retrieval after the declared controls. This does not mean deliberately
adding harmful coordinates improves retrieval: deletion is followed by cosine
renormalization, and these are sensitivities rather than additive semantic units.
It also does not establish that Muon benefits by reducing this mass; the optimizer
contrast for this measurement is inconclusive.

These results separate an optimizer-associated native-basis change from a
retrieval-predictive measurement. They do **not** support “Muon wins because it
uses more helpful dimensions.” The subsequent
[all-four-feature sensitivity](../dense-v3-functional-sensitivity-v1/README.md)
is now complete: no functional feature passes all four recipe comparators.
Degrading mass's B0 benefit is not robust to richer rate/stage controls. This
exploratory qualification does not alter the original results above. No mediation
or unseen-task generalization is claimed; the primary trajectories have one seed.

## Complete outputs and verification

- [All tables](actual/tables.json): 9 primary contrasts, 27 rotation contrasts,
  68 figure points, 60 joined states, 16 folds, 4 predictive summaries,
  4 residual associations, 240 held-out predictions and 15 stage summaries.
- [Readable tables and vector plots](figures/readout.pdf).
- [Actual computation record](actual/readout.json), SHA
  `608965f5297fc196adcce6ca7417bd40e7f35d05e26133f20c0811dd1fd7027e`.
- [Independent verification](actual/independent_verification.json), SHA
  `03f9670ab2a8a2334db439c95f1ef6264485cb816addd48457df305f64b45e23`.

The independent calculation uses Torch FP64 multiplicity-weighted common
resamples for all task contrasts and full SymPy rational normal equations for
all 240 predictions. It checks the same observations; it is not another training
experiment. Every actual state was reconstructed from original vectors and
compared with all saved feature tables and attribution arrays. Full computational
provenance and preserved attempts are in the separate
[engineering chain](../engineering-archive/dense-v3-functional-result-chain-v1/README.md).
The two-page readout was visually inspected, with no Type 3 fonts or overflow.
It is not the final NAACL layout or a manuscript installation.

## Numerical replay from copied source and data

The actual copied-source replay completed at **2026-09-12 17:09:14 UTC**
(session 63428, exit zero). All nine table families, all decisions and the
generated TeX match the saved actual calculation exactly, without repeating
model encoding or coordinate deletion. The [replay receipt](numerical-replay/verification.json),
SHA `752954559ebd2cfbcebe835582031912a18c17cf355fc1668c203f1a6dfffb9a`,
binds all ten numerical inputs, the 42 copied original source files and the
explicitly added byte-identical package initializer. That initializer is not
retroactively described as part of the earlier 686-file archive receipt.

From this repository, using a new absolute output directory:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B reports/dense-v3-functional-inference-v1/replay.py \
  --output /tmp/functional-replay-NEW
```

This is saved-table numerical reconstruction. The metadata-only population view
does not admit models or replace the preserved native primary/data readers.
Checkpoint eligibility, actual encoding provenance, cross-host GPU recovery
and final publication remain separate claims. The full raw functional and
calibration backup has also been [downloaded and verified](../../docs/functional-analysis-restoration.md).
