# Validation-selected final checkpoint results

Selection uses all 4096 validation rows for every one of twelve runs: lowest mean contrastive loss within each optimizer, with exact ties resolved by lower learning rate.
BEIR is joined after this fixed selection. Scores are equally weighted full-corpus nDCG@10 across all fourteen tasks, multiplied by 100.

| Optimizer | Selected LR | Validation loss | BEIR score | Difference from AdamW |
| --- | ---: | ---: | ---: | ---: |
| adamw | 3e-05 | 0.185693 | 58.9263 | +0.0000 |
| muon | 3e-04 | 0.179503 | 59.2431 | +0.3168 |
| normuon | 3e-04 | 0.179016 | 59.3637 | +0.4374 |

All twelve validation recipes are retained in [all-validation-metrics.csv](all-validation-metrics.csv).
This is a descriptive endpoint comparison, not a significance test, a multi-seed training result, a complete five-stage trajectory comparison or a causal mechanism finding.
The original scientific-completion and source-release flags remain false.
