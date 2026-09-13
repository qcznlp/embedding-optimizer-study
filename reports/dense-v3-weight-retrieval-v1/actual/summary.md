# Complete weight-to-retrieval held-dose numerical readout

Actual sixty-state fixed-grid numerical prediction readout using unchanged pure kernels. Not formal whole-primary/source admission, significance, mediation, causal explanation, functional dimension utility, seed replication, source publication or paper completion.

RMSE is on nDCG@10 × 100. The same 60 states, eight-column baseline and
four held-dose folds are used for every feature. Support requires lower
pooled error and improvements in at least three of four folds.

## Original

| Feature | Baseline RMSE | Added-feature RMSE | Improved folds | Defined folds | Predictive support |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 2.905349 | 2.310434 | 3/4 | 4/4 | True |
| saved_segment_stable_rank_fraction | 2.905349 | 3.018061 | 0/4 | 4/4 | False |
| saved_segment_sketch_effective_rank_fraction | 2.905349 | 2.384369 | 3/4 | 4/4 | True |
| saved_segment_row_norm_cv | 2.905349 | 2.376234 | 3/4 | 4/4 | True |
| saved_segment_top_1pct_row_energy | 2.905349 | 2.416472 | 3/4 | 4/4 | True |
| cumulative_displacement_to_weight_ratio | 2.905349 | 1.003975 | 4/4 | 4/4 | True |
| cumulative_stable_rank_fraction | 2.905349 | 3.256706 | 0/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 2.905349 | 3.143705 | 1/4 | 4/4 | False |
| mean_cumulative_subspace_overlap_to_adamw | 2.905349 | 3.742949 | 0/4 | 4/4 | False |

## Exact

| Feature | Baseline RMSE | Added-feature RMSE | Improved folds | Defined folds | Predictive support |
| --- | ---: | ---: | ---: | ---: | --- |
| exact_nonzero_saved_segment_stable_rank_fraction | 2.905349 | 3.010470 | 0/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 2.905349 | 2.347064 | 3/4 | 4/4 | True |
| exact_nonzero_cumulative_stable_rank_fraction | 2.905349 | 3.259430 | 0/4 | 4/4 | False |
| exact_mean_saved_segment_overlap_to_adamw | 2.905349 | 3.064663 | 1/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 2.905349 | 3.311421 | 2/4 | 4/4 | False |

No family-wise significance test is defined for these predictive flags.
Residual associations are descriptive, and the omitted dose still belongs
to the same optimizer families, model, data and training seed. This is not
a held-task, held-seed or external-model validation. Exact full-spectrum
entropy and original truncated entropy are distinct measurements.
