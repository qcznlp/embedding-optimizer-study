# Post-result baseline and conditional sensitivity

Exploratory only. All errors are nDCG@10 points; lower is better. No original result is overwritten.

## B0_original

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
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
| exact_nonzero_saved_segment_stable_rank_fraction | 2.905349 | 3.010470 | 0/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 2.905349 | 2.347064 | 3/4 | 4/4 | True |
| exact_nonzero_cumulative_stable_rank_fraction | 2.905349 | 3.259430 | 0/4 | 4/4 | False |
| exact_mean_saved_segment_overlap_to_adamw | 2.905349 | 3.064663 | 1/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 2.905349 | 3.311421 | 2/4 | 4/4 | False |

## B0_original + cumulative displacement comparator

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 1.003975 | 1.247484 | 1/4 | 4/4 | False |
| saved_segment_stable_rank_fraction | 1.003975 | 1.049918 | 2/4 | 4/4 | False |
| saved_segment_sketch_effective_rank_fraction | 1.003975 | 1.220997 | 1/4 | 4/4 | False |
| saved_segment_row_norm_cv | 1.003975 | 1.142408 | 1/4 | 4/4 | False |
| saved_segment_top_1pct_row_energy | 1.003975 | 1.031840 | 2/4 | 4/4 | False |
| cumulative_displacement_to_weight_ratio | 1.003975 | 1.003975 | 0/4 | 4/4 | False |
| cumulative_stable_rank_fraction | 1.003975 | 1.244274 | 1/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 1.003975 | 0.815587 | 3/4 | 4/4 | True |
| mean_cumulative_subspace_overlap_to_adamw | 1.003975 | 1.268960 | 1/4 | 4/4 | False |
| exact_nonzero_saved_segment_stable_rank_fraction | 1.003975 | 1.056673 | 2/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 1.003975 | 1.260228 | 2/4 | 4/4 | False |
| exact_nonzero_cumulative_stable_rank_fraction | 1.003975 | 1.213301 | 0/4 | 4/4 | False |
| exact_mean_saved_segment_overlap_to_adamw | 1.003975 | 0.839774 | 3/4 | 4/4 | True |
| exact_mean_cumulative_overlap_to_adamw | 1.003975 | 1.452467 | 1/4 | 4/4 | False |

## B1_raw_rate

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 0.651744 | 0.687311 | 2/4 | 4/4 | False |
| saved_segment_stable_rank_fraction | 0.651744 | 0.523875 | 3/4 | 4/4 | True |
| saved_segment_sketch_effective_rank_fraction | 0.651744 | 0.675946 | 2/4 | 4/4 | False |
| saved_segment_row_norm_cv | 0.651744 | 0.630137 | 3/4 | 4/4 | True |
| saved_segment_top_1pct_row_energy | 0.651744 | 0.558976 | 3/4 | 4/4 | True |
| cumulative_displacement_to_weight_ratio | 0.651744 | 0.795009 | 1/4 | 4/4 | False |
| cumulative_stable_rank_fraction | 0.651744 | 1.069759 | 2/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 0.651744 | 0.650790 | 2/4 | 4/4 | False |
| mean_cumulative_subspace_overlap_to_adamw | 0.651744 | 1.020315 | 1/4 | 4/4 | False |
| exact_nonzero_saved_segment_stable_rank_fraction | 0.651744 | 0.527658 | 3/4 | 4/4 | True |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 0.651744 | 0.689136 | 2/4 | 4/4 | False |
| exact_nonzero_cumulative_stable_rank_fraction | 0.651744 | 1.000537 | 2/4 | 4/4 | False |
| exact_mean_saved_segment_overlap_to_adamw | 0.651744 | 0.601381 | 2/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 0.651744 | 1.397513 | 1/4 | 4/4 | False |

## B2_optimizer_stage_rate

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 1.390538 | 1.356792 | 4/4 | 4/4 | True |
| saved_segment_stable_rank_fraction | 1.390538 | 1.434998 | 3/4 | 4/4 | False |
| saved_segment_sketch_effective_rank_fraction | 1.390538 | 1.378448 | 3/4 | 4/4 | True |
| saved_segment_row_norm_cv | 1.390538 | 1.400785 | 1/4 | 4/4 | False |
| saved_segment_top_1pct_row_energy | 1.390538 | 1.395675 | 1/4 | 4/4 | False |
| cumulative_displacement_to_weight_ratio | 1.390538 | 1.407186 | 2/4 | 4/4 | False |
| cumulative_stable_rank_fraction | 1.390538 | 1.442451 | 1/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 1.390538 | 1.538681 | 0/4 | 4/4 | False |
| mean_cumulative_subspace_overlap_to_adamw | 1.390538 | 1.623437 | 2/4 | 4/4 | False |
| exact_nonzero_saved_segment_stable_rank_fraction | 1.390538 | 1.438452 | 3/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 1.390538 | 1.294010 | 4/4 | 4/4 | True |
| exact_nonzero_cumulative_stable_rank_fraction | 1.390538 | 1.313340 | 4/4 | 4/4 | True |
| exact_mean_saved_segment_overlap_to_adamw | 1.390538 | 1.506989 | 1/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 1.390538 | 1.456704 | 0/4 | 4/4 | False |

## B3_nominal_schedule

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 1.438786 | 1.392873 | 4/4 | 4/4 | True |
| saved_segment_stable_rank_fraction | 1.438786 | 1.596516 | 0/4 | 4/4 | False |
| saved_segment_sketch_effective_rank_fraction | 1.438786 | 1.435146 | 3/4 | 4/4 | True |
| saved_segment_row_norm_cv | 1.438786 | 1.448192 | 0/4 | 4/4 | False |
| saved_segment_top_1pct_row_energy | 1.438786 | 1.449264 | 1/4 | 4/4 | False |
| cumulative_displacement_to_weight_ratio | 1.438786 | 1.420297 | 4/4 | 4/4 | True |
| cumulative_stable_rank_fraction | 1.438786 | 1.560751 | 1/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 1.438786 | 1.676009 | 0/4 | 4/4 | False |
| mean_cumulative_subspace_overlap_to_adamw | 1.438786 | 1.924349 | 1/4 | 4/4 | False |
| exact_nonzero_saved_segment_stable_rank_fraction | 1.438786 | 1.624589 | 0/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 1.438786 | 1.356213 | 4/4 | 4/4 | True |
| exact_nonzero_cumulative_stable_rank_fraction | 1.438786 | 1.372748 | 3/4 | 4/4 | True |
| exact_mean_saved_segment_overlap_to_adamw | 1.438786 | 1.645015 | 0/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 1.438786 | 2.139937 | 0/4 | 4/4 | False |

## B3_nominal_schedule + cumulative displacement comparator

| Added feature | Comparator RMSE | Augmented RMSE | Improved folds | Defined folds | Diagnostic flag |
| --- | ---: | ---: | ---: | ---: | --- |
| log_saved_segment_to_weight_ratio | 1.420297 | 1.375658 | 4/4 | 4/4 | True |
| saved_segment_stable_rank_fraction | 1.420297 | 1.556767 | 0/4 | 4/4 | False |
| saved_segment_sketch_effective_rank_fraction | 1.420297 | 1.417966 | 3/4 | 4/4 | True |
| saved_segment_row_norm_cv | 1.420297 | 1.427902 | 1/4 | 4/4 | False |
| saved_segment_top_1pct_row_energy | 1.420297 | 1.429351 | 1/4 | 4/4 | False |
| cumulative_displacement_to_weight_ratio | 1.420297 | 1.420297 | 0/4 | 4/4 | False |
| cumulative_stable_rank_fraction | 1.420297 | 1.561906 | 1/4 | 4/4 | False |
| mean_saved_segment_subspace_overlap_to_adamw | 1.420297 | 1.645313 | 0/4 | 4/4 | False |
| mean_cumulative_subspace_overlap_to_adamw | 1.420297 | 1.335354 | 2/4 | 4/4 | False |
| exact_nonzero_saved_segment_stable_rank_fraction | 1.420297 | 1.581830 | 0/4 | 4/4 | False |
| full_spectrum_nonzero_saved_segment_entropy_rank_fraction | 1.420297 | 1.347976 | 4/4 | 4/4 | True |
| exact_nonzero_cumulative_stable_rank_fraction | 1.420297 | 1.379915 | 2/4 | 4/4 | False |
| exact_mean_saved_segment_overlap_to_adamw | 1.420297 | 1.595777 | 0/4 | 4/4 | False |
| exact_mean_cumulative_overlap_to_adamw | 1.420297 | 2.017396 | 0/4 | 4/4 | False |
