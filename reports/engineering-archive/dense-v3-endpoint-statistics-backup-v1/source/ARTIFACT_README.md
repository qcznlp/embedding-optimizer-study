# DenseOn: complete final-checkpoint optimizer contrasts

This snapshot preserves all six previously computed endpoint contrasts, their
paired-task confidence intervals, tables and figures. It is **not a completed
trajectory study, a multi-seed experiment, a causal explanation or a code release**.

![All primary and secondary endpoint contrasts](figures/endpoint_contrasts.png)

Read [the complete generated table](tables/summary.md),
[primary task effects](tables/primary_task_effects.csv),
[secondary task effects](tables/secondary_task_effects.csv), and
[all plotted values](figures/figure_data.csv). The same figure is available as
[PDF](figures/endpoint_contrasts.pdf) and [SVG](figures/endpoint_contrasts.svg).

The primary comparison averages all four declared rates within each of fourteen
tasks before contrasting optimizers. All three primary simultaneous intervals
cross zero. The secondary comparison fixes each recipe using the complete
validation loss only: AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4. Its NorMuon-minus-AdamW
effect is +0.4374 nDCG@10 points on a 100-point scale, with simultaneous 95% interval
[+0.1463, +0.7285]. The other two secondary comparisons are inconclusive.
One positive comparison and one inconclusive comparison do not establish a
difference between treatments. Inconclusive does not mean equivalent.

Each family uses the original 50,000 common paired-task bootstrap draws, seed
20260903, fixed observed standard errors and a maximum absolute centered statistic
over its three contrasts. Simultaneity is within each family, not across all six.
These intervals concern task variation for one training seed; rates are not seeds.

[The native readout](tables/readout.json) records the full definitions, source
identities, numeric outputs and independent scalar comparison. Its source-relocated
replay receipt is under provenance. Original absolute host paths in those records
are historical locations, not dependencies to fetch from this dataset. The
analysis plan was written after endpoint scores were visible, explicitly reusing
already frozen inference definitions; it is not a new preregistration.

The original 168 final BEIR task scores, twelve validations and validation-only
selection are preserved in the same dataset at immutable revision
`3883b677f87b1982f06016e9fadb8bb95e0cfc96`, under
`corrected-dense-correctness-v3/complete-endpoint-evaluations/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f`.

[artifact_manifest.json](artifact_manifest.json) binds all fifteen payload files.
This is the data/figure portion of the original verified analysis, with unchanged
bytes. Executable analysis source is not included or claimed publicly available.
Full intermediate-checkpoint evaluation, functional dimensions, crossed continuation,
complete source reconstruction and the NAACL paper remain unfinished.
