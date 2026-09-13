# Complete v3 geometry preparation and real diagnostic verification

The v3 geometry producer and fresh numeric reader now cover the complete twelve-run/five-stage
population, the exact-entry amendment, all unordered rate pairs and explicit subspace validity.
The complete synthetic orchestration fixture produces **60 checkpoint rows, 660 run-pair rows,
60 optimizer-pair rows and 10,560 tensor/kind validity rows**, then rereads the entire result.
These fixtures are not real primary checkpoints or scientific results.

The active [v2 implementation protocol](protocol-v2.json), SHA-256
`9b6dad688072c08e8994e45f80c897a7889ab5fbcbccab27b643944c85ae625b`, remains preparation-only.
The original `9f11cbcd...` protocol, sources, output namespace and failed independent check are
preserved in [attempt-1](attempt-1/). No historical geometry or training source was rewritten.

## Definitions and admission

All twelve complete primary runs must be authenticated before any reference tensor read or output
creation. Per-run admission binds complete saved checkpoint state and the same immutable pretrained
DenseOn-unsupervised reference **before supervised adaptation**, not a random initialization.
Every hidden tensor name, shape, parameter count and retained stage must agree.

The original rank-64 sketch, rank-16 subspaces, eight oversamples, two power iterations, seed
20260903, saved-segment/cumulative anchors and all-rate equal-pair averaging are retained.
The segment at the first retained stage uses the pretrained reference; later segments use the
preceding retained checkpoint. Cumulative displacement always uses the reference. A saved segment
is not instantaneous optimizer geometry, accumulated update-norm path length or a causal effect.

Raw records preserve the old full-tensor-index-dependent sketch seed, including excluded auxiliary
tensors in that index. Exact changed-entry counts use the separately accepted amendment and include
unchanged entries in the denominator. Old matrix-mass values have their own unambiguous names.

Zero saved displacement has no defined singular subspace and is never imputed. Nonzero matrices
must support all retained directions above the declared FP32 numerical-rank threshold. Otherwise
the operation refuses them; it does not pad with null directions or silently discard that matrix.
Projected boundary gaps and captured energy are retained descriptively, not selected by a new
cutoff. Supporting sixteen nonzero directions does not prove that the rank-16 boundary is unique.

Only new output namespaces are accepted. Partial legacy files cannot be adopted. The reader
recomputes every raw metric, exact entry count and saved singular basis from the admitted weights,
then regenerates all tables. Refreshed output hashes alone cannot authenticate false measurements.
The replay is exact on the observed source/runtime with four CPU threads; no cross-host bitwise
guarantee is inferred. The public interface has no validation/BEIR score-selection argument.

## Preserved precision failure and explicit correction

The first implementation passed **49 focused / 1,940 isolated tests** and all 792 real raw-record
comparisons against the old implementation. Its independent full-spectrum audit nevertheless
failed a global Frobenius norm check: FP32 reduction gave `0.018231220543384552`, versus
`0.018231654833219556` from FP64. The relative discrepancy exceeded the predeclared `2e-5`
relative / `1e-10` absolute gate. Agreement between two uses of the same kernel was insufficient.

The [independent localization](reduction-localization.json), SHA-256
`3fd2ebb1c9fbef543c32b214b39733dcac91554b69e29110864aff64c2592ca2`, checks the unchanged twelve
preselected matrices using NumPy FP64, Torch FP64 and scalar compensated summation. Three original
reductions fail the original gate; maximum relative error is `2.3820648151802597e-5`. FP64 agrees
with compensated summation to at most `3.5527925007054823e-14` relative error. Inputs are identical.
This is a scalar measurement precision issue, not evidence of a training/optimizer/retrieval error.

The explicit v2 correction promotes **global Frobenius and displacement/weight cosine dot/norm
reductions** to FP64 on the same FP32 kernel inputs. The existing spectral kernel, row/column
statistics, singular bases, metrics' mathematical definitions, ranks, seeds and acceptance
tolerances are unchanged. The failed version has not been relabelled as passing.

## Actual v2 checks

The [real v2 rehearsal](rehearsal-v2.json), SHA-256
`0bd4d9290fd672229ee2d3a5cd4c891854c3e951fc47286bf8e8424190c5c665`, finished at 19:45:17 UTC:

- All three existing natural-diagnostic trajectories / nine checkpoints are deeply admitted.
  Their **792 hidden-matrix records** match the old implementation in every field other than the
  explicitly revised global scalars. All 792 exact-entry records match the prior accepted census.
- Independent NumPy checks cover **2,112 global norms, 1,056 defined cosines and 264 undefined
  zero-displacement cosines**. Maximum norm relative error is `8.367646539383895e-14`; maximum
  cosine absolute error is `3.2890357104520263e-15`. The original tolerances remain fixed.
- **1,056 nonzero tensor/kind cases** support the declared rank 16; **528 zero cases** remain
  undefined. Old and new pair/optimizer summaries agree exactly on these diagnostic inputs.
- Twelve preselected first-block/final-diagnostic-stage cumulative matrices pass independent full
  NumPy FP64 SVD variational bounds and scalar checks. Their saved bases also exactly match the
  old FP32 basis algorithm. This limited selection is not a full-spectrum audit of all matrices.
- Eight actual changed protocols reject. Both actual public CLI actions refuse absent complete
  v3 primary runs before producing output. No formal geometry is generated.

The [fresh CPU replay](fresh-cpu-replay-v2.json), SHA-256
`526b8e9187d29ff8113bbfa3a7bfc65a3c7708fb752b88f45ec73e3504250d90`, rechecks all nine diagnostic
checkpoints and reproduces all four tables. Two real altered output copies reject even after
their manifest hashes are refreshed: a wrong raw norm and a modified nonzero singular basis.
Original models, original output files and stopped dispatchers remain unchanged.

Final **64 focused / 1,955 isolated tests** pass, with no failures, errors or skips. The different
numerical training candidate's eleven old source-contract failures remain unwaived.
Preserve `/tmp/dense-v3-geometry.75bnhK`, `/tmp/dense-v3-geometry-retry.Pysj3k` and
`/tmp/dense-v3-geometry-replay.jd0MyI`. The successful rehearsal has 46 bound artifacts /
167,914,086 bytes plus its receipt; these include real bases and original-helper comparator outputs.
All source checkpoints and the ordinary-file pretrained reference copy remain separate dependencies.

## Scientific limitation exposed by the full-spectrum controls

The check of approximation **bounds** is not a guarantee of approximation **accuracy**. In the
twelve declared diagnostic controls, the eight Muon/NorMuon sketches underestimate the leading
singular value by about **4.20%–5.37%**, overestimating stable rank by **8.97%–11.66%**. The four
AdamW controls have much smaller leading-value error (at most about `4.47e-7` relative).
All values are retained in [full-spectrum-controls.json](full-spectrum-controls.json).

The top-64 entropy effective ranks also differ substantially from full-spectrum entropy effective
ranks, as expected for a truncated, renormalized spectrum. Neither is a direct count of embedding
dimensions useful for retrieval. The approximation errors are not uniform across these optimizer
states. **Do not call the stored approximate stable-rank columns exact ranks or use their absolute
values as an unqualified mechanism claim.** Before primary mechanism interpretation, add an explicit
exact-spectrum/error-controlled robustness check while preserving the existing approximate analysis
and its predeclared retrieval-bridge features. Do not quietly redefine those features after results.
No optimizer-quality conclusion or causal explanation follows from these short diagnostics.

## Next work and boundaries

Complete v3 geometry production/readback integration is no longer missing. Next address exact-spectrum
robustness and integrate the v3 geometry-to-retrieval bridge, functional dimension consumers and
publication, separately routed factorial, then reviewed assembled-source/runtime/release-parent
handoff. The full real primary matrix, primary geometry and mechanism findings still do not exist.
The old twelve executed runs / sixty preserved checkpoints remain on scientific hold.

All owned compute calls exited. No GPU worker, formal training, old-controller transition,
commit/push, HF mutation or protected-helper action occurred. Original numerical cores, paper,
stopped BEIR chain/ledger/lease and existing owner/access/approval gates are unchanged. New source
and evidence are local only. Every implementation incident in this archive is excluded from every
manuscript section, including its appendix. See [commands.md](commands.md) for reproduction scope.

