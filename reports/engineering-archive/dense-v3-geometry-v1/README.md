# Changed-entry measurement correction and real-checkpoint verification

**This is one verified prerequisite for v3 weight-space analysis, not a completed geometry
pipeline or a scientific finding about optimizers.** The original geometry sources and historical
outputs remain unchanged. The new [measurement amendment](amendment.json), SHA-256
`a4b4f6839b185931d2de7f0771487123b971a1e7eb69ef34bb5f9e4350060ec5`, is preparation-only.

## What was wrong, and what is now measured

The old `_nonzero_parameter_fraction` sums the sizes of matrices with nonzero displacement.
It therefore measures **parameter mass in changed matrices**, not the fraction of individual
parameters that changed. A 2×2 matrix with one changed value returns 1.0 through the old helper,
whereas the intended entry fraction is 0.25. This mismatch is reproduced against the unchanged
helper in [counterexample.json](counterexample.json) and the focused regression test.

The new `primary_weight_entries.py` component compares actual saved values with exact inequality
after lossless float64 conversion. It uses no tolerance, thresholded norm or gradient proxy.
Its denominator is all **110,297,088 parameters in 88 hidden matrices**, including unchanged
entries and zero-displacement matrices. The saved-segment anchor is initialization at the first
retained stage and the previous retained checkpoint thereafter. Cumulative displacement always
uses the same immutable untrained initialization. The old matrix-mass quantity is retained under
an explicit different name; old artifacts are not rewritten.

## Actual evidence

The [CPU rehearsal](rehearsal.json), SHA-256
`a269aa6f60ad931a5bd7998c6009b9af8f5a5e93d7f23acc33fd3ea586f84b49`, completed at 18:47:11 UTC.
It authenticates all prior natural-diagnostic artifacts, the complete saved run state and every
base/checkpoint payload before and after reading. It counts all hidden entries for AdamW, Muon
and NorMuon across their existing three-step diagnostic trajectories: **9 checkpoints** in total.
Independent NumPy array comparisons agree with all **1,584 matrix/anchor counts** and all global
fractions. Original weights are unchanged and no model/GPU update is performed.

The [fresh CPU replay](fresh-cpu-replay.json), SHA-256
`233ebd74d5645b0cc02213c77990be5827c6992032b307887aa0330627127625`, completed at 18:50:20 UTC.
A new process authenticates the inputs again, validates the exact tensor population/identities/
anchors, and independently recomputes all 1,584 counts from arrays. Six altered census files
reject: wrong anchor, denominator, duplicate tensor, missing tensor, definition and a false count.
The last case has an internally consistent recomputed summary but still fails against actual
weight values. Six altered measurement amendments separately reject in the first rehearsal.
The real primary CLI refuses absent complete v3 runs before any primary census/output.

At diagnostic step 1, all entries are unchanged, consistently with the retained zero-LR first
step. At steps 2 and 3, all matrices have some change, so the old proxy is exactly 1.0, but the
actual entry fractions are strictly below 1.0. Complete per-matrix counts are retained in
[adamw-census.json](adamw-census.json), [muon-census.json](muon-census.json) and
[normuon-census.json](normuon-census.json). These are short-run measurement checks, **not** an
all-rate comparison or evidence that one optimizer uses dimensions better.

## Preserved failures and tests

The first actual reference preflight failed before counting: HF's immutable snapshot uses soft
links, while the strict reader requires ordinary files. The failed wrapper and
[failure record](attempt-1/failure.json) remain intact. A separately supplied ordinary-file copy
of exactly the same eleven base files is accepted only after matching the existing immutable
digests. Both the copy and original resolved blobs are rechecked. No symlink rule is relaxed.
A second wrapper issue found by inspection—relative metadata overwriting an absolute binding
path—was corrected before retry and covered by a regression. It was not reached by the failed
execution. The amendment and measurement module themselves did not change after freezing.

The initial 44 focused / 1,890 isolated tests passed. Final focused tests pass **45 cases**.
One full-suite invocation used a relative `PYTHONPATH`; two copied-paper subprocesses then
loaded the live checkout's old layout checker. Its 1,891-case report with two failures remains
in [full-tests-final.xml](full-tests-final.xml), explained in
[test-invocation-failure.json](test-invocation-failure.json). The unchanged suite rerun with
explicit absolute source paths passes **1,891 cases, zero failures/errors/skips**, in
[full-tests-final-absolute-path.xml](full-tests-final-absolute-path.xml).
No paper, test or acceptance criterion was changed to pass that rerun. These isolated results
do not clear the different numerical candidate's eleven unwaived source-contract failures.

## Interpretation and next work

Changed coordinate entries are neither independent directions nor functionally useful embedding
dimensions. The count depends on saved-value precision and the coordinate basis; a dense rank-one
change can affect every entry. Do not turn this corrected diagnostic into the paper's mechanism.
The paper still needs weight trajectories linked to functional dimension interventions and
retrieval, with rate sensitivity and the bounded crossed state/operator control.

An additional synthetic boundary test shows that fixed-rank subspace bases can include null
directions: two orthogonal rank-one 2×2 matrices have overlap 1 when both retained bases span two
dimensions, and 0 when retaining only their signal direction. The old subspace kernel is unchanged.
This is **not evidence of rank deficiency in the real primary rank-16 analysis**; that condition
has not been audited. Zero-displacement subspaces remain undefined and must not be imputed.

Next integrate the full v3 spectral/subspace/geometry reader using the correct entry census,
then dimension/retrieval/publication consumers, routed factorial and the reviewed assembled-source/
runtime/release-parent transition. The public census component already requires all twelve complete
v3 primary runs before any 60-stage promotion; no such primary population exists yet.

Preserve the original diagnostic checkpoint directory `/tmp/dense-natural-readiness.a4xZzR`,
the ordinary base copy `/tmp/dense-weight-entry-reference.Um8TxW`, the failed empty namespace
`/tmp/dense-weight-entry-census.zSu9CA`, the successful census
`/tmp/dense-weight-entry-census-retry.DgbmCc` and replay `/tmp/dense-weight-entry-replay.a9BFoN`.
The latter two hold 10 / 6 declared artifacts plus receipts; the base copy is about 0.60 GB.
See [commands.md](commands.md) for the exact CPU-only invocations and fresh-output requirements.

All owned compute calls have exited. Original numerical cores, manuscript, exact stopped BEIR
chain/ledger/lease and old 12-run/60-checkpoint scientific hold remain unchanged. No formal run,
controller transition, source publication, HF mutation or protected-helper action occurred.
Owner WIP/deployment direction and existing access/approval gates remain unresolved. This local
engineering milestone is excluded from every manuscript section.

