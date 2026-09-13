# Literal dimension interventions and fixed-probe input verification

This milestone prepares the numerical and input foundations of the v3 functional-dimension chain.
It does **not** implement the complete 61-state v3 exporter, primary publication, dimension bridge
integration or formal runtime release. No new model was encoded or trained. No optimizer-quality
or mechanism finding is inferred. Original training/analysis sources, protocols and paper are intact.

## Numerical correction and evidence

The [prospective measurement plan](measurement-plan.md) fixes the tests before these new results.
[deletion-boundary.json](deletion-boundary.json) reproduces two algebraic failures in the unchanged
old coordinate kernel:

- Deleting the only nonzero coordinate leaves an undefined cosine, but the old denominator floor
  produces a score and shortlist nDCG of one.
- With a dominant coordinate and nonzero `2**-30` remainders, subtracting the dominant squared
  norm/dot product erases the remaining direction. A genuine positive rank of eight becomes a tie
  at rank one. This is a constructed numerical boundary, not an observed optimizer effect.

The separate `dimension_interventions.py` kernel literally deletes each coordinate and applies
the same FP64 retained-vector cosine used by random deletion. A zero remaining norm rejects the
complete computation; no coordinate/query is discarded or imputed. The original attribution signs,
task-first averaging, spectrum definitions, 80 masks, three seeds, score/rank rotation guard and
shortlist interpretation remain. Its explicit numerical policy retains attribution arrays in FP64
and represents the original undefined relative-margin condition as null rather than JSON NaN.
The old source and old results are not rewritten or silently rebound.

`dimension_intervention_io.py` writes new-only single-state bundles. Readback requires a caller's
fresh numerical computation and compares every table byte, array identity/dtype/value and rotation
check. This is a portable low-level serializer, not primary checkpoint admission.

## Actual retained-vector check and fresh process

[audit.json](audit.json), SHA-256 `750a0c785c21a0efeeb23c4f9063a1d5456851bf6b6b32250935d0da38c438a2`,
uses the one previously retained **pretrained** export, with 224 queries, eight candidates and 768
dimensions. Its old float16 storage and encoding provenance are retained; no primary identity or
new model-encoding claim is attached. Independent CPU Torch FP64 calculations verify:

- **688,128 exact positive ranks / shortlist nDCG values:** 224 × 768 × four bases (native + three rotations);
- **17,920 random-mask ranks:** 224 × all 80 shared masks;
- native and rotated task-mean coordinate attributions; maximum raw margin discrepancy is about 1.8e-15;
- both fixed boundary controls, with zero remaining norm refused and the nonzero remainder retained;
- five actual changed-and-rehashed bundle copies refused after fresh recomputation.

Floating comparisons use the predeclared `rtol=1e-9, atol=1e-12`; ranking requires exact equality.
No tie tolerance or observed-data threshold was introduced. The reference decides ranks itself;
NumPy's nDCG display convention is checked against Torch's log calculation separately.

[replay.json](replay.json), SHA-256 `db1003abaea3a4439e920ddc87703a980317f8f477ddc2b80659866377d39e1b`,
repeats all calculations in a new process, authenticates the prior source/input/artifact set,
reads back the original bundle and refuses five new corrupted copies.

On this **single retained pretrained export**, native nDCG matches the old implementation exactly;
the maximum native margin difference is 1.11e-15. This neither estimates the boundary's prevalence
on other states nor measures its effect on an optimizer comparison. Computing this one full state,
including masks, rotations and spectra, took 45.1 seconds in the first audited process; it is not
a full-grid wall-clock estimate, GPU benchmark or optimizer speed claim.

## All original probe texts and candidates

[probe.json](probe.json), SHA-256 `5313b4c969ef89fbc894f1061a2fd644d4c82235030c3492fbebef2eabbfcc42`,
checks the actual five-file probe, not just its stored dataset fingerprint. Its manifest and ordered
selection ledger match the frozen specification. All 224 rows have unique exact sample/task/query
identities, positive-first ordered candidates, valid original length metadata and nonempty texts.

All **2,016 original query/document text fields** match their IDs in **42 pinned upstream BEIR
query/qrel/corpus files**. Each file's content hash/size matches independent HF LFS metadata at the
immutable revision. Query/positive seeded priorities, fourteen task quotas, relevance exclusions
and candidate-pool membership also agree. The canonical ordered row-identity digest is
`c30be6bd9b86f34c160170168014d904446a857a9eda65cbd05080b9ab1d4344`.

The frozen lexical-negative ordering is consumed and verified against its immutable ledger; the
audit does **not** claim to regenerate that lexical ranking or redraw candidates. It also does not
re-encode vectors, prove semantic relevance or create a primary model result.

[probe_replay.json](probe_replay.json), SHA-256 `34adfd308faf4250da57d51be71245f88a528a15cf12dd2f1daf740bae37b5f5`,
repeats every row/text/source check in a fresh process, using the trusted immutable-file identities
without a new network metadata request. The first wrapper attempt failed on an unresolved HF-cache
symlink; its source, exact command and refusal are preserved in [probe-attempt-1/](probe-attempt-1/).
The narrow retry resolves only that cache role, then retains ordinary-file and remote-hash checks.
Local probe symlinks still fail. No data, protocol or scientific threshold changed.

## Tests, handoff and next work

The retained test receipts report 28 initial intervention cases, 42 complete intervention cases,
20 input cases and **2,215 passing final full-suite cases**, with no failures/errors/skips. The
earlier full-suite pass of 2,195 predates the additional input tests and is also retained. The
separate training candidate's eleven source-contract failures remain unwaived.

The 1,127-line AGENTS, 601-line status page and 957-line README were consolidated into current
entry points, with every complete predecessor in [before/](before/). Previous live handoffs are
in [before-live/](before-live/). No source history, failed receipt or operational/scientific boundary
was deleted. This is local documentation maintenance, not GitHub publication. The artifact snapshot
was refreshed at 23:22:16 UTC from the live experiment working directory without process inspection.

Next bind the verified probe identities and declared numerical policy into a complete v3 61-state
export/analysis/publication/reconstruction contract. Connect the four true functional predictors to
the already verified named-family exact numerical bridge, then complete routed factorial and the
reviewed source/runtime/release-parent transition. Do not redo the finished geometry/bridge audits
or the old full historical dimension replay as missing work.

All owned calls must exit before final acceptance. The final `validation.json` checks both real
replays, all deciding files and source copies, original core/paper bytes, previous acceptance
bindings through before-copies and the exact stopped controller/ledger identities. It explicitly
records `primary_dimension_pipeline_integrated=false` and `scientific_completion=false`.
Formal training, source deployment, controller changes, commit/push, HF mutation and all protected-
helper interaction remain outside this milestone. Owner/access/release gates remain unresolved.
