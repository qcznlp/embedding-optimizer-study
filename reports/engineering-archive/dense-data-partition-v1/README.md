# Query partition audit: IDs alone did not establish text-disjointness

Before selecting replacement queries for the six empty-text groups, a read-only
check compared the actual training and validation query texts with each other and
with all **19,385 evaluation query records across 14 pinned BEIR tasks**. Query/qrel
files were independently authenticated against immutable HF LFS digests. The split
is MSMARCO dev (its `qrels_validation.parquet`) and test for the other thirteen tasks.
No model output, learning-rate score, training update or dataset mutation was used.

## Actual findings and independent replay

[protected-queries.json](protected-queries.json) reports:

- **84 training records** match **51 distinct validation queries** across the two
  partitions. Source-qualified IDs remain disjoint, as correctly reported before.
- **One training query** matches a FEVER test query: training sample 389365 / source
  query 7991, test query 154383. Validation has no such BEIR query match.
- Validation has **zero internal duplicate query texts** under the declared check.

The join initially uses lowercase, Unicode NFKD and collapsed whitespace. The
[independent full-string replay](string-replay.json) reproduces every pair without
using hashes for matching. **All 84 training–validation pairs and the one training–BEIR
pair are already equal as raw strings.** They are not normalization or hash-collision
artifacts. The 84 pairs comprise 81 FEVER→FEVER, one HotpotQA→HotpotQA, and two
MSMARCO→NQ matches. Those correspond to 48, one and two distinct validation rows.

This is an exact-query overlap audit, not semantic near-duplicate detection, a
corpus-document disjointness claim or proof that all other contamination is absent.
It does not measure an optimizer effect or explain a historical retrieval advantage.
The earlier ID-only checks and six-empty-group evidence remain true within their
stated scope and are preserved. The proposal to change only six groups is now
insufficient; its earlier archive and failed preflight are not rewritten.

## Revised replacement scope — still a proposal

The [derived proposal](revision-proposal.json) preserves training data as far as
possible and changes validation when its query was already seen in training:

| Partition | Replace whole groups | Preserve unchanged | Reason |
| --- | --- | --- | --- |
| Training, 500,000 rows | 6 | 499,994 | Five empty groups plus one BEIR query overlap |
| Validation, 4,096 rows | 52 | 4,044 | One empty group plus 51 training-query overlaps |

Training replacements retain quotas: FIQA 3, HotpotQA 1, NQ 1, FEVER 1. Validation
replacements retain quotas: FIQA 1, HotpotQA 1, NQ 2, FEVER 48. Each change replaces
an intact query/positive/seven-negative group at its original position. No text is
invented, positive switched, sample count reduced or algorithm-specific dataset made.

Candidate selection must reconstruct the **original population and seeded order before
new exclusions**, not shuffle an already shortened population. Preserve seeds 42 and
20260826, original mining/negative-sampling rules, all non-target row contents and source
quotas. Reserve all original train/validation IDs and normalized query texts, all BEIR
evaluation query texts, and accepted replacements. Training replacements precede validation.
Do not choose using optimizer outcomes. Require complete revised train–validation and
train–BEIR separation, validation uniqueness, nonempty text and independently reproducible
new materializations. The new protocol must explicitly bind both new dataset versions.

**No replacement query or revised dataset exists yet.** The next task is to implement
and verify that candidate selection/materialization, then amend downstream identities
and run the seven-source natural-data diagnostic. Do not call this proposal training-ready.
The previous v2 primary/validation locks still bind old data and must not be overwritten
or made to pass by changing expected hashes.

## Derived probes and verification boundary

All values in the 1,024-row training representation probe, the 50,000-row branch subset,
and the historical 224-query candidate-breadth subset were checked against their original
parent rows. **None contains a row targeted by the expanded 6/52 replacement set.**
The 32-query calibration is a subset of the training probe. This supports retaining
their text contents, not reusing old parent/protocol identities without an explicit
transition. Historical candidate-breadth findings do not become primary evidence.

The nine new normalized-query/namespace tests and all **1,606 isolated tests** pass with
no failures/errors/skips. Independent string replay and the 6/52 proposal were separately
executed CPU checks. The distinct numerical candidate retains its earlier eleven unwaived
source-contract failures among 963 tests. Tests belonging to one tree do not certify another.

[commands.json](commands.json), [source/](source/) and the final `validation.json` retain
actual commands, source bytes, deciding receipts, input identity checks and the previous
archive's 32 bindings via exact before-copies. All owned calls exited. No GPU worker,
formal training, old BEIR resume, controller transition, commit/push, GitHub write retry,
HF write/deletion or protected-helper interaction occurred. Original data, numerical
source, old checkpoints/protocols, manuscript and the exact stopped BEIR chain/ledger
remain preserved. This milestone is local only. Existing owner WIP/deployment direction
and publication/access gates remain unresolved. Nothing here is a manuscript finding.
