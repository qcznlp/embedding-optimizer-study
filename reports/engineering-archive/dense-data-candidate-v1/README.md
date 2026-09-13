# Prepared revised data: fixed whole-group replacements, original evidence preserved

This local CPU milestone implements the [prospective 6/52-group proposal](../dense-data-partition-v1/revision-proposal.json).
It does not authorize formal training, publish data/source, revise old result identities, or
establish an optimizer finding. The deciding overlap audit, earlier empty-text failure and all
original datasets remain unchanged. Nothing in this engineering account belongs in the manuscript.

## Prepared data and acceptance boundary

The [producer receipt](producer.json), SHA-256
`cec0ccee8c3ebda9bf24690badc50c153c58523373664fcdc79ed270c04fde49`, records:

| Partition | Rows | Whole groups replaced | Rows retained without any field change |
| --- | --- | --- | --- |
| Training | 500,000 | 6 | 499,994 |
| Validation | 4,096 | 52 | 4,044 |

Training targets are positions 230, 634, 750, 13096, 336061 and 389365. The complete validation
target set and all old/new identities/texts are in [selection.json](selection.json). Source quotas,
positions, seed 42 / 20260826 and original mining remain fixed. Prepared serialized data is at
`/tmp/dense-partition-candidate.kHXoGW/{training,validation}/dataset`, with parent-bound manifests
and row ledgers alongside it. These are prospective inputs, **not deployed replacements**.

The full producer checks every text field, actual row-to-ledger ID linkage and source quota;
compares every old/new field to require exactly the declared difference sets; and checks all
train–validation, train–BEIR and validation–BEIR normalized-query intersections. All defined
content defects/intersections are zero, and validation is internally query-text-unique.
This does not claim semantic near-duplicate absence, annotation correctness, unique training
query strings, a retrieval effect, or natural-data GPU readiness.

| Prepared identity | Training | Validation |
| --- | --- | --- |
| Serialized fingerprint | `0c6bd82f699a563c` | `bec7e804f230a163` |
| Canonical row-ledger SHA-256 | `b08cde6c53c9c63d31d3608c6f65618bef098fbd19e0b301f6356e0b568c3b08` | `2ef651c92dc721c13849b2d1d093e80fab47dd02010c2adaf59a899ddecef4a2` |

## Selection rules and independent reconstruction

All 54 used upstream query/score/document files for FIQA, HotpotQA, NQ and FEVER match
independent immutable HF LFS digests at source revision
`1ca463331ed637d25c1058567e932e0d3bad2983`. The unchanged sources need no replacement records.
All seven original source quotas remain unchanged. The original candidate populations and
seeded priority windows are reconstructed **before** new exclusions. Validation uses the original
training-ID exclusion when reconstructing its original population, then applies the additional
prospective protections; it does not silently permute a different population.

Reserve every original training/validation source-qualified query ID and normalized query text,
including removed records, all fourteen-task BEIR evaluation query texts, and each accepted new
query. Complete training replacements before validation. Traverse original ranks; replace whole
groups at ascending target positions. Preserve the first originally eligible score record,
0.95 threshold, pool of ten and original sorted seven-of-ten per-query seed. An empty or duplicate
document group rejects its entire query, rather than switching its positive or imputing text.
[decisions.jsonl](decisions.jsonl) records the complete traversed prefix and every rejection.
No optimizer score, model prediction or downstream result selects a replacement.

The independent reconstruction uses the **unchanged original** `_scan_eligible_candidates` and
full document loader, a separately reconstructed priority prefix, and full normalized strings
rather than digest joins. It rechecks every traversed decision, including the reasons for all
skips. A row-generator serializer reconstructs both complete datasets independently of the
producer's Arrow slice/splice writer, then compares every actual row value and canonical ledger.
Different writers may have different Arrow container bytes/fingerprints; logical value equality,
not a claim of identical containers, is the relevant reconstruction check.

The actual [independent reconstruction](reconstruction.json) **passes** for both complete
datasets. Its worker exited zero. It reproduces all eight source/partition traversals, all 58
replacement groups, every complete materialized value and both canonical row-ledger digests.
The independent full-string partition intersections and validation duplicate count are all zero.
Its separately written datasets remain at
`/tmp/dense-partition-reconstruction.HsE86I/{training,validation}/dataset`.

## Preserved incomplete attempt and checks

The first preparation used a two-row-group CPU score cache, causing repeated random parquet IO
while rejecting many originally ineligible NQ queries. It was interrupted through **its exact
owned tool session**, and exited 1 with `KeyboardInterrupt`. It did not create a full selection
receipt or either dataset. Its source and all 50,628 partial decision records are preserved under
[attempt-1/](attempt-1/), including the explicitly terminal-observed interruption receipt.
No broad process enumeration, other process signal or GPU-helper interaction occurred.

The new preparer retains at most 64 CPU score row groups and records its own task identity.
No eligibility or priority rule changed. Its first 50,628 decisions are **byte-identical** to the
interrupted prefix. The cache hit/eviction test independently checks unchanged returned values.
Preserve both source versions; never relabel the incomplete attempt as a successful preparation.

All 20 initial boundary tests, 21 cached-path tests and five reconstruction tests pass. The
initial full suite has 1,626 passing cases; the final full suite has **1,632 passing cases**, with
zero failures/errors/skips. These belong to the isolated development checkout. The separate
numerical candidate's eleven earlier source-contract failures among 963 tests remain unwaived.
Passing data checks does not certify another source tree, scientific outcomes or formal readiness.

## Next step and unchanged authority

Data selection/materialization and independent reconstruction are now complete. Next prepare an explicit parent-preserving primary/validation
data amendment. The existing v2 draft and old consumers still bind original data. Preserve their
bytes and specify new identities, namespaces and exact downstream handling; do not just refresh
expected hashes. The three pre-existing derived subsets contain no target row, but their new
parent identity must still be admitted explicitly. Then declare and test a seven-source natural-
data preflight from the untrained DenseOn base through the actual prepared training entrypoint.
The original failed FIQA-only prefix is not retroactively made successful.

Formal training/deployment, owner WIP-publication direction, the pending-results pre-commit gate,
GitHub integration write access and HF deletion safety gates remain unresolved. The old 12 runs /
60 checkpoints remain preserved on scientific hold. Original numerical cores, manuscript and
the exact stopped BEIR chain/ledger/lease are unchanged. No formal run, controller transition,
publication, HF write/deletion or protected-helper action occurred. New temporary datasets and
independent reconstruction artifacts must remain available for the next source-bound review.
