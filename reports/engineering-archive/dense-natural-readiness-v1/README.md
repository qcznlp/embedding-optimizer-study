# Natural-data readiness: stopped before GPU execution

The real-data diagnostic did **not** start training. Its fixed original 288-row prefix
contains an empty positive at index 230. The unchanged nonempty-text check rejected it
before materialization, GPU leasing, torchrun or model loading. The empty temporary
directory and [terminal-observation record](failed-prefix-attempt.json) are retained.
That record is not a worker-produced success receipt. No sample was skipped and no
acceptance rule was loosened. The prefix is FIQA-only because the original dataset is
source-concatenated; it would not by itself establish seven-source execution coverage.

## Exhaustive content evidence

The [actual 500k-data audit](full-data-content.json) checks all **4,500,000 text fields**
and every materialized query/positive/negative ID against its original row ledger.
All **500,000 rows match their IDs**, but **five rows (0.001%) contain exact empty strings**:

| Original index | Source | Empty role | Document ID |
| --- | --- | --- | --- |
| 230 | FIQA | Positive | 741 |
| 634 | FIQA | Negative 4 | 741 |
| 750 | FIQA | Negative 4 | 741 |
| 13096 | HotpotQA | Negative 3 | 104163 |
| 336061 | NQ | Negative 3 | 2143 |

Queries are nonempty, all null counts are zero, and no further whitespace-only text
was found. Exact **nonempty** positive–negative text collisions and repeated-negative
texts are both zero within the fixed eight-document groups. This does not rule out
near-duplicates, semantic false negatives or annotation errors. The original ID/file
and fingerprint checks did not establish this text-eligibility property; their earlier
successful receipts are preserved, not relabeled as content checks.

The [upstream-origin audit](empty-origins.json) traces all five cells to **three exact
empty source documents**. Their actual parquet files match independent HF LFS digests
at `lightonai/embeddings-fine-tuning@1ca463331ed637d25c1058567e932e0d3bad2983`.
This is not a tokenizer, attention or optimizer-origin failure. The materializer copies
the chosen document text without a nonempty-text eligibility filter. The separate first
FIQA observation authenticates its document/query/score files; query 5009 has two score
rows with different positives. No alternative positive was substituted or promoted.

The [frozen validation audit](validation-content.json) verifies its original expected
manifest and files, all **4,096** actual row identities and **zero training-query overlap**.
It finds **one empty negative**: validation index 46, FIQA query 5155, negative 0/document
741. There are no exact nonempty positive–negative or repeated-negative text collisions.
Validation text inspection did not read any model outputs or select any learning rate.

These observations do **not** estimate an effect on retrieval, explain an optimizer
advantage, or reverse the separate duplicated-gradient-normalization scientific hold.
The tiny frequency should not be inflated into a proposed scientific contribution.
Nothing in this engineering archive belongs anywhere in the manuscript.

## Source, tests and current boundary

The diagnostic launcher/worker, full-content scanner, source-origin and frozen-validation
audits are retained under [source/](source/). The first two remain **unexecuted on GPU**.
The real CPU content/origin/holdout audits exited zero while explicitly reporting failed
text eligibility. All raw inputs, old checkpoints, source locks and candidate cores are
unchanged. The old 500k row ledger has raw-file SHA `113cc571...`; `735ef35b...` is its
canonical row-content checksum, not the raw-file SHA.

All **59 focused tests** and **1,597 isolated repository tests** pass, with no failures,
errors or skips; the earlier 46/1,584 receipts remain. Synthetic scanner tests cover
null/empty/Unicode whitespace, exact versus nonexact duplicates, role distinctions and
actual-ID mismatch rejection. They do not make real data eligible. The separate prepared
numerical candidate still has its earlier 963 tests / eleven unwaived source-contract
failures. The origin and validation wrappers were exercised by their actual CPU receipts;
they are not additional successful training runs.

[commands.json](commands.json) records the actual calls and outcomes. Run the same full-data
audit with a new `--output` path to reproduce the observations; it is CPU-only and does not
filter or modify the original data. Final [validation.json](validation.json) preserves the
preceding 41 bindings via before-copies, authenticates actual data and source files, checks
the real counterexample and upstream documents, and verifies the exact retained dispatcher
handles, ledger, numerical source and manuscript.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B -m scripts.validate_dense_natural_readiness \
  --repository /root/embedding-optimizer-story-refactor \
  --output /tmp/new-natural-readiness-validation.json
```

An initial handoff call supplied a nonexistent output parent after transient orchestration
state was unavailable. Its terminal failure is retained in `validation-initial-terminal.json`;
no receipt was produced or reported as a pass. The final call uses an explicit absolute path.
No source, scientific threshold or failed text-eligibility decision was changed.

## Next action

Prepare/review the [uniform data-revision proposal](data-revision-proposal.md), including
the held-out partition, before replacing inputs or calling a natural-data preflight ready.
Do not simply overwrite six strings, choose another positive for the same query, use
499,995 training rows, change data for only one optimizer, or refresh old expected hashes.
The new primary v2 proposal still binds the old data and is **not execution-ready**.

The original primary remains 12/12 executed and 60/60 preserved, on scientific hold.
All owned calls have exited; no GPU worker ran in this milestone. The original three
BEIR dispatchers remain stopped with their unchanged ledger/lease/four-step prefix.
No formal training, original evaluation resume, controller transition, commit/push,
GitHub write retry, HF write/deletion or protected-helper interaction occurred.
Source/evidence are local only; owner WIP/deployment direction and the existing publication
gate/access issue remain unresolved. No paper result is supplied by this milestone.
