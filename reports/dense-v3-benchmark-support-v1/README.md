# Benchmark support and endpoint heterogeneity

This is a **post-hoc descriptive readout**, begun after all twelve final-checkpoint
results and their fixed inference were visible. It documents benchmark query
support and decomposes the existing equal-task macro mean. It introduces no new
hypothesis test, task exclusion, reweighting, learning-rate selection, or mechanism
claim. The full 840-cell trajectories, functional analysis, crossed continuation,
portable scientific reconstruction and manuscript release remain required.

The source reads the immutable recovered endpoint snapshot, reuses the unchanged
endpoint input reader, and authenticates every query/qrel file against the earlier
independent immutable-HF digest audit. It reads query IDs and relevance values,
not query text, corpus vectors, model weights, or GPU state. All 168 native result
revisions must match the query/qrel revisions.

## Observed result

The actual analysis exited zero at **2026-09-10 22:25:02 UTC**. Its
[readout](tables/readout.json), SHA-256
`80db28367aeae6a453e3d2c188f84f2e2c5d18a731bd699833118bf709aba6fa`,
binds all three generated tables. A separate Decimal reconstruction from the
168-cell native score CSV agrees with all **84 signed task contributions** and
all six sign summaries. Independent Polars reads agree with all **70 query-count
fields** across the fourteen tasks. The eight component tests pass. These checks
verify this descriptive readout, not additional training or statistical trials.

### Benchmark support

There are **19,385 qrel-covered query IDs** across the fourteen selected splits.
The entire NQ query file has **3,127 rows**, but its selected test qrels cover
only **26 distinct queries**, all with positive judgments. Do not describe the
query file itself as having only 26 rows. FEVER has 6,571 qrel-covered queries.
The fixed macro mean nevertheless assigns each task weight 1/14.

| Task | Qrel-covered queries | Queries with a positive judgment |
| --- | ---: | ---: |
| ArguAna | 1,373 | 1,373 |
| ClimateFEVER | 969 | 969 |
| DBPedia | 349 | 264 |
| FEVER | 6,571 | 6,571 |
| FiQA2018 | 61 | 61 |
| HotpotQA | 2,092 | 2,092 |
| MSMARCO | 4,665 | 4,665 |
| NFCorpus | 169 | 169 |
| NQ | 26 | 26 |
| QuoraRetrieval | 2,788 | 2,788 |
| SCIDOCS | 201 | 130 |
| SciFact | 41 | 41 |
| TRECCOVID | 50 | 50 |
| Touche2020 | 30 | 30 |

All splits are test except MSMARCO dev. DBPedia has **85**, and SCIDOCS **71**,
qrel-covered queries with no positive judgment in these pinned files. No query,
task, corpus entry or score is removed or recomputed here. These are dataset
support counts, not a recovered per-query ranking trace. See
[benchmark_sizes.csv](tables/benchmark_sizes.csv) for raw query-file/qrel row
counts and every immutable revision.

### All six contrasts remain visible

Signs below are descriptive signs of rounded task scores. They are not counts of
statistically significant wins. The primary family keeps all four rates; the
secondary family keeps the original loss-only validation selections.

| Family | Contrast | Positive / negative tasks | Macro difference, points | Original simultaneous-interval decision |
| --- | --- | ---: | ---: | --- |
| Primary | Muon − AdamW | 6 / 8 | +0.1569 | Inconclusive |
| Primary | NorMuon − AdamW | 7 / 7 | +0.2073 | Inconclusive |
| Primary | NorMuon − Muon | 9 / 5 | +0.0504 | Inconclusive |
| Secondary | Muon − AdamW | 10 / 4 | +0.3168 | Inconclusive |
| Secondary | NorMuon − AdamW | 12 / 2 | +0.4374 | Positive |
| Secondary | NorMuon − Muon | 10 / 4 | +0.1206 | Inconclusive |

In the **secondary** Muon−AdamW comparison, NQ's task difference is **+2.701
points**, contributing **+0.19293 macro points**, or **60.90% of the net +0.31679
macro difference**. In secondary NorMuon−AdamW, NQ (+1.419 task points) and
Touche2020 (+1.402) together contribute **+0.20150 macro points**, or **46.06% of
the net +0.43743 difference**. These tasks are identified after inspecting the
results; their reported contributions are not new confirmatory hypotheses.
Every task, including all negative contributions, remains in
[task_contributions.csv](tables/task_contributions.csv).

The 12/14 positive secondary NorMuon−AdamW signs should not be overstated:
FEVER's difference is just **+0.001 point**, one unit of the native task-score
reporting precision. The earlier paired-task intervals and all six original
decisions are retained in [contrast_summary.csv](tables/contrast_summary.csv).
Nothing here estimates query-level or training-seed uncertainty, changes the
inconclusive primary result, establishes NorMuon over Muon, or explains useful
embedding dimensions. These benchmark-scale limitations matter when interpreting
the eventual complete scientific results; they are not implementation incidents.

## Definitions

- `query_file_rows` is the number of IDs in `queries.parquet`, which may contain
  queries outside the chosen evaluation split.
- `qrel_covered_queries` is the number of unique query IDs with judgments in the
  selected split, also present in the query file. It must exactly match the
  original audited query-ID membership, not merely its count.
- `queries_with_positive_qrel` counts query IDs with at least one judgment greater
  than zero. This can differ from the qrel-covered count. No queries are removed.
- Task deltas use native nDCG@10 rounded by MTEB to five decimal places, expressed
  in points by multiplying by 100. Primary values average all four declared rates
  within each task; secondary values use the original validation-only selections.
- Each task's macro contribution is its delta divided by **14**. Signed
  contributions sum exactly to the original macro contrast, up to the old table's
  binary floating-point representation. Exact rational columns are retained.
- A fraction of the **net** macro delta is an arithmetic decomposition, not a
  fraction of improved queries, causal attribution, positive-gain mass, or a
  statistically established effect. Cancellation can make such fractions negative
  or greater than one. A zero net delta leaves that fraction undefined.
- Positive/negative task counts refer only to the signs of rounded point
  estimates, not per-task significance or independent seed replication.

Original simultaneous intervals and decisions are copied unchanged for all six
contrasts. No bootstrap is rerun. Per-query rankings are not present in the
accepted endpoint snapshot, so this analysis cannot identify which queries
changed rank or estimate per-query sampling uncertainty.

## Reproduce

Restore the [existing endpoint snapshot](../../docs/evaluation-analysis-restoration.md)
and the pinned benchmark HF cache. Supply a new output path; existing outputs are
never overwritten. This command is a CPU-only descriptive analysis, not a training
or evaluation launcher:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/python -B reports/dense-v3-benchmark-support-v1/analyze.py \
  --repository /absolute/path/to/repo \
  --evaluation-root /absolute/path/to/recovered/immutable/endpoint/prefix \
  --hf-cache /absolute/path/to/huggingface/hub \
  --output /absolute/path/to/new-benchmark-support
```

The cache must contain each task's original `datasets--OWNER--NAME/blobs/SHA256`
query/qrel objects. The source does not download, modify, or substitute datasets.
`readout.json` records all 28 exact file identities and the actual PyArrow version.
The eight small `unittest` cases exercise counting and arithmetic boundaries;
they are not model executions or statistical evidence.

## Execution provenance and unchanged work

[commands.json](commands.json) preserves the actual analysis, component-test and
independent-count/arithmetic commands with their terminal results. The earlier
read-only count inspection completed all fourteen file checks but then tried the
wrong manifest filename (`manifest.json` instead of `artifact_manifest.json`);
its exit-one output is retained, not described as an evaluation failure or a
successful complete command. No dataset or result was changed by that inspection.

The first read-only preservation check also used the wrong PDF location
(`paper/main.pdf` instead of the existing `paper/build/main.pdf`). Its
[original failed command](verification-attempt-1.json) is retained. The corrected
check uses the actual existing PDF and the same expected digest; no manuscript
was rebuilt or altered to make the check pass.

The [observer snapshot](observations.json) records **186/840** native primary task
cells, eight exact live/R workers, and the live functional coordinator with
**0/61** encoded states at 22:26 UTC. No dispatcher, numerical source, lease,
historical process, HF payload or manuscript changed. The prior current-handoff
document is preserved under [before/](before/CURRENT_EXPERIMENT.md).
This is local analysis provenance, not a GitHub or Hugging Face source release.
