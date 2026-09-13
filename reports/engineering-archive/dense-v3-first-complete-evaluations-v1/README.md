# First six complete v3 checkpoint evaluations

At **2026-09-10 14:59:43 UTC**, all six original pool-B final checkpoints have
complete native fourteen-task readbacks. The independently reconstructed
[score table](tables/summary.md) and [84 task values](tables/task-scores.csv)
are actual full-corpus evaluation outputs, not synthetic results or partial-task averages.

The cohort is the original fixed pool-B queue: AdamW 3e-6 / 1e-5, Muon 3e-4 /
3e-3, and NorMuon 1e-3 / 3e-3, all at step 3907. It was not selected by performance.
Each macro score weights the same fourteen pinned tasks equally and is displayed
as nDCG@10 × 100. The other six final checkpoints and all 48 intermediate
checkpoints remain required. **This is not a complete optimizer comparison,
validation-selected winner, significance test, mechanism result or paper release.**

## What was actually verified

The unchanged original dispatcher produced each `all-fourteen-tasks-verified.json`.
The new CPU-only [readback](source/readback.py) authenticated the original
dispatcher/authorization and both 56-file training assemblies, loaded the original
content-bound contract, bound each model to its original deep-verified training
completion, and called the unchanged native `inspect_evaluation` reader.
All six results exactly matched their original complete receipts.
This includes native checkpoint payload rehashing, task revisions/splits/subsets,
runtime/model metadata, all fourteen genuine original worker exit-zero records,
and stable raw score/shared metadata files.

The first actual invocation read five completed checkpoints; a fresh invocation
then read the complete six-checkpoint cohort. Their five common records are the
same evidence, **not extra independent experiments**. Both actual tool sessions
(6325 and 6748) terminated with exit zero. The second full observation is
[retained verbatim](actual/first-six-complete.json), 1,296,459 bytes, SHA-256
`bc57005e4adf883b19a426fe461b05197585b413eb7de15512e30eb5d39a8e3b`.

A separate standard-library [table reconstruction](source/export_tables.py)
re-read the archived original MTEB JSON and checked its bytes against the live
originals. It independently extracted all 84 nDCG fields and recomputed each
14-task mean using exact rational representations of the stored binary64 inputs.
All six means and all 84 values match the native readback. It does **not**
recompute retrieval rankings or per-query nDCG from model embeddings.
The generated CSV/Markdown hashes are in [the table manifest](tables/manifest.json).

Original draft/source-release flags remain false. No old single-selection guard,
scientific criterion, task denominator, dataset revision or numerical source was
changed. No historical scores, partial means, new model inference, training,
GPU scheduling transition, remote write, Git commit, or protected-helper access
was involved. No generated manuscript include or manuscript text was changed.

## Operational snapshot

At **15:04 UTC**, the exact existing observer reports **101 / 840** accepted
primary task cells, **14 / 14** baseline tasks, and eight live evaluation workers
at its separately recorded task-snapshot time. Six of 60 primary checkpoints
have complete fourteen-task coverage. Pool B has moved to the original step-782
queue; pool A is still evaluating final checkpoints. Validation remains **7 / 12**
and functional encoding **0 / 61**, both exact coordinators live.
See [the dated observations](observations.json); they are not a live heartbeat.

The one-GPU priority request remains unanswered, not approved. Existing sources,
workers, leases and pending queues remain untouched. Continue those jobs; do not
rerun the twelve completed training recipes or duplicate evaluation.

## Reproduction and scope

[commands.json](commands.json) records the actual calls and terminal outputs.
The [before copy](before/CURRENT_EXPERIMENT.md) preserves the previous handoff.
Initial import-format source versions and the read-only path-discovery failure
remain disclosed; they are not model/evaluation failures. The final readback
and final table exporter passed Ruff. The simple arithmetic fixture is not
a model or retrieval test.

The readers use this host's original pinned source/data/model tree. The native
reader observes all fully complete checkpoints present at its start; the table
exporter intentionally accepts only this exact six-checkpoint pool-B snapshot.
From this archive directory on this host, repeat the raw-score table reconstruction
with a fresh output:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  source/export_tables.py \
  --input actual/first-six-complete.json \
  --output /absolute/new/output-directory
```

This archive is local evidence, not remote source publication, a portable
second-host experiment, full primary-grid admission or overall scientific completion.
