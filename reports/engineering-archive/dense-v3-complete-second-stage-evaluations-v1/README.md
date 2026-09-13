# Complete second-stage DenseOn checkpoint evaluations

All twelve corrected primary configurations at **step 1563** now have their
complete fourteen-task BEIR outcomes. The accepted cohort is **36 / 60**
checkpoints: all twelve states at steps **782, 1563 and 3907**.
Only six pool-A checkpoints / **84 task values** are new relative to the
[prior thirty-state readback](../dense-v3-second-stage-pool-b-evaluations-v1/README.md).
All thirty previous records are exactly unchanged.

The [complete second-stage table](tables/second-stage-checkpoint-scores.md)
retains every AdamW, Muon and NorMuon rate. It contains twelve equal-task
nDCG@10 means multiplied by 100, with no score-based selection. The
[168 task rows](tables/second-stage-task-scores.csv) include the six previously
accepted pool-B configurations; they are not 168 newly measured outcomes.

## Executed evidence

The unchanged original [native reader](source/readback.py), SHA-256
`765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2`,
returned zero at **2026-09-11 13:52:01 UTC**. It authenticates both original
56-file source assemblies, the original dispatcher and authority, training and
checkpoint parents, every complete native receipt, all fourteen task identities
and revisions, original successful worker records, and raw score/metadata bytes.
It did not initialize CUDA, re-run a model or recompute retrieval rankings.

The [native bundle](actual/complete-checkpoint-readback.json) is 7,499,403 bytes,
SHA-256 `6174009c4063f48bfc56e61d45a1cec636a032b6f75d6993f5cdb274b115f441`.
It retains original source snapshots and is a **local engineering artifact**,
not a public data-only upload.

The [independent raw reconstruction](actual/independent-raw-reconstruction.json)
returned zero at **13:52:43 UTC**. It checks **504 raw task values, 504 original
exit-zero workers and 576 score/metadata snapshots** directly against their
original files, and reconstructs every exact rational macro mean. The old
thirty records are unchanged; the six new records are exactly the original
pool-A step-1563 configurations, not a performance-selected subset.

A byte-exact [archived reader](source/independent_readback.py) copy returned zero
at **13:54:41 UTC**, reading the archived bundle and the original raw files.
All three CSV/Markdown outputs reproduce exactly. The
[relocated receipt](actual/source-relocated-replay.json) and [commands](commands.json)
retain actual results. This was same-host source relocation, **not an anonymous
HF recovery, a second-host model run, a new significance test or an additional
training experiment**. No new unit-test suite was needed or claimed for this
scope-only adaptation of the already executed independent reader.

## Files and replay

- [tables/second-stage-checkpoint-scores.csv](tables/second-stage-checkpoint-scores.csv):
  all twelve checkpoint means.
- [tables/second-stage-task-scores.csv](tables/second-stage-task-scores.csv):
  all 168 second-stage task scores.
- [observations.json](observations.json): exact, separately timestamped reads of
  the original primary queue, including the final **504 / 840** observation.
- [native-primary-exits.json](native-primary-exits.json): 62 original exit-zero
  records since the prior material handoff's 442-cell snapshot; this is not
  62 additional complete checkpoints.
- [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md): the prior dated
  handoff; [verification.json](verification.json) binds this local archive.
- [commands.json](commands.json): original native/independent/replayed tool
  results. The `diff` command's exit 1 means expected source differences, not a
  failed model run or failed numerical check. The native Torch import warning
  is retained; no GPU-process enumeration was performed.

On this host, with original raw paths and immutable parents still present, the
standalone reader can be used with a **new output directory**:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -I -B \
  /absolute/path/to/this-archive/source/independent_readback.py \
  --bundle /absolute/path/to/this-archive/actual/complete-checkpoint-readback.json \
  --output-directory /absolute/path/to/new-reconstruction
```

Its SHA-256 is `41e25436fecbbb1f9903548a3993b5d01369c98b79d04e96098dcea32cb7be95`.
This command is host-bound to the authenticated original raw files; it is not
yet a portable recovered-data reader. Preserve prior attempts; do not overwrite
existing output directories.

## Remaining scope

The six new pool-A outcomes are **local only** at this milestone. The earlier
[pool-B data snapshot](../../../docs/second-stage-pool-b-evaluation-restoration.md)
remains unchanged and contains only its original six configurations.
No HF/GitHub write, original controller/source change, functional recovery or
manuscript change was made here.

The original primary queues continue at step 2345 with eight live workers.
All **24 checkpoint states at steps 2345 and 3126** and their **336 task cells**
remain required. Three observed stages do not establish a complete trajectory,
convergence rate, seed robustness or retrieval-useful-dimension mechanism.
The fixed endpoint inference is unchanged. Actual functional features, crossed
continuation, portable analysis/source delivery and the final NAACL paper remain
incomplete. This engineering archive belongs outside the manuscript.
