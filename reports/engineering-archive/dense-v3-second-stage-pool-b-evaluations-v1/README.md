# First six complete second-stage checkpoint evaluations

The unchanged native CPU reader exited zero at **2026-09-11 12:00:20 UTC**.
It verifies **30 / 60 complete fourteen-task checkpoint evaluations**: all twelve
configurations at steps 782 and 3907, plus the original pool-B six at step 1563.
The independent raw-file reconstruction passed at **12:04:49 UTC**.

Only **six checkpoint states and 84 task values are new**. The previous 24 complete
records are exactly unchanged, including original worker and raw-file snapshots.
These are additional checkpoint outcomes, not new training runs or replications.

## New descriptive results

Read the generated [six-row checkpoint table](tables/new-checkpoint-scores.md),
[checkpoint CSV](tables/new-checkpoint-scores.csv) and
[all 84 new task scores](tables/new-task-scores.csv).
Every checkpoint mean uses all fourteen pinned full-corpus tasks with equal weight.
Rows follow optimizer/learning-rate order, not score order.

The new cohort is AdamW 3e-6 / 1e-5, Muon 3e-4 / 3e-3, and NorMuon 1e-3 / 3e-3.
It is **not the complete twelve-configuration step-1563 grid**. Six states at
step 1563 and all 24 states at steps 2345 / 3126 remain required. No optimizer
family comparison, new learning-rate selection, interpolation, statistical test,
convergence-rate claim or mechanism finding is made from this partial rate grid.
The [existing complete-endpoint inference](../../dense-v3-final-inference-v1/README.md)
and all manuscript findings are unchanged.

## Native and independent evidence

The [native bundle](actual/complete-checkpoint-readback.json) is **6,257,887 bytes**,
SHA-256 `d35f33cd6cea1294d938c28408e32ee2312477dfb285ea03f393d59f8e496439`.
The unchanged [native reader](source/readback.py), SHA-256
`765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2`,
authenticates both 56-file source assemblies, original dispatch authority,
training-completion parents and checkpoint identity. Each original native
complete-evaluation reread equals its original receipt, with fourteen declared
tasks, revisions and splits plus matching original successful workers.

The [independent reconstruction](actual/independent-raw-reconstruction.json)
checks **420 raw nDCG values, 420 original exit-zero evaluation workers and
480 raw-score/metadata snapshots**. Every exact rational mean agrees.
All 24 records from the [previous complete first-stage bundle](../dense-v3-first-stage-evaluations-v1/README.md)
remain identical. This is same-host raw-file verification, not model re-encoding,
ranking recomputation, new uncertainty estimation or second-host replication.

[commands.json](commands.json) retains the actual CPU calls and exits.
The first new independent verifier mistakenly parsed the multi-record
`run_settings.jsonl` metadata as a single JSON value and exited 1 before writing
analysis outputs. Its [source](source/independent_readback-v1.py) and original
failure are retained. The [format diagnosis](actual/metadata-format-diagnosis.json)
finds fourteen valid JSON lines in each of the thirty such metadata files.
The [fixed verifier](source/independent_readback.py) authenticates every raw byte
and parses individual JSON score/receipt files only. No evaluator, score, numerical
definition, tolerance, original failed experiment or model file was modified.
This verifier history is engineering provenance, not manuscript material.

## Reconstruct this fixed archived cohort

On the original host, with the original authenticated raw files still present,
run the archived verifier into a **new** output directory:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  reports/engineering-archive/dense-v3-second-stage-pool-b-evaluations-v1/source/independent_readback.py \
  --bundle reports/engineering-archive/dense-v3-second-stage-pool-b-evaluations-v1/actual/complete-checkpoint-readback.json \
  --output-directory /absolute/path/to/new-second-stage-reconstruction
```

The verifier intentionally requires this exact thirty-state cohort. The native
reader discovers all complete checkpoints at its execution time, so a later fresh
native readback may contain more states and is not this fixed archived input.
Original absolute raw-file and parent bindings remain mandatory. This entry is
not a portable model replay or public source release.

The [actual archived-source replay](actual/source-relocated-replay.json) exited
zero at **12:11:41 UTC**. It rechecks the fixed archived bundle against the same
original raw files; both CSV files and the Markdown table reproduce byte-for-byte.
This is same-host source relocation, not a new experiment or second-host replay.
The [final preservation check](verification.json) binds the archive payloads,
current handoff, unchanged frozen sources and manuscript identities.

## Execution handoff

At the separate **12:06:20 UTC** observation, primary coverage is **436 / 840**.
All eight original workers are live: two pool-A MSMARCO, two pool-A HotpotQA and
four pool-B step-2345 ClimateFEVER workers. Pool B advanced naturally after its
six step-1563 states completed. The brief seven-worker snapshot during the
handoff is retained; no restart, signal, manual transfer or duplicate queue was
performed. [observations.json](observations.json) contains the exact identities.

The [native exit reconciliation](actual/native-exit-reconciliation.json) accounts
for all **38** successful tasks added since the preceding 398-cell observation.
It is not 38 new complete checkpoints. The functional campaign remains at one
accepted vector state / zero feature states; its separate recovery is unapproved.
No new formal crossed-continuation branch was run.

The [prior handoff](before/CURRENT_EXPERIMENT.md) is preserved. This milestone
adds local engineering/data artifacts only: these 84 new task values are not
claimed as part of the earlier HF evaluation backups. No HF/GitHub write, source
commit, frozen numerical/dispatch edit, protected-helper access or manuscript
edit occurred. The full 840-cell campaign, functional analysis, crossed
continuation and NAACL/reproducibility deliverables remain incomplete.
