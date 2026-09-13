# Complete validation selection and final-checkpoint comparison

All **12 original validation workers completed successfully**, each scoring the
same 4,096 held-out query groups. The original coordinator wrote its completion
and selection records at **2026-09-10 16:54:03 UTC**. A fresh native readback at
**16:58:37 UTC** (reader-reported time) verified all saved inputs and results and
reproduced the original selection exactly.

The [generated selected-endpoint table](tables/selected-endpoints.md) joins the
fixed validation choice to the already complete fourteen-task BEIR endpoints.
It is a descriptive comparison: no significance, multi-seed robustness, complete
training-trajectory result or weight-space mechanism is claimed. All twelve
recipes remain visible in [all-validation-metrics.csv](tables/all-validation-metrics.csv)
and the separate [complete BEIR endpoint grid](../dense-v3-all-final-evaluations-v1/tables/summary.md).

## What determines the learning rate?

The original fixed rule minimizes mean contrastive loss over **every one of the
4,096 validation rows**, independently within each optimizer. Exact ties choose
the lower learning rate. It is not a mean of source means; positive margin is
not a tie-breaker. Neither BEIR nor a new numerical tolerance enters selection.

The selected recipes are AdamW **3e-5**, Muon **3e-4**, and NorMuon **3e-4**.
The [original selection](actual/all-twelve-validation-selection.json) is
128,112 bytes, SHA-256
`580321b217bb443e739656196858b4a87a094b3c4c883aa9baa9686eed219be6`.
The [original completion](actual/completed.json) is 278 bytes, SHA-256
`81f39a1c0ae9123ddcbc226e93021b25cbc34537112b31e109f3610ee6c421a5`.
These are byte copies, not replacement operational receipts.

## Actual numerical and provenance checks

The [readback source](source/readback_validation.py) authenticates the unchanged
original dispatcher and its operational authorization. Its original source
loader, revised data reader and checkpoint admission reconstruct every genuine
run plan. The unchanged native collector then reads **49,152 scored rows**, with
**393,216 candidate cosine values**, and verifies **294,912 scalar metrics** against
the existing independent scalar-float64 reference. Raw FP32 metrics remain the
selection inputs; the reference does not replace them or change any tolerance.

All twelve original started/exited/scored/admission chains agree with their
native results, including twelve observed worker exit codes of zero. An additional
direct read reconstructs all **72 ordered raw-metric means** and applies the
loss-then-lower-rate rule separately, matching the original selector exactly.
The CPU-only readback invocation (session 4061) exits zero. It neither loads a
model for inference nor recomputes any forward pass, reads BEIR scores, acquires
a GPU lease or creates a CUDA context.

[all-twelve-validation-readback.json](actual/all-twelve-validation-readback.json)
is 427,618 bytes, SHA-256
`1bfad0688b108f1e0546af888b508b9a3d5de87350e46f925f57585e04b545c7`.
It retains the native selection, original worker records and content identities
of all 48 raw validation output files. The actual large per-row score files remain
at those original paths; this small archive alone is not their off-host backup.

The [separate table exporter](source/export_selection_tables.py) reconstructs
selection before opening BEIR. It then requires all twelve complete endpoints
and joins only by the already chosen run IDs. The archived exporter replays all
three CSV/Markdown files byte-for-byte; path-bearing manifests retain the actual
locations of each invocation. Replays are not extra models or experiments.

## Runtime handoff and scope

The validation coordinator's exact former handle was PID 13007/start 298006520.
Its live-only inspection subsequently reported that the handle was missing.
The complete original receipts and all twelve worker exits were then read; no
coordinator was restarted. The coordinator's own OS exit code was not observed
and remains null, distinct from the twelve verified worker exit-zero records.
Do not keep polling its stale live-only handle, run another `--watch`, or use
the old empty-pilot `--inspect` preflight on these completed outputs.

Functional analysis still covers the base plus **all 60** primary checkpoints,
not only these three selected endpoints. Its original validation-priority gate
is now satisfied; its exact coordinator remains responsible for obtaining a
free GPU through both original lease namespaces. The old waiting-for-validation
file is a preserved historical event, not proof that validation is still missing.
No resource handoff or dispatch/source/protocol change was performed here.

[observations.json](observations.json) contains dated runtime observations.
[commands.json](commands.json) preserves the actual reader/exporter/copy calls and
the stale live-only observation. [verification.json](verification.json) binds this
milestone; [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md) preserves the
previous endpoint archive's handoff binding. Prior archives are unchanged.

All 48 intermediate checkpoint evaluations, functional measurements/inference,
crossed continuation and final reconstruction/publication remain required.
No manuscript result was installed; scientific completion and source release
remain false. No remote write or protected-helper interaction occurred.

The scripts require this original host's authenticated source/data/score files.
For a new readback, use the archived script with `CUDA_VISIBLE_DEVICES=''`, the
original primary `src` on `PYTHONPATH`, and a new `--output` file. This archive is
not a standalone second-host experiment or completed repository release.
