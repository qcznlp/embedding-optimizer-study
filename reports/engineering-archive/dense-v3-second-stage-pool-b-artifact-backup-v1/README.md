# Six second-stage pool-B evaluations: data-only HF backup and recovery

This milestone preserves the six already accepted corrected DenseOn step-1563
outcomes in a new immutable public data snapshot. It does not create new model
results or complete the full second-stage learning-rate grid.

## Exact contents and provenance

- Original cohort: AdamW 3e-6 / 1e-5, Muon 3e-4 / 3e-3, NorMuon 1e-3 / 3e-3.
  Membership is authenticated against the original pool-B evaluation queue,
  never selected from task scores.
- **281 files / 983,874 logical bytes**: 108 native files, 168 original worker
  records, two CSV tables, acceptance summary, README and manifest.
- **84 raw task values, six fourteen-task means, 90 CSV rows**.
- Parent: `8555e5849b56862948b0fc0688085708ce3e67cf`.
- New revision: `35ce505d3cf3389f4f1a2f35b46259749bc1d7ed`.
- Manifest: `09ec6667d84637e6ba615abc37396cf876cded1667db242d1803cf7170a5a0ff`.
- The [accepted thirty-state readback](../dense-v3-second-stage-pool-b-evaluations-v1/README.md)
  is unchanged. Its prior twenty-four checkpoint records overlap earlier
  snapshots and are not additional new outcomes.

Use the [recovery guide](../../../docs/second-stage-pool-b-evaluation-restoration.md)
for the exact public prefix, manifest anchor and download/reconstruction commands.

## Executed checks

The new adapter's **34 tests passed**, including exact cohort/task coverage,
nonfinite or changed scores, refusal of LFS/ignored/existing upload modes, and
preservation of old root entries and subtrees. The independent reader's
**18 tests passed**, including fully rehashed semantic-corruption controls.
These are CPU data/transport checks, not training, GPU or statistical experiments.

Local preparation and the first supplied-snapshot numerical replay passed.
The one-shot upload returned zero at **2026-09-11 12:41:35 UTC**; it used only
regular new-file additions under the new prefix. The remote audit authenticates
every uploaded payload and confirms **20 other root entries, all five previous
corrected subtrees, the card and root attributes unchanged**. No deletion,
overwrite, source publication or second upload attempt occurred.

Complete **anonymous** recovery returned zero at **12:43:54 UTC**. A byte-identical
copy of the standalone reader ran with `python -I -B` from `/tmp` and returned
zero at **12:44:33 UTC** using only the recovered snapshot. It reproduced all
84 scores, original worker identities/exits, six exact means and both CSV tables.
This is actual same-host source relocation and off-host-data recovery, not a
physical second-host model replay or a second statistical analysis.

## Evidence layout

- [artifact_manifest.json](artifact_manifest.json): the exact externally anchored
  remote manifest; source-bearing original native bundles are not uploaded.
- [actual/remote-audit.json](actual/remote-audit.json),
  [actual/download-verified.json](actual/download-verified.json), and
  [actual/recovered-reconstruction.json](actual/recovered-reconstruction.json):
  actual immutable-upload, anonymous-download and independent numerical receipts.
- [actual/adapter-tests.xml](actual/adapter-tests.xml) and
  [actual/reader-tests.xml](actual/reader-tests.xml): original test results.
- [source/](source/): the byte-exact new adapter, reader and bounded tests.
  Only `verify_recovered.py` is standalone portable code; the uploader and tests
  bind their original host/workspace and must not be rerun from this archive.
- [commands.json](commands.json): actual command results, including download
  session completion and the informational anonymous-Hub warning.
- [actual/observations.json](actual/observations.json) and
  [actual/native-primary-exits.json](actual/native-primary-exits.json):
  separately timestamped read-only observations of the original primary queue.
- [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md): prior dated handoff.
- [verification.json](verification.json): local archive binding and final checks.

No new model, primary controller, failed functional coordinator or protected
helper was started, restarted or modified. Primary evaluation keeps its original
sources and resource ownership. GitHub publication and historical HF withdrawal
were neither retried nor bypassed.

## Scientific limits

The cohort is partial. Retaining all six recipes here does not justify selecting
an optimizer, comparing full rate grids or filling missing trajectory cells.
No per-query ranking trace, model tensor, raw example text, executable source
body or credential is in the public data snapshot. Original paths, commands and
process IDs are evidence only, not recovery instructions.

Full 840-cell outcomes, real functional features, crossed continuation, verified
paper results and a portable source release remain incomplete. Backup success
is not scientific completion. Engineering provenance stays out of the paper.
