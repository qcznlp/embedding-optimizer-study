# Current DenseOn experiment and handoff

Last observed **2026-09-10 12:58 UTC**. This is a dated execution snapshot, not a
live heartbeat or a scientific conclusion. Exact separately timestamped process
observations are in the [latest snapshot](reports/engineering-archive/dense-v3-factorial-worker-v1/observations.json);
the [earlier snapshot](reports/engineering-archive/dense-v3-weight-artifact-backup-v1/observations.json) is unchanged.
Read [AGENTS.md](AGENTS.md) before acting; its safety and authority limits still apply.

## Goal

Deliver a defensible **NAACL paper and reproducible repository/model-analysis
artifacts** about how AdamW, Muon and NorMuon change DenseOn weight space and
retrieval. DenseOn is the only active architecture. No new LateOn work or blog.

The evidence chain is: complete retrieval comparison → reached weight states and
trajectories → functional dimension utility → held-out-dose prediction, with a
separately bounded state-by-reset-operator continuation. Spectral fingerprints
alone are not an embedding-specific explanation; prediction is not mediation.

## Current evidence

| Component | Verified state | What remains |
| --- | --- | --- |
| Corrected primary training | **12 / 12 complete**, no training run queued | Do not retrain these completed recipes |
| Primary checkpoint durability | **60 / 60**, 1,200 files, 89,889,820,336 logical bytes | Final source/analysis release remains separate |
| Zero-update baseline | **14 / 14 full-corpus tasks complete** | Reference only, not an optimizer comparison |
| Primary BEIR | **24 / 840 task cells**, eight exact workers live | All 12 rates × five stages × 14 tasks; no partial-task mean |
| Validation-only LR selection | **7 / 12**, coordinator live/S | All twelve before the fixed selector runs |
| Weight geometry and update map | Both geometry branches **60 / 60**; all-state map/readback complete | Link to complete functional/retrieval outcomes |
| Weight-artifact durability | **298 / 298 files**, 4.98 GB, immutable public backup and complete anonymous recovery verified | Not a source release or second-host experiment |
| Functional dimension analysis | **0 / 61** encoded states, coordinator live/S | Waiting for all validation results; vectors, interventions and inference pending |
| Crossed continuation | **0 / 12 formal branches** | GPU calibration/admission, worker handoff, branches and complete outcomes |
| Paper / source release | Draft and local WIP, **not complete** | Verified generated findings, portable reconstruction and release gates |

The training/durability evidence is the content-bound
`launch/observations/all-twelve-training-sixty-checkpoints-evaluation-running-20260910.json`
in the experiment tree, SHA-256
`1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3`.
Ten original worker exits are observed zero; the final two have verified complete
artifacts and observed termination, with OS exit codes **unobserved/null**.
Do not manufacture exit records or weaken the preserved original guards.

There is no complete primary optimizer ranking or established useful-dimension
mechanism yet. Current weight displacements differ in spectral concentration,
but that is not proof of more useful embedding dimensions. Learning-rate cells
are not independent training seeds. Historical scores cannot fill missing v3 cells.

## Which experiment and source are current?

The fixed [v3 primary protocol](configs/dense_primary_v3_protocol.json) has SHA
`4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b`.
All runs use the same revised 500K groups, seed 42, one positive/seven negatives,
no in-batch negatives, temperature 0.02, length 8192 and global batch 128.
AdamW rates are 1e-6 / 3e-6 / 1e-5 / 3e-5; Muon and NorMuon rates are
1e-4 / 3e-4 / 1e-3 / 3e-3. Steps are 782 / 1563 / 2345 / 3126 / 3907.

These are fresh full-horizon `verified-v3-*` runs from the same pretrained base,
not continuations of the older scientifically held campaign. The owner's
training-priority exception allowed the frozen, content-bound execution while
deferring code publication. It did not erase draft/release gates or authorize
unreviewed controller, source, HF-deletion or GitHub transitions.

Host-specific paths below are locations, not commands to execute on another host:

| Role | Current location |
| --- | --- |
| Development / paper / handoff | `/root/embedding-optimizer-story-refactor` |
| Frozen 56-file primary assembly | `/root/embedding-optimizer-primary-v3` |
| Actual data / runs / evaluations / active launch records | `/root/embedding-optimizer-v3-experiment` |
| Historical experiment and retained source data | `/root/embedding-optimizer-study` — not the current trainer |
| Primary local models | Experiment `outputs/dense-correctness-v3/dense/verified-v3-*/checkpoint-*` |
| Primary HF models | The **60 exact entries** in [the download index](docs/primary-v3-checkpoints.json) |

The independent primary assembly SHA is
`e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8`.
The current interpreter on this host is `/usr/bin/python`, not the old `.venv`.
The [formal runtime](configs/formal_runtime.json) and imported source identities
must match; a development installation does not certify a formal worker.

## Safe current observations

These **read existing observers**; they do not create jobs, take leases or resume
anything. They are host-specific. If a handle is missing, inspect original
receipts before proposing any action; an observation timeout is not termination.

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /root/embedding-optimizer-v3-experiment/launch/task-observer/status.py

CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /root/embedding-optimizer-v3-experiment/launch/functional-dimensions/observe.py
```

The baseline and weight-geometry jobs are terminal; do not poll their stale
live-only handles or rerun them. Existing primary evaluation dispatchers own the
current eight workers. Validation has not obtained another token; functional
encoding waits for its complete selection. The one-GPU priority question is
**unanswered, not approved**. Do not re-ask it, infer approval from an automatic
continuation, stop a lease-holding parent or force-unlock a slot.

## Next actions, in scope

1. Finish the existing full 840-cell evaluation matrix and all twelve validation
   results without changing corpora, scoring, context or frozen workers.
2. Complete all 61 functional states and the original interventions/inference;
   join the full outcomes to the already completed weight measurements.
3. Complete the separately gated crossed continuation: two fixed source states
   at step 2345 × two reset operators × three order seeds, all twelve 50K runs,
   five stages each and 168 final full-corpus retrieval units. The current
   [fresh-worker component](reports/engineering-archive/dense-v3-factorial-worker-v1/README.md)
   connects the unchanged factory, full 391-step Trainer, five-stage finalization
   and deep native run reader. Its 57 new / 194 combined checks are bounded CPU
   source/refusal tests and explicitly mocked orchestration, not GPU calibration,
   default-topology GPU admission or a formal run. The 69 parent files remain
   unchanged. No admitted launcher, resource handoff or implicit resume is supplied.
4. Reconstruct the complete scientific outputs from their original artifacts,
   generate the paper's results and figures, then pass the full publication,
   portability, clean-source and authorized release requirements. Do not hand-edit
   generated findings, remove pending markers or replace failed historical hashes.

Keep all rates, stages, tasks, original features and frozen inference rules. A
null or inconclusive result remains reportable; do not select a preferred story.
No implementation/debugging narrative belongs anywhere in the manuscript.

## Recovery and handoff

Use [checkpoint-restoration.md](docs/checkpoint-restoration.md), the authenticated
60-entry index and the standalone recovery script for a new machine. They avoid
the moving HF default branch and the historical checkpoint namespace. File
verification does not deserialize models or prove whole-run/bitwise resumption.
Complete analysis also needs the exact code, data/probes and unfinished outcomes.

The [completed recovery evidence](reports/engineering-archive/dense-v3-recovery-entry-v1/README.md)
includes all-60 anonymous remote metadata verification and one real 20-file,
1.35-GB checkpoint download with independent offline file/seal verification.
It does not claim a second-host experiment or complete source publication.

The complete native weight measurements also have a separately verified
[public snapshot and restoration guide](docs/weight-analysis-restoration.md):
all original spectra, bases, tables, map arrays and figures, at dataset revision
`209b4517e64ac5373db47b04ada39104e9080016`. All 298 files were actually recovered
anonymously and independently verified offline. The
[original backup evidence](reports/engineering-archive/dense-v3-weight-artifact-backup-v1/README.md)
preserves the scope, original metadata and each actual execution record.

The current local WIP has **not** been published as a complete GitHub release.
Do not retry the recorded GitHub 403 through another identity. Historical HF
withdrawal was rejected: do not retry, split or bypass it. Never touch `gpu.py`
or its processes; no broad process/GPU-process inspection. Preserve the exact
old stopped controller chain and every original/failed artifact. Detailed
identities, source scopes and remaining authority are in [AGENTS.md](AGENTS.md).

The former default recovery guide and README are preserved in the
[before-copy](reports/engineering-archive/dense-v3-recovery-entry-v1/before/).
They are historical evidence, not instructions to execute the active experiment.
