# Fresh factorial worker lifecycle

Completed locally on 2026-09-10. This is a bounded implementation milestone,
not a formal branch, GPU pilot, scientific result or publication release.
Primary training, all sixty checkpoint backups and the complete weight-analysis
backup were already finished. Existing GPU evaluation queues were unchanged.

## What now connects

The new internal [factorial_v3_worker.py](../../../src/embed_optim/factorial_v3_worker.py)
connects the unchanged factory to the actual inherited 391-step Trainer call,
all-five-stage/final-package finalization and original deep whole-run consumer.
Both genuine calibration chains must enter through the original factory; a
hand-entered learning rate is not a worker request. All 69 accepted parent
source bindings stay unchanged; the new worker is separately content-bound.

`run_branch` is called by all four ranks of an **already admitted, exclusively
owned** NCCL worker. It does not launch processes, initialize or destroy a process
group, take a lease, provide the outer authority, or replace any live dispatcher.
The caller still needs the separately reviewed source/runtime/main-completion/
resource handoff and real default-topology GPU verification. No such handoff
was performed here, and a successful component record never claims one.

The lifecycle refuses an existing attempt or output, requires fresh optimizer,
scheduler, gradients and Trainer state, and calls `train(resume_from_checkpoint=None)`.
After all four ranks report the full horizon, the existing bound Trainer saves
the five stages and final package. Rank zero then re-reads both original
calibrations, the complete data identity, every native checkpoint/model/moment
and the exact final package before writing the separate worker-complete record.
This same-worker read is not an independent fresh-process proof or an observed
OS exit. Failures retain a rank/phase record and all partial payloads, with no
implicit retry or overwrite. Exception text and environment are not serialized.

The separate CPU-capable `read_execution` re-runs the original native whole-run
consumer, checks all four rank observations, creation/configuration provenance,
file inventory and timestamps. It does not execute training or grant scientific,
resource or source-release acceptance. The prior native seals and run identities
are not rewritten. Fresh-only orchestration does not remove the independently
tested lower-level bound resume capability; an admitted outer resume workflow
and genuine GPU save/reload check remain future work.

## Executed checks and failures

The first **53-case** suite completed with **50 passes / 3 failures**. These were
real omissions in the new, not-yet-executed worker: loaded-context metadata was
not rechecked, and the fresh reader did not compare the saved factory recipe and
requested arguments. The final shared creation-record check covers both paths.
The first source, test, original log and XML are preserved in `source-first/`
and `actual/`. An unused-import lint report preceded the first pytest call;
[attempts.json](attempts.json) records its exact scope without claiming the
post-lint preserved source is the original pre-lint file.

The final combined suite passes **194 / 194**, zero failures/errors/skips,
with exactly 194 serialized testcase records: **57 new worker cases**, plus
30 factory and 107 run-component regression cases. Both actual test handles
are terminal: first **33576 / exit 1**, final **28447 / exit 0**. Formatting and
lint pass for the two new authored Python files; no whole-repository test or
publication claim follows from this focused suite.

The new cases include real 70-file assembled-source verification and a real
CPU refusal before factory/model construction, writes or CUDA initialization.
All positive worker-lifecycle and deep-consumer wiring cases use **explicitly
mocked** GPU context, collectives, factory/Trainer execution, saves and native
reading. Their synthetic native placeholder is rejected when the actual native
consumer is restored. They are not four-GPU runs, calibrated rates, saved genuine
models, independent training seeds or 12 completed branches. Earlier real CPU
model-loading/Trainer-numerical checks were not repeated as missing experiments.

The final assembled copy is in [actual/source-final/](actual/source-final/).
Its 96 inherited physical files (69 bound parent dependencies plus retained
assembly/support files) are preserved; the separately added worker makes 97.
The current new worker/test are also retained under `source-final/`. This is
local source preservation, not GitHub publication or an admitted release.
[verification.json](verification.json) binds original/final evidence and the
unchanged source scopes; [plan.md](plan.md) preserves the bounded original task.

## Current jobs and remaining work

At the separately timestamped **12:58 UTC** observations, primary BEIR has
**24 / 840** accepted task cells, baseline **14 / 14**, and eight exact live/R
primary workers. Validation is live/S at **7 / 12**, with no failure/selection/
completion record; functional encoding is live/S at **0 / 61**, waiting for
validation. See [observations.json](observations.json). These are actual handle
observations, not lock-only liveness claims or a utilization measurement.

The one-GPU priority request remains unanswered, not approved. There was no
resource handoff, competing job, frozen-source change, old-controller transition,
protected-helper access, Git/HF write or manuscript modification. No retrieval
winner or functional-dimension mechanism is claimed from incomplete outcomes.

Next finish the existing full primary evaluation, validation and functional
campaigns. The factorial still requires genuine GPU calibration, actual
four-GPU Trainer/save/reload admission, the authorized worker/resource transition,
all twelve real 50K branches and their complete probe/168-task outcomes.
Complete scientific consumption, cross-host reconstruction, the NAACL paper
and the authorized reproducible source release remain required. Keep the full
goal active; this component does not substitute for those scientific results.
