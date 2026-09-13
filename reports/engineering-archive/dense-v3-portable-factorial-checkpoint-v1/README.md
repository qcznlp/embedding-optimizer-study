# Authenticated portable saved-factorial checkpoint reader

2026-09-13. Real artifact-readback integration, not a new scientific experiment.
The new `embed_optim.saved_factorial_checkpoint` entry accepts explicit local
checkpoint/runtime paths and an externally trusted component-receipt SHA-256.
See [the user guide](../../../docs/saved-factorial-checkpoints.md).

## What changed, and what did not

The two original validation bodies are retained in a separate artifact reader;
AST tests prove their only changes are the explicit runtime read location and
its argument plumbing. The original `factorial_v3_run_contract.py` and
`factorial_v3_checkpoint.py` files remain byte-identical. All historical source
identities and the original fresh-run source rejection remain unchanged.

External receipt authentication and complete payload hashing precede tensor/
pickle-backed decoding. Ambiguous JSON, changed payloads, missing rank state,
incorrect runtime bytes and changed saved identities fail closed. The reader
also verifies the actual runtime before decoding and rechecks runtime bytes
afterward. Trusted external hashes do not make arbitrary pickle files safe.

No numerical kernel, model, optimizer moment, statistical result or manuscript
was changed. The new entry is for existing saved artifacts, not full-run
scientific admission, calibration recomputation or GPU execution.

## Actual outcomes, including the failed first attempt

| Check | Observed outcome |
| --- | --- |
| Initial focused reader tests | 17 pass; session 30075, terminal f6993a, actual exit 0 |
| First genuine package attempt | Build/audit exit 0; case A exit 1 before decoding because the driver copied only the runtime JSON, omitting its required relative reconstruction files; session 77872, terminal 3c1d1d, outer exit 1; case B not launched |
| Corrected genuine package attempt | Full wheel runtime assets used, with actual reader-runtime verification added; build/audit/cases A and B all exit 0; session 97051, terminal 641a57, outer exit 0 |
| Expanded focused tests before CLI additions | `unit-final.xml` records 43 tests, zero errors/failures/skips; session 69365 terminal output was lost and its handle no longer exists, so its OS/tool exit is **unknown**, not inferred from XML |
| Final source + two new CLI controls | 45 tests, zero errors/failures/skips; style/build/original full audit all actual exit 0; session 64992, terminal 2c059c, outer exit 0 |

The missing-runtime failure and original driver/source are retained, not
reclassified. The corrected test supplies the required runtime reconstruction
assets instead of relaxing their checks. The final package reader is
byte-identical to the successfully tested corrected wheel. CLI argument
routing and required-digest refusal are unit-tested; the genuine executions
use the packaged Python API, not an installed console-script invocation.

## Genuine checkpoint coverage

Both checkpoints were previously downloaded anonymously in the native resume
verification. This turn reuses those exact files and original download receipt;
it does not claim another upload or download. The copied receipt SHA-256 is
`3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332`.

| Case | Source / operator / order | Step | Trusted component SHA-256 |
| --- | --- | ---: | --- |
| A | AdamW state / AdamW / 314159 | 313 | `00e0dab0a4aaf52a5783ffb6ae2d5399952d55f2055b65b033fb5c4a0baa400d` |
| B | AdamW state / Muon / 314159 | 313 | `dda70c52285f3eb1025b990bd00a46dedf89e54c9a9e65d8fce29aa9f1aa7795` |

Both native readbacks exactly equal their earlier accepted native results.
Each checks 134 model tensors, complete named optimizer state and scheduler,
training arguments and all four saved rank RNG states. All **22 loaded study
modules** are wheel-local. Each execution actively refuses the four forbidden
project roots plus the saved runtime root (the latter duplicates one root),
then records zero refused operations during the actual checkpoint read.
Network/process-spawn guards record no attempts and CUDA is not initialized.

These are Python audit-hook controls, not an OS sandbox, and the experiment is
same-host relocation, not a physical second-host test. Only two step-313 saves
were decoded here; this is not coverage of all sixty saves, every source/seed,
or step-391 final-state acceptance. Synthetic byte envelopes used in refusal
tests are explicitly non-model/non-pickle fixtures, never scientific evidence.

## Retained files and remaining work

`before/` preserves the pre-change documents/config and original validators;
`reader-first.py` and `verify-first.py` retain the failed attempt's versions.
`actual/`, `actual-corrected/` and `actual-final/` retain separate observed
outcomes. `after/` and the three distribution directories freeze the tested
sources and final guide. `manifest.json` binds every archived file except itself.

The existing 24 scientific runs, 120 checkpoints and complete numerical/paper
replay remain completed. Full scientific-consumer/current-source integration,
the explicit fresh-factorial source transition and reviewed source publication
remain open. The separate GPU-resume endpoint mismatch remains unresolved;
these successful CPU reads neither repair nor waive it. No GPU work, protected
helper interaction, external write, checkpoint deletion or source publication
was performed in this task.
