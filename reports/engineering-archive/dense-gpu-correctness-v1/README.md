# GPU correctness verification — engineering only

Completed GPU campaign: **2026-09-06 06:29 UTC**. The primary matrix remains scientifically on
hold. This report is excluded from the manuscript; no new retrieval or optimizer-quality result
is claimed. All diagnostic launchers have exited and the original BEIR chain remains stopped.

## Numerical results

| Actual four-GPU path | Raw/clipped gradient checks | Interpretation |
| --- | --- | --- |
| Unchanged Trainer, short inputs | All 9 raw checks fail at approximately 0.25; only 6 clipped checks pass | Double normalization is confirmed on the real mixed-precision path |
| Isolated candidate, short inputs | 9/9 raw and clipped pass; maximum raw error 1.1921e-7 | Candidate normalization agrees with independent manual backward on this fixture |
| Candidate, 8192-token default backward | 0/9 raw/clipped pairs pass the unchanged tolerance; maximum raw error 0.069019 | Failed numerical-reproducibility check, not accepted or suppressed |
| Candidate, paired deterministic backward | 9/9 raw/clipped pairs pass; maximum raw error 4.7684e-7 | Changing only owned attention backward flags resolves the tested discrepancy |

Each step checks **all 134 trainable tensors** (including 88 hidden matrices); each optimizer
executes groups of 128/128/32 distinct queries. Raw gradients and final weights are identical
across the four ranks within each run. Short-case profiler events include actual BF16
FlashAttention forward/backward CUDA kernels; FP32 parameters and TF32 settings match production.
All 18 files of the independently downloaded source checkpoint remain unchanged.

For the long-input pair, only the first 32 fixed synthetic rows are expanded, in every one of the
nine text columns, before unchanged 8192-token truncation. The complete generated-data signature
is identical between conditions. All four ranks reach 8192; both conditions finish without OOM.
Peak allocated memory is 50,988,549,632 bytes, including the audit's **second reference model and
gradient comparison allocations**; do not quote it as primary-training peak memory or a speed
comparison. The paired control changes only `deterministic_flash_attn` on the 22 owned attention
instances and their config, for both actual and reference copies. Three installed dispatch/kernel
sources are SHA-256 bound. Neither checkpoint/config file nor production/library source changes.
The control supports a kernel-reproducibility explanation on this fixture. It does not make the
default-path failure disappear or establish numerical harmlessness in long training.

### GPU checkpoint continuation

The full-model candidate executes one three-step synthetic epoch, saves actual Trainer checkpoints
at steps 1/2/3, then fresh Trainers resume from 1 and 2. All three optimizers use the declared
**eight data-loader workers per rank**, persistent workers and prefetch factor four. Native GPU
deserialization is used, with no CPU transport adapter. All **six continuations / 24 rank
comparisons** exactly restore entry model/optimizer/scheduler, actual consumed row order, and
rank-local CUDA/CPU/Python/NumPy RNG. This does not measure the worker processes' own RNG states;
their deterministic preprocessing produces the exact observed rows.

No continuation is strictly bitwise identical after subsequent calculation. Maximum final weight
element discrepancies across all ranks/restart points are:

- AdamW: 2.9802e-8;
- Muon: 6.1899e-5;
- NorMuon: 3.4392e-5.

At restart 1, all three start with only 1.1921e-7 maximum raw-gradient discrepancy. After a
Muon-family update, the next gradient discrepancy grows (maximum 0.0039079 for Muon and
0.0098956 for NorMuon). These measurements narrow the next audit target to update/replay
amplification; they do not isolate its cause, prove a serialization bug, or establish harmlessness.
The existing gradient tolerances are reported as additional measurements, **not newly introduced
optimizer-state equivalence bounds**. This is same-process-group continuation with fresh Trainers
and fresh diagnostic optimizers at real trained weights—not independent-process/host restart or
resumption of a historical primary optimizer state.

### Retained first-attempt failure and tests

The first old-path attempt completed its nine numerical comparisons and saved each step, but
failed while reading a legacy TF32 property for its final receipt. Its exact executed source,
step records and rank logs remain in `initial-source/` and `initial-live/`. Only the diagnostic
metadata getters changed; the second complete old-path attempt reproduced all nine quarter-scale
failures. The pinned PyTorch release documents the separate new precision interface and the
unsupported API mixing. [PyTorch 2.9 CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices).

The full regression run passes **1,319 tests, zero failures/errors/skips** in 155.896 seconds;
the raw JUnit receipt is `pytest-final.xml`. The new diagnostic/test files pass lint
and formatting, and both checkouts pass whitespace checks. Tests/archival checks do not override
failed numerical comparisons. Production numerical source is unchanged. Only unbound top-level
handoff notes are added to live `AGENTS.md` and `PROJECT_STATUS.md`; their original bytes are
retained in `before-live/`. The original controller's entire source contract is rehashed.

## Evidence and continuation

Raw main receipts are `short-live.json`, `short-candidate.json`, `gpu-resume.json`,
`max-context-default.json`, `max-context-deterministic.json`, and `deterministic-control.json`.
The latter binds the actual deterministic-control result and flag-only intervention. Rank logs,
executed source copies, original handoffs and the unchanged earlier audit receipts are retained.
The artifact validator preserves the failed acceptance flags and checks actual source bytes,
all prior probe bindings, 124 archived handoff files and eight interrupted worker logs:

```bash
python -B scripts/validate_dense_gpu_correctness_artifacts.py \
  --repository /root/embedding-optimizer-story-refactor \
  --output /tmp/new-dense-gpu-validation.json
```

Use a new output path. The GPU diagnostic entrypoints are host-scoped and require the exact paused
handoff; they are not portable launch commands for another host. Their executed command components
are recorded in `commands.json`, with the full control arguments also in its receipt. All use the authenticated
temporary checkpoint download described in `docs/checkpoint-restoration.md`, never live checkpoints.
The synthetic save/resume checkpoints remain under `/tmp/dense-gpu-resume.8t7UPz/` for targeted
follow-up; they are not paper artifacts or replacements for the original primary checkpoints.

Next: isolate replay/update amplification, complete the reference-optimizer boundary (the earlier
NorMuon epsilon finding is unchanged), and prepare a reviewed correction/replication plan. Do not
start formal retraining or scientific promotion from these engineering results. No source push,
runtime numerical deployment or HF deletion occurred. A WIP-source publication exception remains
separate from the existing no-pending-paper pre-commit gate.

## Original resource handoff and audit design

The owner's September 6 instruction prioritizes correctness and scientific validity. The exact
project-only BEIR handoff completed at 05:56:33 UTC; see `handoff/result.json` and `before.json`.
Three dispatch processes remain stopped in place and retain the original lease. Eight verified
leaf evaluators exited via pidfd-bound signals, after checking their entrypoints, creation times,
parents, empty child sets and default/unblocked SIGTERM disposition. No group signals or broad
process inspection were used. The archive preserves 124 source/result/state files byte-for-byte.
All checkpoints, production sources and live ledger bytes remain unchanged. No `gpu.py` utility
or process was inspected or touched. The handoff helper was formatted after execution; its six
bound production sources, exact process handles and actual signal events are retained in the raw
receipt. Do not mistake later formatting for a new live operation.

The first GPU audit uses a verified independent download of the real NorMuon 3e-4 stage-4 model,
FP32 parameters, BF16 model-local autocast, FlashAttention-2 and actual four-rank NCCL Trainer
execution. It evaluates fresh diagnostic AdamW, Muon and NorMuon optimizers on 288 distinct short
synthetic queries each: global groups 128, 128 and 32. It checks exact token identities, raw and
clipped gradients, actual scheduler cadence and cross-rank gradient/final-weight equality.

The manual reference bypasses Trainer and Accelerator backward. It differentiates the same
microbatch mean loss divided by the actual number of microbatches, then explicitly averages across
four ranks. Model-local autocast and FP32 output conversion match production. This isolates the
normalization path; it is **not** a new independent objective-formula oracle. The separate earlier
CPU loss audit supplies that narrower check. Prior absolute/relative gradient tolerances are kept
unchanged. Failed comparisons remain failures; multiplying old-path gradients by four is labelled
a separate diagnostic, never an acceptance pass of the old path.

The initial short-input design alone does not validate maximum length, production data-loader
workers or checkpoint replay. The explicit extensions and their incomplete/failed boundaries are
reported above. None certifies long-horizon training or optimizer quality. No formal run is launched
and no live numerical fix is deployed.

## Resumption boundary

The original three dispatchers remain alive and stopped with their creation-time-bound handles
in `handoff/before.json`. Do not run another finalizer, release their lease, use the old zero-step
migration, edit source under them or blindly signal numeric PIDs. Before any continuation, confirm
that diagnostics have ended and revalidate these exact handles and unchanged source/ledger bytes.
The original scheduler will register interrupted workers as failed; it may then finish remaining
jobs and return a nonzero attempt status, after which the parent can retry missing results. Its
in-memory progress preserves completed-step accounting; a new controller's old resume path does
not. In-flight worker logs must also be archived before any retry can overwrite them. Scientific
promotion and a formal replication decision remain separate gates. None of this belongs in the
manuscript.
