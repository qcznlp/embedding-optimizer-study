# First checkpoint-replay gradient difference: bucket-layout control

Completed 2026-09-06 07:39 UTC. **Engineering diagnosis only. No scientific or runtime release.**

## Result

The first-replay gradient discrepancy is localized on the tested short-input fixture. All
per-microbatch local gradients and their local accumulated sums are bitwise equal between the
uninterrupted and resumed branches. The actual DDP reducer uses **16 buckets** in uninterrupted
training versus **one bucket** in the first resumed update. Replaying those exact observed
partitions on the captured local sums reproduces both actual post-reduction gradients bitwise.
Holding the local gradients fixed and switching only the partition/order exactly reproduces the
resumed branch's gradient difference.

The result holds both with default FA2 backward and with only the owned attention backward
determinism flags enabled. In both variants, the actual reduced gradients also match the
authenticated prior capture made without these tensor-observation hooks. This connects the new
diagnosis to the [preceding update-propagation probe](../dense-gpu-replay-localization-v1/README.md),
not just to a different instrumented run.

| Check | Default backward | Deterministic backward |
| --- | ---: | ---: |
| Exact local microbatch tensor pairs: 134 parameters × 4 microbatches × 4 ranks | 2,144/2,144 | 2,144/2,144 |
| Exact reconstruction using each actual bucket layout: 2 layouts × 4 ranks | 8/8 | 8/8 |
| Fixed-gradient layout switch reproduces actual resumed gradient | 4/4 | 4/4 |
| Identical-gradient, identical-layout repeat | 4/4 | 4/4 |
| Uninterrupted / first-resumed bucket counts | 16 / 1 | 16 / 1 |
| Parameter tensors with a reduced-gradient difference | 123/134 | 123/134 |
| Maximum absolute reduced-gradient difference | 1.1920929e-7 | 1.1920929e-7 |
| Exact agreement with prior uninstrumented reduced gradients: two branches | 268/268 tensors | 268/268 tensors |

The local-sum payloads are also byte-identical between backward modes for each rank. The two
variants and the four rank copies are matched correctness controls, not independent optimizer
trials or statistical evidence. Only Muon 3e-4 is run here because the previous probe established
the shared first-update input for Muon/NorMuon; this does not add an optimizer-quality comparison.

The combined engineering explanation is now: identical restored state and local backward
gradients → different DDP bucket partition/order → small FP32 reduced-gradient difference → a
few changed BF16 rounding outcomes → propagation through Newton–Schulz and the actual update.
The previous probe already reproduced the optimizer-plus-scheduler update exactly from each
captured gradient. **Do not describe this as a newly discovered serialization/optimizer defect,
numerical divergence, harmlessness proof, or explanation of better retrieval.** Floating-point
summation order is relevant to reproducibility, not by itself evidence of incorrect mathematics.

The separate duplicate-normalization defect in unchanged production training is still unresolved
there, and remains the decisive scientific hold. This diagnosis does not validate that matrix.
Both parent campaigns still intentionally exit 1: **0/16 strict full-replay rank checks pass**
across two modes × two restart points × four ranks. Exact entry state, row/RNG replay, scheduler
and final progress are retained. No failed threshold is relaxed or erased.

## Method and scope

`audit_dense_gpu_gradient_origin.py` wraps the existing full DenseOn candidate GPU resume audit.
It explicitly selects one recipe, uses the same digest-authenticated trained model weights and
fresh diagnostic optimizer, and preserves NCCL/FP32 parameters/BF16 model forwards/TF32, zero
dropout, non-reentrant checkpointing, eight loader workers per rank, and the 288-row synthetic
128/128/32 epoch. Baseline and resumes run within one process group. **An independent process or
host restart and historical optimizer continuation are not tested here.**

Leaf-tensor hooks copy the four relevant local microbatch gradients and return `None`; no gradient
or communication function is replaced. Tensor hooks execute before the leaf accumulation/node
post-hook phase in the documented autograd order. See the
[PyTorch 2.9 hook-order documentation](https://docs.pytorch.org/docs/2.9/notes/autograd.html#backward-hooks-execution).

After actual backward has finished, the script queries zero-valued replicas of the reducer's
current buckets to capture parameter IDs, order, sizes and dtype. It rejects communication hooks,
missing/duplicate parameters or mismatched ordering. PyTorch's pinned DDP implementation starts
with one bucket when unused-parameter detection is disabled, then rebuilds using observed gradient
readiness. The installed source is copied under `observed-source/torch-ddp.py`; the upstream code
is [PyTorch v2.9.1 distributed.py](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/nn/parallel/distributed.py).

Independent replay sums the four local microbatch gradients in order, concatenates them in the
observed bucket layout, predivides FP32 values by four, then applies NCCL SUM and restores parameter
views. This follows the default reducer's predivision/SUM arithmetic, checked against
[the pinned reducer source](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/csrc/distributed/c10d/reducer.cpp),
and must reproduce every actual reduced tensor exactly. The crossed control changes the bucket
layout only; it does not feed a reconstructed gradient into training.

The deterministic variant changes only the config and 22 attention instances in each of the three
owned model loads. No installed library file, primary checkpoint, production code, protocol or
tolerance changes. Observation allocations can affect timing, so no speed claim is made; exact
agreement with the prior uninstrumented gradient capture is a separate measured guard.
This short-input result does not erase the earlier, distinct default-kernel 8192-token failure.

## Evidence and commands

- `default/` and `deterministic/` contain byte-identical original receipts and all four ranks' logs.
- `observed-source/` stores the executed diagnostic, helpers, tests and installed DDP source.
  Its adjacent `PYTORCH-LICENSE` preserves the installed distribution's copyright/license notice.
- `commands.json` records both full GPU invocations and the full regression environment.
- `pytest-full.xml`: **1,344 JUnit cases**, zero failures, errors or skips, in 133.528 seconds.
  Seven new guard tests were added; the full suite used absolute audited source paths.
- `before/` preserves the previous handoff files; all 64 bindings of validation
  `829f29f1cc132d15fcbcfebf70c2b6c1fcf23bdb01dd6dda87d1ca95117574ed` remain verified.
- Eight local-sum files total **9,537,539,384 bytes**, stored under
  `/tmp/dense-gradient-origin.VIPdDw/` and `/tmp/dense-gradient-origin.DlSnUG/`, with exact digests
  in the rank records. They have **not** been uploaded to HF or committed to Git. Keep them for
  further reconstruction; the JSON receipt alone cannot replace these tensor payloads.
- Newly generated diagnostic checkpoints remain under `/tmp/dense-gpu-resume.nbzt97/` and
  `/tmp/dense-gpu-resume.oq4plk/`. The prior captured-gradient file is separately verified against
  its trusted receipt before deserialization, never inferred from a filename.

Read-only artifact verification (no GPU allocation, controller lease or training):

```bash
python3 -B -m scripts.validate_dense_gradient_origin \
  --repository /root/embedding-optimizer-story-refactor \
  --output /tmp/new-gradient-origin-validation.json
```

The validator checks all raw assertions, executed sources, prior bindings, external tensor-file
digests, unchanged live numerical source and exact paused dispatcher handles/ledger. It rejects
disabled assertions (`python -O`). GPU reruns require fresh `mktemp` output/log directories and
the exact project handoff guard. Do not bypass that host-scoped ownership check on another host.

## Next action, not an authorization

This particular first-gradient origin is explained; do not repeatedly rerun the same diagnosis.
Test a **diagnostic fixed-order reduction control** together with deterministic backward, then
exercise save/resume in a fresh process. Keep any new communication rule explicitly isolated and
verify its global-average arithmetic before using it as a reproducibility control. A full fix must
also address the separate normalization and official-optimizer conformance boundaries.

Only a reviewed, source-bound numerical transition and newly named formal replication can supply
paper evidence. Do not resume a historical trajectory halfway under different scaling or silently
promote previous outcomes. BEIR's original three dispatchers remain stopped with their original
lease/ledger; the old zero-step migration is still inapplicable. All diagnostic launchers have
exited. The 12 completed primary runs and 60 backed-up checkpoints remain intact. No formal
training/evaluation, production deployment, source push, manuscript change or HF deletion occurred.
