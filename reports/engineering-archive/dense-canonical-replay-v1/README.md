# Fixed-layout reduction: gradient and checkpoint acceptance control

Completed **2026-09-06 08:32 UTC**. Engineering-only, isolated source. No production or scientific release.

## Result and interpretation

The identified short-input replay discrepancy disappears under the explicit fixed-layout reduction
control with deterministic attention backward. The same controlled implementation also satisfies
the existing independent gradient oracle, including maximum-length inputs. A separately launched
process group reproduces the sealed baseline exactly without training another baseline.

| Check | Observed result |
| --- | --- |
| Short-input global-average gradients, three optimizers × three updates | 9/9 raw and clipped checks pass, all 134 parameter tensors |
| Short-input maximum raw / clipped absolute error | 5.9604645e-8 / 1.4901161e-8 |
| 8192-token global-average gradients, same three recipes and updates | 9/9 raw and clipped checks pass; every rank reaches 8192 |
| Maximum-length maximum raw / clipped absolute error | 2.3841858e-7 / 7.4505806e-9 |
| Same-process-group checkpoint continuations | 6 continuations, 24/24 strict rank checks bitwise exact |
| Independently launched process-group continuations | 6 continuations, 24/24 complete fingerprint checks exact |
| Full repository regression | 1,367 JUnit cases; zero failures, errors or skips; 136.369 seconds |

The original gradient tolerances are unchanged: absolute 5e-6, relative 5e-4. Exact replay uses
no tolerance. Four rank copies and the two matched restart points are correctness controls, not
independent optimizer trials. AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4 start from one authenticated
trained DenseOn state with fresh diagnostic optimizers. This is not a comparison of their quality.

This closes the bounded replay verification question left by the
[bucket-layout origin probe](../dense-gradient-origin-v1/README.md). It does not retroactively
turn any earlier default-path failure into a pass. Ordinary bucket-dependent floating-point
variation was not itself proof of an optimizer or serialization bug. The distinct production
double-normalization defect remains unfixed there; all 12 primary runs remain on scientific hold.

## What is controlled

`dense_canonical_reduction_control.py` attaches a hook only to each owned diagnostic DDP instance.
It requires all 134 trainable parameters to be FP32 on one device, the same topology on four ranks,
and no pre-existing communication hook or unused-parameter search.

The reducer waits for all bucket gradients, concatenates them in lexical parameter-name order,
predivides exactly once by four, and issues one NCCL SUM. It copies the result back to the original
bucket views and resolves their CUDA-aware futures. It rejects missing/repeated parameters and
incomplete reductions. The Future callback preserves the CUDA stream dependencies and storage
lifetime; neither an installed library nor the production reducer is edited. See the
[PyTorch 2.9 communication-hook contract](https://docs.pytorch.org/docs/2.9/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.register_comm_hook)
and [Future synchronization contract](https://docs.pytorch.org/docs/2.9/futures.html#torch.futures.Future).

The underlying DDP bucket partition still changes from one initial bucket to 16 rebuilt buckets.
All four ranks record this, while the controlled collective vector remains identical in layout.
Every completed backward communicates exactly once. No gradient is quantized by this control.

The already verified `SingleNormalizationTrainerCandidate` is unchanged: Trainer retains the
actual accumulation counts 4/4/1, while Accelerator's second divisor is one. Only the owned model
config and its 22 attention instances change their backward determinism flags from false to true.
Both actual and manual-reference model copies receive that same attention setting. All production
optimizer formulas, parameters, BF16 forward/TF32 policy, losses and clipping rules remain unchanged.

This one-vector control holds communication until backward is ready and allocates another full
gradient vector. **It is a correctness intervention, not a performance recommendation or an
automatically selected formal-training default.** The maximum-length audit peaks at
51,584,606,720 allocated bytes, including two full-model oracle copies and instrumentation;
that is not a formal-run memory estimate or optimizer speed comparison.

## Checkpoint and independent-process coverage

The same-process campaign runs the actual Trainer, callbacks, scheduler and native GPU
deserialization. Each recipe executes three updates on a fixed 288-query synthetic epoch, global
groups 128/128/32, and saves steps 1/2/3. The first warmup update has learning rate zero; the two
later updates are active. Fresh Trainer instances continue from both saved restart points.
All paths exercise eight persistent data-loader workers per rank, with prefetch factor four.

Before ending this campaign, each rank exports exact baseline fingerprints for both restart
points. They retain every field of the original strict replay comparison: all subsequent entry
weights and optimizer/scheduler states, raw/clipped gradient tensors and hashes, row identities,
losses, local CUDA/CPU/Python/NumPy RNG, final weights/state, epoch and global step. Tensor hashes
include dtype and shape; container and key types are preserved. Only already-consumed steps are
filtered, exactly as in the original comparator. Two additional guard tests explicitly check that
later entry states/raw gradients cannot be silently omitted and empty replays are rejected.

The independent command begins **after the baseline launcher exited zero**, with a new torchrun
identity. It verifies the trusted baseline receipt SHA-256, every bound helper source, the complete
checkpoint file inventories and all rank fingerprint files before loading state. It runs only the
six continuations, not another baseline, and compares the complete fingerprints exactly.
The final audit reopens and compares these fingerprints independently of the runtime pass flags.
All diagnostic source checkpoints and the independently downloaded primary anchor are unchanged.

The two process groups are `810f6b70-82b0-4e1d-9d1d-168f66b981a0` and
`efdf98cf-651d-4c0c-bb91-4c14f72f4678`. Both use the same physical host/GPU mapping and pinned stack.
This is **not** another-host restart, long-horizon reproduction, dropout stress, natural-text
maximum-length quality, or continuation from every historical optimizer state.

## Evidence and preservation

`commands.json` contains all four GPU commands, their observed zero launcher exit codes and the
full regression command. `normalization/`, `same-process/`, `max-context/` and `fresh-process/`
preserve original receipts, all rank observations and all 32 stdout/stderr logs byte-for-byte.
`observed-source/` preserves the executed wrappers, reducers, helpers and 23 new guard cases.
`before/` and `before-live/` preserve the preceding handoff pages. No earlier failed receipt,
threshold, protocol or primary checkpoint is replaced.

The 153 baseline checkpoint files total **13,482,133,524 bytes** and remain under
`/tmp/dense-gpu-resume.15V3UA/`. The 36 baseline/replay fingerprint files total **77,997,932 bytes**
and remain under `/tmp/dense-canonical-replay.s4tJbC/` and
`/tmp/dense-gpu-fresh-replay.PWPDCN/`; additional continuation checkpoints are retained too.
These diagnostic payloads have **not** been uploaded to HF or committed to Git. The final validator
binds their exact paths/digests and verifies the checkpoint bytes and complete fingerprint equality.
Preserve the named temporary trees until an explicit archival/retention decision.

Read-only verification, with a new output path:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
python3 -B -m scripts.validate_dense_canonical_replay \
  --repository /root/embedding-optimizer-story-refactor \
  --output /tmp/new-canonical-replay-validation.json
```

This audit fails closed when assertions are disabled. It preserves all 49 bindings of previous
validation `8ebba68f9623660c3251a8596e3c72fc64d57cb67eaba6de3ce3bc44746a76ac`, checks the unchanged
six live numerical sources in both checkouts, and rechecks the exact stopped BEIR handles/ledger.
No protected helper is inspected or touched.

One handoff-only progress refresh was initially run from the isolated checkout, where the
matrix's relative output paths have no primary artifacts; it consequently reported zero runs.
`progress-wrong-checkout.json` preserves that observation. The authoritative refresh runs the
unchanged artifact reader from the live experiment checkout, writes the explicit isolated
snapshot path and recovers 12/12 and 60/60. No training artifact or numerical implementation was
changed. The README now states this working-directory requirement explicitly.

## Next decision

Prepare the reviewed normalization and optimizer-reference correction, plus a state-preserving
runtime and replication transition. The NorMuon small-matrix epsilon nonconformance still needs
resolution; its earlier 352/352 real-model update matches do not establish universal conformance.
No formal retraining, evaluation resume, source WIP publication, manuscript release or HF deletion
is implied by this diagnostic milestone. None of this engineering narrative belongs in the paper.
