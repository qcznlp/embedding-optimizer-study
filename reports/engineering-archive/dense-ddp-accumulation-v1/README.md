# Four-rank accumulated-gradient diagnosis

Status: **unchanged production path fails; paired CPU diagnostic control passes; no deployment or pause**.
This changes the experiment's correctness assessment, not its scientific story. Do not place this
engineering issue, its tests or its numbers in the manuscript. Prior checkpoint/content evidence
remains valid for preservation; it does not establish conformity of the accumulated gradient.
The [validation receipt](validation.json) binds the failed executions, paired control, source
identities, sampled-log consequence check and current handoff. Its production-gradient verdict is false.

## Evidence

The actual `OptimizerTrainer` imported from live main runs under the pinned installed libraries
on four CPU/Gloo ranks. It uses the production collator, explicit loss, optimizer builder,
length-grouped loader, batch size 8 per rank, accumulation 4, clipping norm 1 and non-reentrant
gradient checkpointing. The fixture is a random two-layer, 32-dimensional ModernBERT with 14
trainable tensors, zero dropout, 256 synthetic queries and eight own candidate documents per query.
CPU FP32/SDPA replaces GPU BF16/FlashAttention; no production model, dataset or checkpoint is loaded.

The diagnostic observes **consumed** batches inside `training_step`, not prefetched collator calls.
Every full optimizer step covers 128 distinct global query IDs; rank gradients must agree exactly.
Before clipping, an independent row-wise float64 cosine/log-sum-exp objective supplies a whole-batch
reference through the same initial weights. Comparing raw gradients prevents clipping from hiding
a scaling error. Tolerances were fixed at absolute 5e-6 / relative 5e-4 before the first execution.

- `attempt-1.json`: the initial actual four-rank execution fails at its first raw-gradient check.
  `before/audit_dense_ddp_accumulation.py` preserves its exact executed source (SHA `64d3cc78...`).
- `attempt-2.json` and `attempt-2-gradient.json`: the same default path fails again with additional
  observations. Both normalization divisors are 4. Actual/reference norm ratio is
  **0.25000001014415**, cosine **0.9999999999999469**, and residual after fitting that scalar is
  below **2.66e-7**. This is a scale defect, not an unexplained rank mismatch.
- `compensated-control.json`: only the diagnostic-owned Accelerator's backward input is multiplied
  by its second divisor. No installed or production source is patched and no tolerance changes.
  All three optimizer fixtures pass two steps each: all 14 raw/clipped gradient tensors, 256
  distinct consumed rows per optimizer and identical final weights across ranks. Largest raw
  element error is **1.20e-6**; largest clipped error is **1.64e-7**. The two-step diagnostic retains
  warmup, so its first update has zero learning rate; this is not a multi-epoch optimizer-equivalence
  or independent optimizer-algorithm proof.
- `upstream-source-comparison.json`: installed Trainer and Accelerator source files exactly match
  [Transformers v5.3.0](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/trainer.py)
  and [Accelerate v1.13.0](https://github.com/huggingface/accelerate/blob/v1.13.0/src/accelerate/accelerator.py).
  In this no-label custom-loss path, Trainer divides by current accumulation while Accelerator
  divides again by its plugin's accumulation value. Both release versions are pinned by the study.
- `logged-clipping-consequence.json`: ten immutable completed-run histories contain 3,910 logged
  norms; **3,169** lie strictly in (0.25, 1). Removing the extra quarter factor crosses the declared
  unit clipping threshold at these same visited states. These are sampled logs, not every update
  and not a counterfactual re-training or a measured retrieval effect.

AdamW/Muon normalization means this cannot be described as simply a learning rate divided by four.
Time-varying clipping factors can change momentum histories; there is no demonstrated basis for
declaring the discrepancy harmless. Nor does this check establish a particular optimizer's advantage
or authorize discarding checkpoints, redefining the protocol or launching fresh primary runs.

## Reproduce without touching a live job

Use a fresh temporary directory for each attempt. Never use a primary training/evaluation path.
From the isolated checkout, expose the **live audited** source before the diagnostic checkout:

```bash
audit_dir=$(mktemp -d /tmp/dense-ddp-audit.XXXXXX)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ACCELERATE_USE_CPU=true \
PYTHONPATH=/root/embedding-optimizer-study/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
-m scripts.audit_dense_ddp_accumulation --repository /root/embedding-optimizer-study \
--workdir "$audit_dir"
```

The unchanged production path is expected to exit nonzero, with its failure in `audit.json`.
To reproduce the **diagnostic intervention**, create another new directory and add
`--diagnostic-trainer-backward-only`. Do not turn that flag into an undeclared production fix.
The successful result belongs only to that compensated CPU fixture, not the running matrix.

`unit-tests-v1.xml` retains the original 16 helper checks; `unit-tests-v2.xml` contains 18 passing
checks, including detection of a scaling error even when clipping hides it. These unit passes do
not overturn the actual four-rank default-path failure. `observed-source/` preserves the instrumented
source used by both the second failure and paired control.

The separate `full-tests.xml` records 1,220 passing unit/regression tests, with no failures, errors
or skips. That suite does not launch the standalone four-rank audit: its green result is **not**
evidence that the production accumulated-gradient failure has been fixed.

## Current execution and authority

At 06:10 UTC, **10/12 executions are finished and 54/60 checkpoints are preserved remotely**.
The two exact NorMuon handles remain running at 1740/1630; artifact steps are 1744/1634 of 3907.
Both now have sealed checkpoint 1563 as well as 782. The earlier independent ten-complete-run
content audit remains 1,010 files / 82,373,100,558 bytes; no new full-content audit is implied here.

The owner has been asked to allow pausing the two remaining jobs and automatic post-processing.
No answer, pause, source publication, controller migration, training fix, re-training or HF deletion
has occurred. Independent checkpoint backup must continue. Existing controllers do not implement
this documentation hold, so do not imply they have been stopped. Earlier post-processing-only and
WIP-publication requests, and the rejected HF withdrawal plan, remain separate unresolved authority.

This diagnostic does not certify full DenseOn, GPU/NCCL, BF16/FlashAttention, full-length inputs,
dropout, incomplete final accumulation groups, data-loader/rank-RNG restoration or scientific outcomes.
Once authorized, resolve the normalization ownership, verify complete and partial accumulation on
the actual GPU path, and decide the required replication/impact assessment before downstream claims.
Do not weaken frozen gates, silently relabel existing runs or import this incident into the paper.

`before/` also preserves the preceding handoff documents and progress snapshot byte-for-byte.
The previous paper-wording validation remains a dated publication-only receipt, not a later training
certification. No live source was changed. `gpu.py` and its processes were neither inspected nor changed.
