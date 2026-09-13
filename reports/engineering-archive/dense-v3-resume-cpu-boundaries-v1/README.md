# Resume divergence: CPU loading and sampler boundaries narrowed

2026-09-13. Actual CPU diagnostic session **25560**, terminal **470a2d**,
exits **zero**. No GPU training or endpoint comparison was repeated. The two
previous GPU endpoint mismatches remain failures; this is not a tolerance waiver.

## New evidence

1. The declared `FactorialBatchSampler`, installed Accelerate sharding,
   `DataLoaderShard` and `skip_first_batches` produce exactly the uninterrupted
   suffix after skipping 313 × 4 local microbatches. All three declared order
   seeds and four logical ranks pass. Each suffix contains 312 microbatches /
   2,484 groups per rank, including the final four batches of five. Four further
   seed-314159 checks with eight persistent workers/prefetch four agree with
   the zero-worker index sequence. Wrong-seed and off-by-one controls differ.
2. Both genuinely restored step-313 saves (AdamW-source/AdamW and
   AdamW-source/Muon, order 314159) are authenticated before loading. The actual
   installed SentenceTransformer constructor reproduces all 134 saved FP32
   model tensors exactly on CPU. After explicitly zeroing the target model,
   the **original installed `BaseTrainer._load_from_checkpoint`** restores
   those tensors bit-for-bit again. No reimplemented model loader is substituted.
3. The original `FactorialOptimizer` and `restore_training_state` reconstruct
   every named optimizer-state entry and complete scheduler state exactly on
   CPU from both actual checkpoint payloads. No optimizer step is performed.

`sampler.json`, `loaded-a.json`, `loaded-b.json` and `completed.json` retain
the observed results, runtime and relevant module hashes. `check.py` is the
exact executed diagnostic. The two trusted external component digests and
original anonymous-download receipt are the same as the
[portable checkpoint reader](../dense-v3-portable-factorial-checkpoint-v1/README.md).
No checkpoint was downloaded again, replaced or uploaded.

## What this does not establish

The sampler test uses indices `0..49999`, not raw training text, collated
tokens or gradients. Logical rank sharding executes sequentially on CPU, not
under a four-process NCCL group. Worker comparisons disable CPU pinning and
do not exercise CUDA transfer. The model-loading test uses genuine full-size
models on CPU and the original loading function, not instrumentation of the
earlier failed GPU attempts. No original rank-level first-update trace exists
in this evidence. The diagnostic did not install saved RNG into its process.

These checks rule out an index-skip error or CPU load/state conversion error
**in the tested paths**. They do not rule out GPU load/device placement,
collated-input differences, gradient arithmetic or communication-order effects.
They do not identify NCCL or FlashAttention as the cause. Full cause localization,
exact GPU-resume acceptance and source release are still false.

The next discriminating boundary is a separately recorded four-rank comparison
of loaded device tensors, exact first-step token inputs and pre/post-reduction
gradients. It must precede any further long endpoint attempt. Retain original
kernels, source identities, comparison requirements, failed outputs and both
GPU lease namespaces; never touch the protected helper.

`tests/test_factorial_resume_order.py` adds reusable index-order and negative
regressions to the repo. They are engineering checks, not new scientific runs
or paper findings. No implementation-error narrative is added to the manuscript.
All 14 regression cases pass: session 43292, terminal 15ebd9, actual exit zero.
Ruff check/format pass (af8c19). No product numerical source was edited.
