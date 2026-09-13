# Two actual GPU resumes with explicit restored-counter device placement

Owner authority is the active DenseOn paper/reproducibility goal and the direct
instruction “你有权做一切事情，目标是尽快完成任务”. This is the scoped repair
of the failed recovery checks, not a new scientific experiment or permission
to modify protected processes, checkpoints, original numerical code or evidence.

The original two GPU resumes failed because CPU-loaded scalar Adam counters
were not moved by PyTorch's standard loader: strict named groups omit fused/
capturable flags, while the original custom step uses fused AdamW on CUDA.
All eight failed rank logs, source and receipts remain in the original entry
and the final-evaluation-closeout archive. They are not overwritten or passed.

## Scope of the extension

The original `RunBoundFactorialTrainer`, checkpoint/optimizer/scheduler loader,
all mathematical kernels, source/data identities and endpoint comparator remain
unchanged. An explicitly added `TrainerCallback.on_train_begin` places only
already loaded Adam scalar counters onto their parameters' devices, preserving
their exact FP32 bits. Every named counter, shape/dtype, trusted step 313,
moment placement, complete population and component identity is checked.
Original moment objects/group fields are untouched; no optimizer state is reset.
No kernel or group-schema flag is changed to make the restore work.

Each rank writes a separately bound placement receipt before its first update.
The final CPU reader additionally requires actual movement of all 134 Adam
counters or 46 auxiliary Adam counters, then uses the original exact comparison
of all 134 model tensors, named optimizer state, scheduler and all four RNG
states. The original 391-step horizon remains; only steps 314–391 are executed,
78 updates / 9,936 queries per case. W&B is disabled.

## Inputs, provenance and resources

Reuse the original anonymous immutable HF step-313 downloads and their native
CPU evidence. Do not redownload, relabel those receipts as newly generated or
change their original authorization. The new authority separately binds the
old source, authority, download receipt, cases, complete scientific queues and
failed attempts. Both original lease namespaces and the original exclusive
four-GPU pool ownership/supervision remain required. No automatic retry.

- Source SHA: `b2089cd2fae33b55a018a4471a427196143012b6a98303d23a703c8e1dc435d8`.
- Adapter SHA: `255dcfb58baea6f6271325ee5b0fd8299eafb94a2dce06baa244ddf1cdb2452d`.
- Authority SHA: `1e7fbcbbde7021ece3a18e5608f8cb909991860911097d0bb54e7b7a7043989a`.
- Original download receipt SHA: `3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332`.
- New outputs: `/root/embedding-optimizer-v3-experiment/outputs/dense-v3-factorial-resume-device-recovery-v2`.
- 29 CPU adapter controls pass (961c0f), 37 original/extension operational
  controls pass (61da0c), actual new preparation passes (1f599c).

These tests and preparation are not GPU resume equivalence. Require actual
rank/reader/coordinator exits and the exact endpoint result. Preserve any failed
successor attempt, do not relax a comparison or edit a live entry. This does not
establish every-checkpoint/NorMuon resume coverage or a physical second host.
