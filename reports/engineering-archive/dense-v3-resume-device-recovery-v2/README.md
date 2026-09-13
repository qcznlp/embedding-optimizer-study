# Actual GPU recovery with value-preserving Adam counter placement

The original GPU recovery attempts exposed a restore-path defect: CPU-loaded
Adam scalar step tensors stayed on CPU while the original fused CUDA kernel
required them on each parameter's device. All eight original failed ranks and
both original exit-one coordinators remain preserved in the
[final-evaluation closeout](../dense-v3-final-evaluation-closeout-v1/README.md).

## Scoped repair

New `src/embed_optim/optimizer_state_devices.py` validates all named Adam
counters, shapes/dtypes, their externally authenticated checkpoint step and
existing moment placement before moving counters. It preserves exact FP32
counter bits and does not alter moment objects, group fields or kernels.
The adapter is explicit; it is not yet integrated into general optimizer
loading or claimed as a complete source release.

The new source-bound verification entry uses the **original** bound Trainer,
checkpoint/optimizer/scheduler loaders, complete scientific inputs and numerical
kernels. A declared `TrainerCallback.on_train_begin` invokes the adapter after
loading checkpoint 313 and before any update. This is a restoration extension,
not a hidden modification of frozen original source. Each rank records its
adapter binding, moved counter identities, exact bytes and unchanged component.

The original anonymous HF downloads and native readbacks are reused and retain
their original source/authority. No duplicate model download, false new-download
receipt or fresh optimizer initialization occurs. Outputs are new; both original
GPU lease namespaces and four-rank supervision remain enforced. W&B is disabled.
The exact model/optimizer/scheduler/all-rank-RNG endpoint comparison is unchanged.

## Completed CPU checks and actual GPU launch

- Adapter controls: **29 passed**, actual session 57440 / terminal **961c0f**.
- Recovery/callback controls: **37 passed**, session 19309 / terminal **61da0c**.
  This includes the original 29 operational/comparison controls and eight new
  callback/identity/download-provenance controls, not extra scientific trials.
- Actual source-bound preparation: **1f599c / exit zero**, original admission
  reused, no duplicate model reads/downloads claimed.
- Original-versus-new source diff is retained; `diff` exit one indicates
  expected source differences, not a failed test. No original file was edited.

Live directory: `/tmp/dense-v3-resume-device-recovery.uQ0ynb0k`.

| Binding | SHA-256 |
| --- | --- |
| New coordinator/worker source | `b2089cd2fae33b55a018a4471a427196143012b6a98303d23a703c8e1dc435d8` |
| Device-placement adapter | `255dcfb58baea6f6271325ee5b0fd8299eafb94a2dce06baa244ddf1cdb2452d` |
| New authorization | `1e7fbcbbde7021ece3a18e5608f8cb909991860911097d0bb54e7b7a7043989a` |
| Reused original download receipt | `3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332` |

Actual coordinator sessions **39663 / 75985** are live. At **07:12:37 UTC**,
the exact observer **bdac2e / exit zero** confirms both coordinators live/S,
all eight direct registered ranks live/R, both dual-lease admissions and no
failure. Coordinators: **987741/start325530641**, **987742/start325530642**.
No endpoint reader or comparison has run at this observation.

Read only the current registered handles with the unchanged copied observer:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' /usr/bin/python -B \
  /tmp/dense-v3-resume-device-recovery.uQ0ynb0k/observe.py \
  --source-sha b2089cd2fae33b55a018a4471a427196143012b6a98303d23a703c8e1dc435d8 \
  --authorization-sha 1e7fbcbbde7021ece3a18e5608f8cb909991860911097d0bb54e7b7a7043989a \
  --output /tmp/resume-device-observation-NEW.json
```

**Actual GPU placement is now verified**, not just planned: metadata read
**0f678d / exit zero** authenticates all eight rank receipts. At 07:13:34–37,
each AdamW rank moves 134 counters and each Muon rank moves 46 auxiliary Adam
counters from CPU to its own CUDA rank, retaining every counter's exact bits.
The native rank-zero logs (**865c79**) show both continuations have passed the
former first-step failure and reached **at least step 318/391**. These log
progress bars do not establish endpoint equality or measured speedup.

Keep this live entry/source/authority/helper unchanged. Do not restart,
duplicate, switch kernels, modify original checkpoints or relax comparison.
Require all four rank exits plus the independent original native CPU endpoint
reader in each case. A failed successor remains a failed attempt.
This covers two genuine 78-update recoveries, not every saved checkpoint,
NorMuon, fresh independent source seeds or a physical second host.

At **07:17:43 UTC**, original observer **eec176 / exit zero** confirms both
coordinators live/S, all eight exact ranks live/R or S, no failed receipt and
no endpoint reader yet. Bounded original rank-zero progress reads **48f452**
show **347/391 for both runs**. Do not infer equality or completion from those
progress values; retain the original final comparison requirement.
