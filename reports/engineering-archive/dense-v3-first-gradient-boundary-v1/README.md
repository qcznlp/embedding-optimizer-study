# Actual four-rank first resumed backward boundary

2026-09-13. Both GPU diagnostics finished at their explicitly bounded stopping
point. **No optimizer update was executed.** This is a new diagnostic, not a
rerun of the completed scientific matrix or a repaired full resume.

| Execution | Observed terminal result |
| --- | --- |
| Trace controls | 10 pass; session 26660 / ccdcac / exit 0 |
| CPU preparation | session 76884 / bf1572 / exit 0 |
| AdamW diagnostic, four ranks | session 50164 / 84aa35 / exit 0; all rank exits 0 |
| Muon diagnostic, four ranks | session 69733 / 4a2137 / exit 0; all rank exits 0 |
| Independent CPU reduction comparison | session 86535 / bc1ffd / exit 0 |

Both cases use their genuine anonymously restored step-313 checkpoints from
AdamW-source continuations under order seed 314159. They retain the original
70-file numerical execution closure and all native run/data/calibration/model
admission. The previously tested scalar-counter placement adapter runs before
the new state check. No old source or checkpoint identity was changed.

## Observed boundaries

Every rank verifies all 134 model tensors and the complete optimizer/scheduler
state against the authenticated save **after actual CUDA loading**. CPU copies
of device tensors agree bit-for-bit. It then compares the next four actual
loss inputs with independently collated expected rank-local dataset indices:
all tensor shapes, dtypes and bytes match. These are genuine text/token inputs,
not the index-only fixtures from the earlier CPU check.

Public leaf-gradient hooks copy, but never replace or modify, the contribution
from each backward call. Each rank observes **134 × 4 = 536** contributions.
The original inherited `training_step`, loss, backward and DDP implementation
perform all four microbatches. No communication hook is installed.

An explicitly declared diagnostic trap at the original clipping call captures
all 134 **post-DDP/pre-clip** parameter gradients and intentionally stops before
clipping. Trainer and optimizer remain at step 313. No finalizer, training
completion receipt or new training checkpoint is produced. All actual post-DDP
gradient tensors agree bit-for-bit across the four ranks within each case.

The trap is a diagnostic stopping mechanism, not an alternative clipping kernel
or a claim that an entire training step was executed unchanged. Synchronizing
CPU copies can affect timing. CPU sums of the observed leaf contributions are
labelled explicitly; they are not direct reads of DDP's internal bucket buffers.

## Independent check of gradient scaling

After both jobs exit, an independent CPU reader authenticates all gradient
files and compares the observed post-DDP gradient with the FP64 mean across
four ranks of their saved CPU leaf-contribution sums. It covers all
**149,014,272 elements / 134 tensors** in each case.

| Case | Reference L2 norm | Observed L2 norm | Relative L2 difference | Fitted scale |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 2.2670751129268742 | 2.267075112687868 | 2.3197369829269274e-8 | 0.9999999998945746 |
| Muon | 2.416728794220124 | 2.4167287935657913 | 2.4740885748993806e-8 | 0.9999999997292478 |

Maximum absolute difference is 1.1175870895385742e-8 in both cases. These
are descriptive measurements against independently ordered arithmetic, **not
a new acceptance tolerance**. They show no extra accumulation/world-size
scale factor at the tested boundary. They do not establish exact equality
with the original uninterrupted step-314 gradient.

DDP's recorded logging fields at this boundary include `has_rebuilt_buckets=0`
and `bucket_sizes="596057088"` in every rank. These are observations about the
new process, not evidence of the original uninterrupted reducer state or proof
that bucket rebuilding caused the earlier endpoint discrepancy. Other logging
fields (including the recorded `num_buckets_reduced=0`) are retained verbatim;
they are not used as a completed-collective counter.

## Implication and next discriminating test

The tested GPU load path, expected actual token inputs, inter-rank gradient
agreement and global scaling now have direct evidence. The prior two long
endpoint comparisons remain failures. We still lack a paired same-weight/
same-input comparison across fresh and already-used DDP reducer states.

A bounded next test can compare cold versus reused-reducer backward passes
without any optimizer update, preserving weights, inputs and RNG, and compare
the observed **local leaf contributions** separately from **post-DDP gradients**.
Only matching local contributions with differing post-DDP results would locate
that difference downstream of local differentiation in the tested setting.
Even that would need separate validation before claiming the old endpoint
failure repaired. Do not blame NCCL/FlashAttention, relax exact endpoint checks,
change kernels or repeat a long endpoint attempt based only on the present data.

## Artifacts and scope

Executed source and raw small receipts/logs are preserved here. Large safetensor
gradient arrays remain in the original diagnostic directory, with exact paths,
byte counts and SHA-256 in `gradient-bindings.json`; they are not Git/wheel
payloads or an HF upload. `manifest.json` binds the archived files except itself.
The original directory is `/tmp/dense-v3-first-gradient.FDK2kcqs`.

Both original dual GPU lease namespaces were acquired and inherited. Only new
direct child processes were supervised; their actual exits are retained. No
GPU-process enumeration, protected-helper access, historical-controller change,
external write, checkpoint deletion or scientific-result change occurred.
All jobs in this report are terminal and must not be polled or restarted as
missing work. The paper receives no implementation-error narrative.
