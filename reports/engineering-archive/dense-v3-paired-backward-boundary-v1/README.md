# Paired first and repeated GPU backward boundary

2026-09-13. Two genuine four-rank step-313 restores each run two backward passes
on the same four microbatches per rank. **No optimizer update, clipping, training
checkpoint or scientific completion is produced.**

The original 70-file numerical closure and native Trainer/loss/DDP implementations
are unchanged. The second pass explicitly calls the same native `training_step`
with three `no_sync` microbatches and one synchronized microbatch. It restores
first-input Python, NumPy, CPU Torch and rank-local CUDA RNG; actual token tensor
bytes and unchanged model/optimizer/scheduler states are checked. No communication
hook or alternative numerical kernel is installed.

## Actual result

All **134 parameters × four local leaf contributions = 536 per rank** agree
bit-for-bit across the two passes, in all eight ranks. Independently saved CPU
sums also agree. Within each pass, post-DDP gradients agree across all four ranks.
Between passes, **123/134 post-DDP tensors differ in both cases**.

| AdamW-source continuation | Changed elements / 149,014,272 | Relative L2 difference | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| AdamW, order 314159 | 2,829,460 | 2.0473483980114333e-8 | 1.4901161193847656e-8 |
| Muon, order 314159 | 3,456,445 | 2.1862897063898537e-8 | 1.4901161193847656e-8 |

DDP logs change from `has_rebuilt_buckets=0` to `1`; the repeated pass records
16 rebuilt buckets with the same sizes and parameter grouping in both cases.
These observations locate this pair's difference downstream of the observed
local derivatives. They do not uniquely identify a backend kernel, directly
observe the original uninterrupted step-314 gradient, or prove the historical
long endpoint mismatch fully explained. CPU leaf sums are not internal bucket
buffer reads. Hook synchronization can affect timing. Numerical magnitudes
above are descriptive measurements, not a new acceptance tolerance.

The next test is a separately bound restoration adapter: reuse the same first
four microbatches before any clipping/update, restore their RNG, and require
the repeated post-DDP gradients to match this authenticated warm reference.
Only after that gate may a new bounded endpoint check test actual recovery.
The old exact model/moment/scheduler/RNG/counter comparison must remain unchanged.
No recovery result is substituted into the scientific matrix or manuscript.

## Execution history, including failed predecessor

| Entry | Actual terminal evidence |
| --- | --- |
| Initial helper tests | 14 pass; 73922 / 2e57d9 / exit 0 |
| Initial preparation | 95598 / 1cf94c / exit 0 |
| Initial AdamW pair | 17406 / 663091 / exit 1; rank exits [1, 1, -15, 1] |
| Initial Muon pair | 39489 / 33677f / exit 1; rank exits [1, -15, 1, 1] |
| Corrected helper/native normalization tests | 17 pass; 11434 / 0de2c0 / exit 0 |
| Corrected preparation | 30104 / f2d912 / exit 0 |
| Corrected AdamW pair | 32954 / c06ef0 / exit 0; four rank exits 0 |
| Corrected Muon pair | 85232 / 6ee966 / exit 0; four rank exits 0 |
| Independent CPU comparison | Both pool files and final summary complete; launch tool/session and OS exit unobserved after output loss |

The initial diagnostic stopped before second backward because its new guard
incorrectly required `model_accepts_loss_kwargs=false`. Original Trainer also
divides by accumulation when the actual item count is `None`. The corrected
diagnostic calls native `get_batch_samples` and explicitly verifies that count.
Tests exercise installed original `Trainer.training_step` under both flag values.
This is a diagnostic guard correction, not a discovered training-loss error.
The failed source/receipts remain unchanged in `failed-first/`; no failure is
reclassified as a success. The original CPU comparison is not rerun to invent
an observed exit.

## Preservation and authority

`executed/` and `actual/` retain the corrected sources and small raw receipts.
`gradient-bindings.json` lists the large arrays retained at their original paths;
they are not copied into Git/wheel or uploaded. The final `manifest.json` binds
this archive except itself. Terminal tool results are distinguished from artifact
completion. The failed predecessor is `/tmp/dense-v3-paired-backward.lNjcKc0t`;
the corrected entry is `/tmp/dense-v3-paired-backward-fixed.JXOJBril`.

Both original GPU lease namespaces were acquired and inherited by owned children.
No protected helper access, process enumeration, historical-controller mutation,
external write, scientific checkpoint replacement or manuscript change occurred.
All GPU jobs recorded here are terminal; never restart them as missing work.
