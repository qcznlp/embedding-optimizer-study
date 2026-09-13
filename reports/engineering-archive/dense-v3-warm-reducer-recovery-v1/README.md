# Exact endpoint recovery after explicit reducer warmup

2026-09-13. **Both genuine four-rank resumes now reproduce their original
endpoints bit-for-bit.** The previously failed recovery attempts remain unchanged.
This is engineering recovery evidence, not another scientific run or paper result.

| Case, fixed AdamW-source/order 314159 | Coordinator | Ranks | Fresh comparison reader |
| --- | --- | --- | --- |
| AdamW continuation | 58312 / 20aa07 / exit 0 | Four exits 0 | Exit 0 |
| Muon continuation | 65312 / 02e59e / exit 0 | Four exits 0 | Exit 0 |

Both restore the same genuine anonymously downloaded step-313 save used in the
earlier failed checks, retain the original 70-file numerical closure, and execute
the remaining **78 updates / 9,936 queries** to step 391. Scientific checkpoints,
full-corpus results, statistical rules and manuscript text are unchanged.

## What changed

The prior [paired boundary](../dense-v3-paired-backward-boundary-v1/README.md)
found identical local leaf gradients but different cold/rebuilt-reducer gradients.
The new separately bound outer entry retains the scalar Adam-counter device
placement adapter and adds exactly one restoration pass at first native clipping:

1. Let the original resumed Trainer perform its first four backward microbatches,
   without clipping or updating. Observe actual token features and local gradients.
2. Reuse the same four batches and first-input RNG in the same native training
   step/no-sync schedule through the existing DDP object. No original loss,
   differentiation, communication, clipping or optimizer kernel is replaced.
3. Require unchanged model/optimizer/scheduler states, identical first-input and
   post-pass RNG, all **536 local contributions/rank** equal to the recorded pair,
   and all **134 post-DDP tensors/rank** equal to its authenticated warm reference.
4. Only after every rank passes, call the original clipping implementation and
   continue the unchanged native loop. Subsequent clipping calls delegate directly.

All eight actual gates pass before any optimizer update. The cold pass's extra
backward work does not advance the optimizer, scheduler, Trainer step, input stream
or RNG stream. The reference comes from the earlier no-update diagnostic, not
from replacing gradients or reading the desired final endpoint into the model.
No alternative communication hook, kernel setting or error tolerance is used.

## Exact endpoint comparison

A new CPU process performs the original native checkpoint-content admission and
calls the unchanged original recursive bitwise comparator. Both cases pass:

- all **134 model tensors / 149,014,272 elements**;
- complete optimizer state: AdamW **402 tensor + 934 scalar leaves**;
  Muon **226 tensor + 937 scalar leaves**;
- all ten scheduler scalars;
- each of four rank RNG payloads: five tensors, one NumPy array and 631 scalars;
- original selected Trainer counters: global step, maximum steps, epochs, batch
  size and final epoch value.

Actual reader bindings are AdamW
`e4ce02a361adb79c39a1dd05d0868392470d72bd61ea5b495a7ced9f48eec4bf`
and Muon `a53216fa398592e65469e0d11e06e8f94feaea92cacbe564a3c44c220962290f`.
Whole serialized checkpoint files need not share metadata/path bytes; equality
above is complete named model/optimizer/scheduler/RNG values and the declared
Trainer counters, not logging elapsed times or the differently sized first log window.

This demonstrates a sufficient restoration procedure for the **two tested cases**
on this host. It does not prove a uniquely identified NCCL kernel cause, all-save
or NorMuon coverage, physical second-host equivalence, or an already packaged
general restoration entry. Those limits do not undo the observed exact equality.
The earlier device-only endpoint failures are preserved, not relabelled or erased.

## Controls and remaining source delivery

The original trace and actual installed native-normalization controls are retained.
Initial 25 controls pass (54712 / 46539d / exit 0); the final expanded set has
**29 passes** (84248 / 4fff68 / exit 0), including refusal of changed tokens,
local/post-DDP gradients, RNG, missing parameters/contributions and invalid
reference conditions. CPU preparation is 34364 / d6efc3 / exit 0.
Three document controls pass (12637 / c87c46 / exit 0); whitespace check passes.

The completion guide now distinguishes verified current v3 evidence from
historical controller instructions. It preserves the old checklists and does
not waive release gates. No implementation-error narrative enters the paper.

A separate read-only single-checkout publication check fails at the historical
`config.py` binding (20618 / ff7df0 / exit 1). The subsequent complete source-set
census identifies exactly **config.py and optimizers.py** in both original
publication closures (56 / 60 files). They now contain the verified primary
training implementations, whereas the older analytical contract binds previous
versions. The already completed original multi-source native/numerical replay
remains valid. This needs an explicit versioned consumer/source transition, not
replacement of old hashes, reversal of the validated training integration or
another scientific rerun. The factorial's separate old input-module binding
boundary also remains. No full repository release is claimed.

## Artifacts and authority

Executed helpers/entry, tests, authority, small raw receipts/logs and source census
are archived here. Large diagnostic saved-state binaries stay in their original
output directories; the complete bound component JSONs are retained under `actual/`.
These diagnostic endpoints are not uploaded over the original scientific backups.
`manifest.json` binds all archived files except itself.

Original work directory: `/tmp/dense-v3-warm-reducer-recovery.rP4NVV4i`.
Entry SHA `3edaa4ea6ed39006bcf178cf611bee46811d0319e7d2608857eb250df7242fbb`;
authority SHA `49de01d10a5bca26436db205dd0159df49813daf73abe68dd03ee23f52022b08`.
Both original lease namespaces remained acquired/inherited; only owned children
were supervised. No protected-helper access, process enumeration, stopped-controller
mutation, remote write or deletion occurred. All jobs in this report are terminal.
