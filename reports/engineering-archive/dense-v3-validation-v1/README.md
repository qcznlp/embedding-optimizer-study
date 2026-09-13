# V3 validation scoring and selection admission

The revised primary chain now has a separate validation scorer, raw-score reader and recipe
selector. The [protocol](protocol.json), SHA-256
`96a8aa2fc684d29ec67649c2e637c0cd4c7dcdd0f31f2e354b7d0955a18fb1c3`, is **preparation-only**.
The parent v3 primary protocol, earlier validation/analysis protocols and numerical sources are
unchanged. The new consumer requires complete v3 runs, revised validation data and exact source
bindings; a diagnostic checkpoint cannot become a primary recipe by relabelling its directory.

## What was exercised

- Actual admission checks all **4,096 revised validation rows**, every input file and complete row
  identities, retaining every original field. This is not model scoring of all 4,096 rows.
- A declared 59-row fixture covers **all 52 revised validation groups and all seven sources**.
  It preserves original order/fields; its 531 tokenized texts have maximum length **1,379**.
- On one leased GPU, each of the three existing three-step natural diagnostic models scores those
  rows. All **59 × 8 cosine scores and 59 × 6 metrics per model** are numerically identical to
  the unchanged original scorer helpers. All **134 parameter values per model** remain unchanged.
- All three saved bundles pass independent scalar float64 replay, with maximum observed loss
  difference below `2.05e-7`. The predeclared absolute `2e-5` / relative `2e-6` replay tolerance is
  unchanged. Raw FP32 metrics are preserved, not replaced by float64 replay values.
- A separate fresh CPU invocation authenticates the producer artifacts and reads all three bundles.
  It rejects **12 changed protocols and six changed score bundles**. The latter deliberately have
  refreshed hashes in new, explicitly invalid copies, so row/metric checks, not just digest checks,
  reject them. Originals are unchanged.
- The actual primary selection CLI refuses absent full-horizon v3 runs at its first whole-run
  gate. **No primary recipe or retrieval result is produced.**

[gpu-result.json](gpu-result.json), SHA-256
`51936005e30ddd2519fba66e5d6847abaaee25cb0ad3d6eb0d4e2e6294b88200`, is the actual GPU-parent
receipt, observed at 17:23:33 UTC. Its 20 new bound artifacts total 2,402,464 bytes, plus the receipt.
Preserve `/tmp/dense-v3-validation.Gf6KJ2`; model inputs remain in the preceding natural-preflight
directory and are not copied or updated here. [admission.json](admission.json), SHA-256
`40107d077d154da07e3707b02624d9be4757dcf8f45f6a5caa72e51b3a6752ae`, is the fresh CPU audit
in `/tmp/dense-v3-validation-admission.dCL5F1`. All owned calls have exited.

The scorer retains FP32 model weights, BF16 forward/FP32 scores, evaluation mode, FA2, batch 16,
maximum length 8,192, temperature 0.02, one positive plus seven explicit negatives and no in-batch
negatives. Mean loss weights all rows equally, not all sources equally. Selection requires all
twelve final validations and chooses minimum mean loss, then lower learning rate on an **exact**
tie. Positive margin and BEIR scores are not selection inputs. Scalar replay tolerance is not a
recipe-selection tie tolerance. Pessimistic ranking counts each tied negative above the positive.

## Tests and boundaries

The final focused suite passes **67 cases**; the earlier 65-case invocation is retained separately.
The complete isolated suite passes **1,777 cases**, zero failures/errors/skips. It covers genuine
scoring arithmetic, exact ties versus near ties, global row means, row identity/order/completeness,
all-twelve-runs-before-selection, corruption and draft side-effect refusal. Toy scoring and grid
fixtures are not model findings. The different numerical candidate's 963-case suite retains its
eleven unwaived old source-contract failures; this checkout's tests do not clear them.

This is bounded scoring/reader engineering evidence. It establishes neither full 4,096-row model
validation, all twelve final models, maximum-context scoring nor cross-host replay. The three
59-row diagnostic passes are not independent optimizer-quality trials. The raw diagnostic losses
must not be interpreted as a selected recipe or comparative retrieval result.

The final validator rechecks preceding evidence, actual inputs, all changed-case artifacts and
archived source copies; it also verifies unchanged original numerical cores, manuscript, main
ledger and exact stopped three-handle BEIR chain. No formal run, controller transition, commit,
push, HF write/deletion or protected-helper action occurred. Old 12/12 executed runs and 60/60
preserved checkpoints remain on scientific hold.

## Continue from here

V3 validation scoring, raw-score readback and complete-grid selection are no longer missing
implementation. Next integrate v3 outcome statistics, weight-space/dimension and publication
consumers, plus the separately routed factorial control and reviewed assembled-source/runtime
handoff. Do not repeat completed diagnostics, change old expected hashes or promote old outputs.
Owner WIP/deployment direction and existing publication/access gates remain unresolved.
This is local-only engineering provenance and belongs in **no manuscript section**.
