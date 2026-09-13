# Actual four-rank factorial Trainer integration

This continues the preserved September 7 component preparation after all twelve
corrected primary runs completed. It uses CPU only and does not change or delay
the live evaluation, validation or functional queues. It is not a formal DenseOn
run, scientific admission, release, or checkpoint replacement.

Assemble the exact 56-file primary source snapshot plus the four existing
factorial components and the original diagnostic worker into a fresh directory.
The worker retains the complete 50,000-row / 391-update horizon and four real
Gloo ranks, with the actual inherited ST training step and explicit InfoNCE.
First inspect the prepared loader; then exercise training, all five saves and
externally bound resume. Compare exact consumed row identities, parameters,
named moments and scheduler state against uninterrupted execution. Include both
reset optimizers and preserve every failed attempt before making any repair.

`integration.py` authenticates and mechanically assembles sources, records each
actual subprocess launch and exit, and refuses source changes or existing output
namespaces. CPU-only diagnostics do not admit the default 134-tensor DenseOn
topology, calibration, genuine branch data or a complete scientific run.

Execution root: `/tmp/dense-v3-factorial-trainer-integration.dmkUEso6`.
The complete original execution directory is mechanically preserved in `actual/`;
original absolute path fields are retained, not silently rewritten for relocation.

## Actual bounded acceptance — September 10, 2026

All thirteen diagnostic launcher processes are terminal: eleven successful
four-rank attempts and two preserved first failures. No diagnostic is a formal
training run. The original twelve-run primary matrix remains complete and separate.

- All three declared seeds pass the real prepared-loader/readback: exactly
  50,000 distinct groups, 12,500 per rank, the direct seeded permutation and all
  1,564 local micro-batches. The last update contains all 80 remaining groups.
- Routed AdamW and Muon both complete actual 391-update ST/InfoNCE training, with
  native component saves at 79 / 157 / 235 / 313 / 391. Every rank has bitwise
  equal model, named optimizer and scheduler endpoint tensors/states.
- A separate AdamW execution stops after update 79, and a separate Muon execution
  stops after update 313. Each fresh process resumes its externally bound complete
  checkpoint. The remaining row sequence is exact; both final model/optimizer/
  scheduler states are **bitwise equal** to their uninterrupted references.
  This fixture retains dropout 0.1 and clipping 1.0.
- A separately named CPU fixture disables dropout and clipping to expose actual
  unscaled gradients. On updates 1 and 391, every parameter on every rank agrees
  with an independent FP64 `sum / global_query_count` oracle: global counts 128
  and 80, absolute/relative tolerances 5e-6 / 5e-4. Maximum absolute errors are
  3.2230320084e-6 (routed AdamW) and 4.3795242606e-6 (Muon). Quarter/fourfold-gradient
  controls are rejected. These comparisons ran inside the actual original workers;
  raw gradient tensors were not persisted, and the collector does not claim a
  subsequent fresh gradient replay. Sources, inputs, seeds and complete executions
  remain available to reproduce that diagnostic.
- All 18 new placement/no-admission-bypass cases pass with the corrected primary
  factory; the existing 82 optimizer tests pass in the development tree. These
  are focused regressions, not a new whole-repository green claim.

## Failures and narrowly scoped repair

`loader-first` fails before training: the original toy module names do not match
the real hidden-matrix routing rule. Only the toy names change in `source-second`.
The original DenseOn topology, partition rule and original worker remain preserved.

`train-adamw-first` then fails before its first update: Accelerate performs an
empty state placement round trip during optimizer wrapping, but the factorial
component treats any unarmed `load_state_dict` as checkpoint loading. The two
positive placement tests fail in `placement-first.xml`; all sixteen negative
controls already pass. The repair in `source-third` permits only an **identical,
empty, step-zero no-op**. It changes no parameter object, rate, routing, setting
or counter. Nonempty/unbound checkpoint loading, altered empty payloads and
rewinding an already-stepped optimizer still fail. No optimizer arithmetic,
primary source, upstream package, frozen protocol or live dispatcher changes.

The preserved source sequence is `source-first` -> `source-second` (toy naming)
-> `source-third` (strict empty placement); `source-gradient` changes only the
diagnostic worker. Every assembly contains the same original 56 primary files.
Original failed logs, exits, test XML and source manifests are retained.

## Remaining work and boundary

Read `complete.json` for the bounded acceptance and immutable archive identities.
Actual default-topology DenseOn/four-GPU verification, genuine source-state/data/
calibration binding, whole-run reading and the twelve formal branches remain
required. The CPU toy cannot clear those gates. All primary evaluation/validation/
functional queues remain unchanged; the pending one-GPU priority question is not
approval. No old controller transition, GPU-helper access, GitHub write, HF erasure,
manuscript change or scientific/publication admission occurred.
