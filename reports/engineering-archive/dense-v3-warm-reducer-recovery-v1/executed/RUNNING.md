# Bound warm-reducer recovery check — 2026-09-13

Source `probe.py`: 3edaa4ea6ed39006bcf178cf611bee46811d0319e7d2608857eb250df7242fbb.
Authority: 49de01d10a5bca26436db205dd0159df49813daf73abe68dd03ee23f52022b08.
Do not edit these live sources/helpers or duplicate these attempts.

- Initial 25 helper tests: session 54712 / 46539d / exit 0; `tests.xml` retained.
- Expanded 29 helper/native-normalization/exact-gate controls:
  session 84248 / 4fff68 / exit 0; `tests-final.xml`.
- CPU preparation: session 34364 / d6efc3 / exit 0.
- AdamW coordinator: session **58312**, launch ec9a67; terminal exit not yet observed.
- Muon coordinator: session **65312**, launch dbf33f; terminal exit not yet observed.

## Terminal closeout — authoritative later state

Both coordinators are terminal, **exit zero**:
AdamW 58312 / **20aa07**, Muon 65312 / **02e59e**.
All eight rank exits and both fresh comparison-reader exits are zero.
Readback tool **68ec9b / exit 0** confirms both exact endpoint comparisons:
all 134 model tensors, complete optimizer state (402 / 226 tensor leaves),
scheduler, all four saved rank RNGs and selected Trainer counters agree bitwise.
Both cases perform exactly 78 updates / 9,936 queries from step 313 to 391.
No tolerance was introduced. Do not poll or restart these completed handles.
This verifies the two declared cases on this host, not every checkpoint,
NorMuon, physical second-host resume, a packaged general adapter or new science.

Additional delivery checks: three focused document controls pass, session
12637 / c87c46 / exit 0; diff whitespace check b6f964 / exit 0.
The source-context check 20618 / ff7df0 exits 1 at the original publication
binding for config.py. The subsequent read-only census d5de13 / exit 0 finds
exactly config.py and optimizers.py differ in both old publication source sets
(56 / 60 files). Historical protocols were not modified. These are source-version
integration requirements, not a failed numerical replay or GPU restore.

Both use genuine restored step-313 saves and the original 70-file numerical
closure. After the first four native backward microbatches, an explicit
restoration pass repeats them with the same weights/input/RNG. Before any
clipping/update, all 134 post-DDP gradients must match the authenticated warm
reference, all 536 local leaf contributions must match, and post-pass RNG must
equal the original first pass. Unchanged model/optimizer/scheduler states are
checked separately. Only then may the original remaining 78 updates proceed.
Final endpoint comparison retains original exact recursive bitwise equality
for model, optimizer, scheduler, all four rank RNGs and selected Trainer counters.
These are diagnostic outputs, not new scientific cells or substitute results.

Both original lease namespaces and exact protected-controller handoff remain
mandatory. No protected-helper access, process enumeration or historical-controller
mutation. Exact owned identities are recorded in `run/pool-*/started.json`.
Use the existing sessions; do not infer termination from elapsed time or files.

Actual first-gradient gate readback: tool e496dd / exit 0. All eight rank receipts
report exact 134 post-DDP tensors, 536 local contributions, equal post-pass RNG,
unchanged model/optimizer/scheduler and zero optimizer updates before the gate.
Both native loops subsequently pass step 318. Final endpoint comparison is pending.
