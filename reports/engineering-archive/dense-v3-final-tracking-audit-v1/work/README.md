# Complete current training-log inventory

Actual online observations: **2026-09-12 22:29:25 UTC**. Independent offline
readback: **22:39:06 UTC**, actual terminal `729188`, exit zero.

All **24 scientific runs** (12 primary and 12 crossed continuations) are present
and `finished` in the original personal W&B project. The saved history contains
**5,172 ordered rows**: 4,692 primary and 480 continuation observations.
Every loss, gradient norm and learning-rate value matches its authenticated
native Trainer history exactly: **15,516 scalar comparisons, zero differences**.
Every recorded learning rate also matches the independently derived original
linear schedule. Summary endpoints are 3907 / 391; final logged rows are 3900 /
390. The missing endpoint metric rows are not fabricated.

The unchanged original **per-run** primary configuration/name/group/tag/terminal
check passes all twelve primary runs. Its old whole-entry historical protocol
guard remains unchanged and is not claimed to pass. Genuine model completion is
authenticated upstream evidence, not a mocked completion flag or repeated model
read. The two original primary NorMuon unobserved OS exits remain unknown; W&B
`finished` does not replace process-exit evidence.

## Continuation metadata limitations

All 37 requested Trainer arguments are present online, but **19 custom research
recipe fields are absent** from each continuation W&B configuration, including
the real optimizer routing, model revision, context and temperature. The full
original native recipe, calibrated rates and factory identity are preserved in
each run record and in the bound native-input copies. Do not claim those fields
were originally logged to W&B. The standard Trainer `optim=adamw_torch_fused`
field is a default argument, not an identifier for the bound custom optimizer;
use `native_recipe.optimizer` and the original factory/checkpoint evidence.

Nine continuation configuration learning rates differ from their native values
by **one adjacent binary64 value** (maximum absolute difference
`2.168404344971009e-19`). These inexact serialized argument values are explicitly
retained, not rounded into equality. The actual logged learning-rate sequences
match exactly. No configuration, history, summary, tag or run status was edited.

## Evidence and bounded checking

- `work/actual/readout.json`: complete actual inventory and all 24 record hashes.
- `work/actual/runs/`: whitelisted online configurations, full named histories,
  original native recipes/factories and observed discrepancies.
- `work/actual/input-bindings.json`: **115 native metadata/source bindings**.
- `native-inputs/` and the copy inventory: exact original native evidence copies;
  no model tensor, example corpus, credential store or unrelated process read.
- `work/offline-verification.json`: independent actual readback of all histories,
  source bindings, original schedules and retained metadata differences.
- `work/test_tracking.py`: twelve separate synthetic refusal controls, terminal
  `b4f629`, exit zero. These are not extra scientific runs or findings.
- `work/tool-receipts.json`: actual tool observations, including the unavailable
  all-run API session exit code. The complete readout and the independently
  observed offline exit-zero verification are distinct evidence.

The new auditor initially used the historical configuration namespace and failed
before API queries. `work/tracking-before-native-source.py` and
`work/initial-preparation-failure.json` preserve that first attempt. The successful
reader uses the original primary configuration source plus five unmodified pure
tracking functions. No training, numerical kernel or scientific rule changed.

All API access was read-only and targeted exact run identities, never project
enumeration. This closes current training-log accounting, not complete factorial
inference, final paper/source publication, GPU-resume equivalence or a fresh
model-admission replay. Engineering details stay out of the manuscript.

At **22:39:06 UTC**, continuation full-corpus BEIR is **10/168** with eight exact
registered workers live/R and neither coordinator failed. Earlier `2/168` and
`8/168` observations remain intact. Completed training, checkpoints, primary
outcomes and probe reconstruction must not be restarted as missing work.
