# Fixed factorial recipe and actual model-loading factory

Completed locally on 2026-09-10. This is implementation preparation for the
predeclared DenseOn continuation, not a trained branch or a scientific result.
Primary training was already complete; its full-corpus evaluations were not
stopped, changed, duplicated or delayed by GPU work here. This work used CPU only.

## What is implemented

`src/embed_optim/factorial_v3_factory.py` generates the two-source × two-operator
× three-order-seed recipes from the existing run identity. It uses the unchanged
primary model/loss loader, but constructs the separate factorial arguments:
50K groups, random seeded order, four ranks × eight groups × four accumulation,
391 updates, final group size 80, five stages, 8192 context, explicit seven
negatives, temperature 0.02, fixed auxiliary AdamW LR 3e-6 and fresh optimizer
state. Hidden rates must come from both authenticated calibration chains in the
actual preparation path. No primary argument-builder defaults are reused.

The loader checks all 134 saved FP32 tensors and the full normalized ModernBERT
configuration, exact Transformer/CLS-pooling modules, tokenizer backend, token
limits/sides, prompt settings, dimensions, training mode, zero dropout, empty
gradients and enabled gradient checkpointing. Only path/version metadata and the
legacy serialized checkpointing key are excluded from config comparison; live
checkpointing is explicitly required. A changed weight-compatible configuration
must not silently become a different continuation experiment.

The internal constructor requires an already initialized four-rank NCCL context.
It authenticates both calibrations through the existing `prepare_run`, checks
fresh output and isolated logging identity, compares source/recipe/factory
digests across ranks, seeds before loading and constructs the existing bound
Trainer. It does not call `train`, resume, acquire a lease, initialize a process
group, launch a worker, log remotely or publish. It does not waive the separate
main-completion, source, resource or GPU-admission requirements. The separate
factory creation record still needs a future admitted worker/consumer binding.

All 68 accepted parent files, including all 56 primary files, are unchanged.
The additional factory is 13,755 bytes, SHA-256
`bddb443318481d3d7dbddd7b5eb5f5a0f651939537dd281486ad8c305721209b`.
The actual source assembly is preserved in `actual/source-first/`.

## Executed evidence and limits

- Initial focused suite: **26 cases**, no failure/error/skip. Final combined
  suite: **137 cases**, no failure/error/skip, exactly 137 serialized testcases.
  The latter is 30 factory cases plus 107 existing run-component cases, not 137
  independent numerical experiments. Constructor orchestration tests explicitly
  mock the process group, loading and Trainer; they are not four-GPU execution.
- Both actual CPU loading calls use the real unchanged loader on both genuine
  v3 step-2345 sources. Each model has 134 tensors / 149,014,272 parameters, all
  matching its saved FP32 bits. Original source files remain unchanged. The
  explicit diagnostic changes only the requested attention backend to CPU SDPA;
  it performs no forward/backward, calibration, optimizer update or training.
- These calls use **synthetic rate/data metadata** for recipe wiring. They do
  not consume actual GPU calibration outputs or pass the full `prepare_run`
  admission chain. No synthetic recipe is admitted as a formal continuation.
- The first loading call rejects 14 configuration counterexamples; the final
  call adds tokenizer-side/limit controls and rejects **20 / 20**. Each changed
  in-memory field is restored and the complete original configuration rechecked.
- First actual receipt: `actual/actual-loading-first.json`, 67,904 bytes, SHA
  `42d4e7d3597f20b64f973a786751a2b8b4dc92aa6ca7438ea14b1d7381e95c24`.
- Final actual receipt: `actual/actual-loading-final.json`, 68,440 bytes, SHA
  `10516d79cb814d44fc6fda8904a7349f194228f54e3094a857fa5aa27cb21fe8`.

The factory source is identical in both calls. The first audit helper had a
Ruff F821 report concerning its nested reference to a subsequently deleted local
model variable; the actual loading call still completed. The final helper passes
the model as an explicit argument and passes Ruff. Both original helpers, logs,
receipts and test versions are preserved; no failed numerical receipt is hidden.
An initial test import-order lint error and one unapplied patch-context mismatch
were corrected before their relevant final checks; neither launched training.

Actual owned handles are terminal: import inspection 77931; initial tests 29515;
first loading 54074; final tests 94632; final loading 53689. Each exited zero.
Do not rerun these completed checks as missing work. Formatting and lint pass for
the new package, test and final audit helper; no whole-repository release is claimed.

## Current experiment and next work

At the exact 11:03:02 UTC task snapshot, primary BEIR is **16 / 840**, baseline
**14 / 14**; all eight exact task workers are live/R, without a failure receipt.
At 11:03:59, validation is live/S at **7 / 12** and functional encoding is live/S
at **0 / 61**, waiting for validation. The one-GPU priority question remains
unanswered, not approved. No dispatcher, numerical source, old stopped controller,
protected helper, manuscript, HF object or GitHub state changed.

Next finish the full primary evaluations and functional measurements. Factorial
work still requires real GPU calibration, actual four-GPU bound-Trainer admission,
the explicit source/runtime/resource transition, all twelve real branches and
their full outcomes. Cross-host run/calibration transport, complete scientific
consumption, the final NAACL paper and the authorized reproducible release remain
incomplete. Preserve the original scientific choices and all historical gates;
this factory is not a substitute completion criterion.
