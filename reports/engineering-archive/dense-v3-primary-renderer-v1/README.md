# V3 primary bridge rendering — bounded candidate component

Updated 2026-09-07 08:07 UTC. This is separate from the still-running joint replay
74613 and does not modify any of its thirteen bound sources. No formal training,
primary admission, manuscript installation or publication occurs here.

## Concrete interface gap and implemented component

The preserved old publication parser accepts the complete synthetic signal and
baseline-equivalent controls, but rejects v3's legitimate unidentified-extension
and unresolved-nonzero-direction cases. All four controls use the existing
complete sixty-row/nine-feature numerical kernel, not hand-entered summaries.
The [probe result](probe-result.json) retains every table and the old errors.

A separate scalar exact-error example has a positive mathematical RMSE reduction
whose binary64 display underflows to zero; the old parser rejects its exact
positive decision. This is explicitly **not a complete OLS fit or model result**.
It tests only the contract that displayed values must not decide support.

The [candidate renderer](candidate/primary_v3_publication_render.py) recomputes all
four folds, 540 predictions and nine associations before rendering. It retains
all nine features, exact decisions, defined-fold counts and association statuses.
Undefined comparisons remain undefined, never negative evidence or available-case
pooling. Baseline-equivalent and resolved-but-unsupported cases remain distinct.

## Verified scope

- [Fourteen focused cases](focused-first.xml) pass, zero failures/errors/skips.
  They do not change the separate latest whole-development count of 2,573.
- Four synthetic ACL-style samples compile twice with no overfull boxes and no
  Type 3 fonts. The [bounded layout/source check](layout-check.json) is
  `c57277b1b7870b536a9ea77401f533f83145811f0f4f16c37fd2a42dfcda225a`.
- The baseline-equivalent and partially undefined tables were visually inspected;
  columns, labels and missing-value markers are readable and not clipped. The
  [samples](layout-samples/) are artificial layout tests, not paper findings or a
  complete eight-page manuscript-layout verification.
- The first sample generator failed before output because it used the wrong ACL
  style-file address. Its exact source is retained in [layout-attempt-1/](layout-attempt-1/).
  The retry uses the actual vendored style; no numerical kernel changed.

The component is **not integrated into the package or complete v3 publication**.
Still required: complete primary/secondary/dynamics/system/weight-state rendering,
checkpoint-backed authoring and source admission, the local-role publication
closure, the functional include and strict manuscript-consumer integration.
Synthetic outputs must never be copied into `paper/generated/`.

## Reproduce only this component

Use the CPU-only environment and pinned interpreter from AGENTS.md. From this
development checkout, run the explicitly scoped tests:

```bash
/usr/bin/python -B -m pytest -q \
  reports/engineering-archive/dense-v3-primary-renderer-v1/test_candidate.py
```

`probe.py` writes to a new result path. `render_demo.py` requires a new empty
temporary directory; `check_demo.py` checks compiled samples and writes a new
receipt. The first retry directory is `/tmp/dense-v3-primary-render-retry.F3cNl6`;
its complete generated payload is also preserved in `layout-samples/`.
No script here authorizes deployment, a controller change, commit/push or HF writes.
