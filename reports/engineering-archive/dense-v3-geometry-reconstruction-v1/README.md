# Original geometry reconstruction — complete numerical verification

Updated 2026-09-07 06:00 UTC. **All final numerical/test calls are terminal/pass; initial failures are preserved.**
No primary training, retrieval result, scientific finding or publication is supplied here.
The [plan](plan.md) records the exact parent and complete native-shape scope.

## Implemented scope

The local-role reader checks twelve complete recorded v3 run identities, five stages each,
all 88 hidden matrices and every native FP32 rank-16 basis. It validates recorded exact-entry,
metric and spectral-health schemas, including zeros, before using the unchanged aggregation
kernels. It freshly reconstructs 60 checkpoint rows, 660 run-pair overlaps and 60 optimizer
summaries; the 10,560 health rows retain authenticated original measurements.

No checkpoint tensor, weight metric, singular spectrum or spectral-health quantity is newly
measured by this portable reader. Source/data/model locations in the provenance stay unchanged,
but are never used as file-location fallbacks. Complete source contracts and an externally
provided archive hash are required. The primary authoring entrypoint still refuses absent runs.

## Evidence already completed

- The first [real-record check](real-record-check/result.json), SHA-256
  `9a55ba6493ce844651f2ac1997a48c30f01c914d2dcf6955203ec3b6c816c966`, checks nine previously
  accepted diagnostic stage files: 792 raw matrix records and 1,584 health records. Fresh
  checkpoint rows match exactly. It is not a full sealed-run or basis-payload admission.
- The complete [first same-process smoke](first-smoke/result.json), SHA-256
  `d6a2e9d9f38fae79b64575f0a18f0ef92683b2196b0f2630f94f6a1da5d7a646`, uses 60 synthetic
  states with actual native shapes, a full-zero first stage and heterogeneous later zero/nonzero
  matrices. Its four tables match exactly. It predates the zero-statistic consistency check.
- The [independent scalar/set oracle](first-smoke/independent-oracle.json) verifies 14,480
  one-hot native-axis bases, all 660 pair overlaps (132 undefined), parameter-weighted checkpoint
  summaries and all 60 optimizer groups. It calls no production overlap/aggregation/census
  helper. Rational-vs-floating-point checks use 2e-14 relative / 2e-15 absolute tolerances;
  the production replay remains exact, with no changed statistical or spectral threshold.
- The new reader initially checked a zero spectrum but omitted the requirement that an exactly
  zero matrix also has zero row/column statistics and row energy. Ten specific counterexamples
  fail on the preserved [initial source](attempt-1/source/primary_v3_geometry_primitives.py).
  The check was added without modifying a weight/optimizer/statistical kernel. The resulting
  [real-record retry](real-record-check-retry/result.json), SHA-256
  `950c7e65098f208ed7861f6c2cf1cc9d73d2797781b07acc34279887bb31b8dd`, still matches all nine
  diagnostic checkpoint rows exactly.
- The first complete focused suite has 63 cases, with 62 passing and one test-helper failure:
  `copy.copy(module)` raises before the deliberately foreign-source check. Its exact
  [failed XML](attempt-2/focused-failed.xml) and source are preserved. The helper now uses
  `ModuleType`; the production source gate is unchanged. Four independent-oracle refusal
  controls were added, producing a 67-case retry. That retry is now terminal/pass,
  without errors, failures or skips; see [focused-final.xml](focused-final.xml).

## Complete cold checks and bounded acceptance

Use [RUNNING.md](RUNNING.md) for exact owned process handles and namespaces. The 67 focused
cases, full 2,548-case regression and both complete cold audits are terminal/pass.
Independent replay 20415 matches all eight numerical outputs; both attempts refuse every one
of the 28 fully rehashed semantic controls. Each child imports only 72 archived package modules
with network and original producer/source paths blocked. The source-bound `validation.json`
records only this bounded reconstruction scope after documentation/preservation verification.
Do not promote an observation timeout, passing prefix or same-process smoke into completion.

Every checkpoint admission, raw weight measurement, score-independent timing and coordinate
basis in the integration fixture is explicitly synthetic. Authentic source contracts and shape
metadata do not make it primary evidence. It does not establish physical cross-host portability,
the original retrieval bridge, functional inference, model encoding or primary publication.
The original-outcome/vector/inference parents remain unchanged and are not rerun as missing work.
No implementation incident here belongs in any manuscript section.
