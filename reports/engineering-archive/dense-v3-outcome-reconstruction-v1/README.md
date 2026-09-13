# Portable original outcomes — complete reader verification

Updated 2026-09-07 04:35 UTC. **All numerical/test runs are terminal/pass; no scientific result is supplied.**
The new CPU reader reconstructs every original validation/BEIR input and all ten outcome tables.
62 focused cases, both complete cold audits and the 2,481-case full retry pass. The separate
[acceptance record](validation.json) checks this bounded scope and preserves earlier failures;
use [the exact execution record](RUNNING.md), not old preparation counts.
The previous fixture-only README is preserved in [before/fixture-preparation-README.md](before/fixture-preparation-README.md).

The preceding [raw-vector acceptance](../dense-v3-vector-reconstruction-v1/validation.json)
has since completed at 03:30:55 UTC, SHA-256
`a5ec82bafa467462dc17600ae2c041e0cf5f67f988bc028a0983d5e47c82aed1`.
Its two cold audits and all owned numerical sessions are terminal. Preserve its 39 bindings,
25 prior bindings and 12,474 external checks before implementing this separate next reader.
That parent acceptance does not include this separate outcome reconstruction milestone.

## Preserved first raw-scoring fixture

The [exact generator](fixture-source/prepare.py) ran as owned session **22465**, terminal exit 0.
Its [receipt](fixture.json) is copied from `/tmp/dense-v3-outcome-primitives.Hmgb0X/result.json`,
SHA-256 `aa71790c4a7a3c803d921593c72570b70889bf957a84ed09a90f1dcd72e27ff9`
(828,270 bytes). The complete retained producer is
`/tmp/dense-v3-outcome-primitives.Hmgb0X/explicit-synthetic-producer`.

- Actual frozen v3 contracts and all 4,096 validation row identities are read and revalidated.
- Twelve synthetic validation jobs retain 49,152 rows, each with eight FP32 candidate scores
  and all six original FP32 metrics. Existing scalar replay, full-row summaries and selection
  readers inspect every job. Selection finishes before any BEIR fixture is generated.
- Sixty synthetic checkpoint jobs retain all 840 pinned task results, model metadata, exact
  run settings and per-job admission-shaped records. Existing task readers verify every file;
  allowed unused auxiliary NaNs remain explicit and never replace the primary nDCG.
- The unchanged outcome kernel builds and freshly inspects all ten complete original tables,
  including the systems-table schema. Every score, checkpoint admission, timing and hardware
  value is **synthetic**. Real source/data identities do not make these model results.
- The receipt binds 52 source files and 2,643 generated files / 35,396,744 bytes. Its explicit
  primary-admission, cold-relocation and scientific-completion flags are all false.

The generator's command was:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  /usr/bin/python -B /tmp/dense-v3-outcome-primitives.Hmgb0X/prepare.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --validation-data /tmp/dense-partition-candidate.kHXoGW/validation \
  --output /tmp/dense-v3-outcome-primitives.Hmgb0X/explicit-synthetic-producer
```

These paths are preserved evidence, not rerun destinations. Use new explicit directories for
another attempt. No encoder, corpus retrieval, formal run or remote operation was performed.

## Implemented local-role reader

The new `primary_v3_outcome_primitives` and `primary_v3_outcome_reconstruction` modules derive
the exact v3 BEIR checkpoint wrapper/cache key, reconstruct every validation plan and selection
from raw records, parse all 840 task results, and compare all original tables exactly. They use
the same complete recorded run population as vector admission and retain original provenance
strings without opening those locations. Checkpoint/data metadata is authenticated, not freshly
verified against absent model tensors or raw text. Timing is retained, not remeasured.

The [first complete audit](complete-audit.json) uses all 49,152 raw records and all 840 cells.
Its fresh child imports only 71 archived package modules, exactly reconstructs 14 numerical files
covering ten tables, and rejects 21 independently rehashed semantic controls. A one-unit change
to the primary mean still fails fresh calculation, with new numerical output retained. See
[commands.md](commands.md) for source/runtime/anchor details and explicit claim boundaries.
The original validation/statistical kernels, all rates/stages, selection rules and undefined-
value policies are unchanged. This repeats previously checked numerics, not an independent new
statistical implementation, actual model encoding/retrieval or physical cross-host execution.

The first raw fixture's upstream placeholders remain unchanged and fail the new complete schema.
The separate [complete integration fixture](complete-fixture.json) supplies all seal/run-identity
fields and consistent logical payload metadata in a new namespace, explicitly without actual
checkpoint payloads. Unit tests separately synthesize all query identities for self-contained
testing; the cold fixture retains the previously verified actual validation identities. Neither
fixture is actual primary evidence. The actual authoring entry point still refuses absent runs.

The [first failed focused/full tests and cold audit](attempt-1/README.md) are preserved with exact
source and inputs. Only the new test's overly broad message assertion and the audit launcher's
old editable-install search paths changed; the access guard and production readers did not.
Never turn a simulated checkpoint schema, successful replay or green test count into primary
acceptance, a publication permission or an optimizer superiority claim.

The subsequent [geometry/bridge/inference closure](next-geometry-boundary.md) remains separately required. Follow the
[inspected boundary notes](../dense-v3-vector-reconstruction-v1/remaining-closure.md), including
the distinction between retained spectral measurements and fresh model-weight computations.
Keep all existing operational, source-release, primary and manuscript gates in force.
