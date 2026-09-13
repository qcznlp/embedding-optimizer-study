# Revised primary contract: actual inputs and shared checkpoint admission

This milestone prepares the next primary training/backup/evaluation boundary. It is **not an
execution authorization, a completed primary run, an actual upload/evaluation campaign, or a
scientific result**. Nothing in this engineering archive belongs in any manuscript section.

## Actual input evidence

[inputs/input-bindings.json](inputs/input-bindings.json) was produced at **2026-09-06 11:10:33 UTC**
by the actual full-identity candidate, `/tmp/dense-identity-source.8rUgLF`, with GPUs hidden.
The original materialized dataset and row-manifest linkage pass for **500,000 queries**. All
**16 data payload files / 2,147,849,132 bytes** are content-bound. The cached untrained
`lightonai/DenseOn-unsupervised` base is independently compared against HF's immutable revision
`0edbd55684eb782bce55ee74c95b25c97cbe7f43`: **11/11 files** match LFS SHA-256 or Git-blob SHA-1
and byte counts. The saved receipt distinguishes these digest formats.

One actual common data/model identity is observed; twelve configuration identities are derived
from it. This is **not twelve independent execution preflights**. No model load, optimizer update,
device setup, data rewrite, HF write or training-output creation occurs. The original model, data,
seed, explicit seven-negative/no-in-batch objective, context 8192, epoch, global batch 128,
learning-rate grid and five stages are retained. The common inputs are real, not the synthetic
288-row diagnostic fixture used in the preceding GPU campaign.

## Prepared consumer implementation

[proposal/dense_primary_v2_protocol.json](proposal/dense_primary_v2_protocol.json) remains
`prepared_not_execution_authorized`. Its byte identity is
`e21a7226c09740d38d49855738267c080f615f8d4f60c2abf9dbf3ccd874edd2`.
It binds the audited input receipt, fourteen prepared training files and eighteen consumer/runtime
files. It retains the exact original fourteen BEIR tasks and their immutable dataset revisions,
the five checkpoint steps (782/1563/2345/3126/3907), original scientific analysis choices, and two
disjoint four-device pools. The excluded historical implementation comparison is not carried into
the scientific analysis. No older protocol is modified or relabeled.

- `primary_contract.py` is the shared input boundary: fixed run/config/source identity plus the
  complete checkpoint seal, all model/optimizer/scheduler and per-rank RNG payloads, and actual
  Trainer stage. It rejects undeclared identities, corrupt/incomplete payloads and unsealed files.
- `primary_training.py` offers read-only inspection. Actual execution requires a reviewed release,
  **one assembled checkout containing the exact committed sources**, runtime validation, no short-
  horizon overrides, and an actual preflight identity equal to the frozen recipe. It cannot combine
  the old numerical checkout and the new consumer checkout into a formal run.
- `primary_io.py` reuses the same checkpoint boundary for backup and evaluation. Backup is additions-
  only in a new protocol-content-addressed namespace, refuses existing remote prefixes/receipts,
  and verifies every payload at the **returned upload commit**. Read-only re-audit keeps that original
  commit; it does not replace it with current HEAD. This is tested with an explicitly mocked backend;
  **no real new upload or remote restore has been performed**.
- Prepared training and evaluation acquire the same cooperative GPU lease. Evaluation namespaces
  depend on protocol and complete checkpoint content, require their own exact admission record,
  and use the unchanged pinned full-corpus worker. **No new worker was run.** Worker exit zero alone
  is expressly not complete fourteen-task scoring, whole-run acceptance or publication readiness.

The selected consumer sources are copied byte-for-byte under [consumer-source/](consumer-source/).
They are new local files in the isolated development checkout, not deployed into either numerical
candidate or the live experiment. The complete training source was already preserved under the
preceding `dense-full-identity-v1/candidate-source/` archive. The read-only loader can check those
separate roots; state-changing consumers cannot. Digests detect substitution/drift and are not
digital signatures or a substitute for trusted independent remote-download authentication.

## Real-file rehearsal and tests

[rehearsal/result.json](rehearsal/result.json) records a fresh CPU-only rehearsal against the
authenticated outputs of the preceding real four-GPU entrypoint campaign:

- **12/12 real diagnostic checkpoints** pass the shared payload reader under their exact diagnostic
  identities; all **240 payload files** remain identical to the original producer receipt.
- All twelve diagnostic identities are rejected when compared to the corresponding **primary**
  recipe. Their successful diagnostic admission does not make them primary training artifacts.
- All **60/60 retained old primary checkpoints** fail the new primary admission without a model or
  pickle load. No identity/seal is retrofitted and no checkpoint is modified or deleted.
- All **three real one-byte-corrupted optimizer copies**, preserved by the preceding admission
  campaign, are rejected at the optimizer file digest. These copies are never continuation inputs.
- **12/12 real read-only CLI plans** match the frozen recipes. A real CLI `--execute` request against
  the draft fails before training. No formal output root, model update, HF write or evaluator appears.

The new focused tests pass **51 cases**: 44 consumer/schema/fixture/mocked-backup checks and seven
real-receipt coverage/claim guards. The earlier 44-case receipt and 1,468-case full-suite receipt
are retained. The final full isolated suite passes **1,475 cases**, zero failures/errors/skips,
recorded under `tests/isolated-full.xml`.
No test was excluded and no frozen numerical tolerance or old source lock was relaxed.

These tests belong to the **isolated paper/audit checkout**, whose old numerical core is unchanged.
The separate identity candidate still has its earlier **963 cases: 952 pass, 11 unwaived old
source-contract failures**. This milestone does not turn that different source tree green.

## Read-only reproduction

On this host, the primary recipe can be inspected without GPUs or a training output:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src \
  python3 -B -m embed_optim.primary_training \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_primary_v2_protocol.json \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --experiment-root /root/embedding-optimizer-study \
  --run-id verified-muon-3e-4 --inspect
```

Recorded input-preparation, rehearsal and test commands, plus the explicitly labeled draft-builder
reproduction invocation, are in [commands.json](commands.json).
The evidence validator rechecks preserved predecessor bindings, current input/source/payload
identities, the real rehearsal and test coverage, the unchanged manuscript/numerical source,
and only the three exact retained BEIR dispatcher handles:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B -m scripts.validate_dense_primary_contract \
  --repository /root/embedding-optimizer-story-refactor --output /tmp/new-primary-contract-validation.json
```

The real payload rehearsal requires the explicitly named host-local artifacts. They were not
uploaded in this milestone; the small archive is not a portable model-state download. Input
preparation and the earlier GPU campaign remain distinct source-bound receipts, not rerun claims.

## Remaining execution work and scientific boundary

Next implement/review whole-run terminal and complete-task acceptance, then adapt downstream
validation selection, outcomes, weight-space/dimension and publication consumers to the new
primary identity. Preserve their scientific estimands and selection rules. The routed factorial
AdamW control still needs its own reviewed numerical integration; do not bypass that gate.
Natural-data execution readiness and the evidence-preserving retirement/handoff of the old paused
controllers remain pending. Do not reuse the obsolete zero-step migration or adopt old results.

The primary comparison is still **12/12 executed, 60/60 preserved, on scientific hold**. No new
accepted retrieval or mechanism result follows here. Formal deployment/WIP-publication direction
and the no-pending-results pre-commit rule remain unresolved. The previously recorded GitHub
integration 403 was not retried or bypassed. This milestone is local only: no commit/push, formal
training, evaluation resume, controller transition, HF deletion or protected GPU-helper interaction.

Prior source/results are preserved, including exact previous status pages in `before/` and
`before-live/`. All owned diagnostics have exited; the same original three BEIR dispatchers remain
stopped with the unchanged ledger, lease and four-step completed prefix. Paper content is unchanged.
