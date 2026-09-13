# Whole-run, full-task and 840-cell acceptance readers

This milestone implements the previously missing **whole-run and full-task/grid artifact
acceptance** for the revised primary experiment, then tests real producer files and one real
diagnostic evaluator output. It does not execute the primary matrix, accept historical results,
or produce a paper finding. No implementation incident here belongs in any manuscript section.

## What is implemented

The new [primary_completion.py](source-current/src/embed_optim/primary_completion.py) is bound by
[dense_primary_v2_completion_protocol.json](dense_primary_v2_completion_protocol.json). That
reader lock is a new **non-execution-authorized proposal**; its parent is the unchanged
`dense_primary_v2_protocol.json` (`e21a7226...`). No old source lock or criterion is updated.

Whole-run admission requires the exact full run identity, scientific recipe and actual terminal
Trainer step/epoch; the complete scheduled checkpoint prefix; a continuous, non-overlapping timing
ledger from step zero; agreement with recorded system measurements and actual file sizes; and a
final inference export whose weights, model configuration, pooling and tokenizer bytes match the
terminal checkpoint. The formal wrapper also requires the original training-view fingerprint.

Every checkpoint's entire content seal is checked before deeper deserialization. The CPU reader
uses `weights_only=True` for optimizer/scheduler state, checks all **134 DenseOn tensors and
149,014,272 parameters** for the declared topology and finite values, exact optimizer group and
moment-state semantics, finite/nonnegative second moments, complete parameter-state coverage,
Adam counters, scheduler horizon/LRs and the actual new NorMuon implementation label. No trainable
model is instantiated and no parameter is updated. Payloads and terminal metadata are checked
again afterward. The new reader does not use the old aggregator's obsolete shared Muon/NorMuon
implementation-label assumption.

Full-task admission requires the exact fourteen distinct task files, immutable dataset revisions,
expected split/default subset, pinned MTEB/package versions, checkpoint-specific model metadata,
768D cosine semantics, and finite normalized nDCG@10 matching `main_score`. The exact primary
admission record and protocol/checkpoint-content-addressed cache must also match. A worker's exit
code or a score filename by itself is not sufficient.

The full-grid adapter validates **all twelve complete runs before reading any task outcomes**,
then requires all **60 stages / 840 distinct run-stage-task cells**. It emits the exact score-row
schema needed by subsequent analyses without dropping a rate, choosing a BEIR winner or computing
unreviewed statistics. It still reports `scientific_completion=false`: validation selection,
outcome/weight-space/dimension/factorial consumers and publication remain separate required work.

## Actual whole-run rehearsal

[whole-run/final.json](whole-run/final.json) uses the unchanged authenticated outputs of the prior
four-GPU identity campaign. It does not train another baseline. It accepts all **three complete
three-step diagnostic baselines**, covering **nine full-model checkpoints**, all 134 parameter
states per checkpoint, their model/scheduler/optimizer payloads and final exports.

All **three continuation-only directories** are correctly rejected as *whole runs*, despite their
successful Trainer completion at step 3. They contain the continuation's checkpoint 3, not the
earlier checkpoint/timing prefix. This is not a new serialization/optimizer failure or a reversal
of the preceding successful save/resume observation: a successful continuation and a complete
reconstructed run are different claims. Preserve/reconstruct the complete accepted prefix for a
whole-run handoff; do not silently add zero timing, copy a later branch or pretend absent stages
exist. The reader accepts ordinary in-place continuation only when the retained prefix is complete.

The first new reader attempt failed all three positive controls because it assumed every diagnostic
dataset has the formal data manifest field. The actual synthetic producer omits it. The preserved
[initial.json](whole-run/initial.json) and initial source show that exact failure. The reader now
checks a manifest whenever the expected input identity contains one; the actual primary identity
always contains its authenticated 500k materialization manifest. The real scheduler inspection also
confirmed the pinned LambdaLR's empty-dictionary partial-function state. Neither production data,
optimizer state, numerical tolerance nor a primary criterion was changed. The intermediate successful
source/result pair is retained as well; final source was exercised again after the result-reader work.

## One real task, then CPU-only reader replay

The one-GPU diagnostic ran the unchanged BF16/FA2 full-corpus **SciFact** worker at the original
immutable dataset revision. It used the previous Muon **diagnostic** checkpoint-3, authenticated
before and after, in `/tmp/dense-primary-task.arBdZO`; the model was not a new primary model.
It leased only GPU 4 through the shared cooperative lease. The original BEIR controller was not
resumed. The evaluator exited zero and the lease was released; no training or HF write occurred.

The first *reader*, not the evaluator, failed: the raw task file includes six undefined auxiliary
`nauc_recall_at_100*`/`nauc_recall_at_1000*` values. The pinned MTEB implementation explicitly returns
NaN when its auxiliary nAUC normalization is undefined, and its serializer retains those values.
The study's predeclared **nDCG@10 and main score are finite**. The old frozen scientific reader
independently accepts this same raw result; the evidence validator verifies that compatibility.

The new reader therefore records recognized undefined auxiliary nAUC fields separately, without
using them in nDCG, imputing zero, changing the raw JSON, dropping a task or changing the scientific
estimand. Non-finite nDCG, main score, time, unknown fields or infinities still fail. The original
failed reader result, raw NaN bytes, model metadata, run settings, worker log and source are preserved
under [task/](task/). The same unchanged files pass the [final CPU reader replay](task/reader-final.json).
There was **no second GPU evaluation**. A separate source-level constant-metric control reproduces
MTEB's undefined branch; it does not reconstruct the actual task's per-query confidence vectors or
claim a cause for every observed auxiliary value.

This verifies one real output schema and its full-corpus worker path, **not** all fourteen actual
tasks, the full 840-unit grid, optimizer quality or a primary retrieval finding. Do not use the
diagnostic nDCG value in a paper table. The raw model checkpoint remains unchanged and unuploaded.

## Tests and reproducible inspection

All **63 focused tests** pass, covering finite/tensor/optimizer/scheduler semantics, timing gaps,
missing or duplicated stages/tasks, invalid primary scores and metadata, explicit auxiliary-metric
handling, source-lock drift, and twelve-run-before-840-cell adapter ordering. Tensor/schema/grid
fixtures are explicitly synthetic mocks, not fabricated scientific runs. The actual producer
rehearsals above are separate evidence. All **1,538 isolated repository tests** pass with zero
failures/errors/skips. The earlier 55-case receipt is retained.

Those tests do not certify a different checkout. The numerical identity candidate still has its
earlier **963 cases: 952 passing and eleven unwaived old source-contract failures**. No numerical
candidate test was reclassified or its old locks relaxed in this milestone. The source remains
uncommitted, undeployed and unpushed.

The new real full-grid inspection is read-only and currently **fails correctly because the new
primary run directories do not exist**:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src \
  python3 -B -m embed_optim.primary_completion matrix \
  --completion-lock /root/embedding-optimizer-story-refactor/configs/dense_primary_v2_completion_protocol.json \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_primary_v2_protocol.json \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --experiment-root /root/embedding-optimizer-study \
  --results-root /root/embedding-optimizer-study/results/dense-primary-v2
```

The [commands.json](commands.json) record retains the actual calls and expected/unexpected failures.
The evidence validator checks the preceding archive's thirty bindings via exact before-copies,
real producer payloads and current source, preserved failures, unchanged task output, compatibility
with the original nDCG reader, full-grid refusal, test coverage, original numerical source,
manuscript and the three exact stopped BEIR handles:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  python3 -B -m scripts.validate_dense_primary_completion \
  --repository /root/embedding-optimizer-story-refactor --output /tmp/new-primary-completion-validation.json
```

## Next steps and authority

Whole-run/task/grid reader preparation is now implemented and directly tested; do not keep treating
it as an unimplemented checkbox. Next prepare **natural-data execution readiness**, integrate the
new identity/grid with validation selection and outcome/weight-space/dimension/publication consumers,
and separately integrate the routed factorial control. The exact-source assembled checkout,
evidence-preserving old-controller handoff and owner deployment/development-publication direction
remain prerequisites for formal execution. Do not use the obsolete zero-step migration, import
historical scores, or infer primary approval from a diagnostic worker call.

The old primary matrix remains **12/12 executed, 60/60 preserved, on scientific hold**. No new
accepted primary scientific result exists. All owned jobs have exited, including the single
diagnostic evaluator. The exact original BEIR dispatcher chain remains stopped with unchanged
ledger/lease/four-step prefix. Production numerical source, both prepared numerical candidates,
old protocols/checkpoints and manuscript are unchanged. No formal training, original BEIR resume,
controller transition, commit/push, HF deletion or protected GPU-helper interaction occurred.

This milestone is local only. The owner WIP/deployment question and no-pending-results pre-commit
rule remain unresolved; the previous GitHub integration 403 was not retried or bypassed. Do not
claim the new source/evidence is available from a public clean clone. The source-bound raw artifacts
are engineering provenance only; no blog or implementation-error manuscript content is reintroduced.
