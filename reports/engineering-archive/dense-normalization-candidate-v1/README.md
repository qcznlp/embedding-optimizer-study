# Single-normalization repair candidate

Status: **both tiny and real full-model four-rank CPU checks pass complete and partial
accumulation; no deployment or primary training pause.**
This is engineering evidence, never manuscript content or an optimizer-quality finding.
The unchanged live path still has the failure documented in
[`../dense-ddp-accumulation-v1/`](../dense-ddp-accumulation-v1/README.md).

## Candidate and actual checks

`scripts/dense_trainer_normalization_candidate.py` explicitly subclasses the study Trainer.
It retains Trainer's batch scheduling and loss normalization by the **actual current**
accumulation count, while setting Accelerate's extra backward divisor to one. It changes
neither the mean contrastive loss, optimizer, clipping threshold nor learning rates.
It checks the pinned library/source identities and refuses unsupported distributed modes.
Use one active Trainer per process because Accelerate's gradient state is process-shared.
This is an undeployed candidate, not an installed-library monkeypatch or a new primary matrix.

The separately executed `scripts/audit_dense_trainer_normalization.py` uses four CPU/Gloo
ranks, a random two-layer 32D ModernBERT, FP32/SDPA, zero dropout and non-reentrant gradient
checkpointing. Each optimizer consumes exactly 288 unique synthetic queries in three global
groups: **128, 128, 32** (eight queries per rank, accumulation **4, 4, 1**).

- `tiny-candidate.json`: all three optimizer fixtures pass all three raw/clipped gradient
  checks against the independent whole-group float64 row-wise reference. All 14 parameter
  tensors are checked at each step. Maximum raw element error is **3.458e-6**, and maximum
  clipped error **2.198e-7**, under the unchanged original absolute 5e-6 / relative 5e-4 tolerances.
- The learning-rate factors are **0, 1, 0.5**, followed by zero after exactly three scheduler
  advances. The first warmup update has zero LR; two updates have positive LR. Final weights
  agree exactly across all four ranks. This is not independent optimizer-algorithm validation.
- `negative-control.json` deliberately uses the scheduled divisor four on the final one-batch
  group in the diagnostic instance. The full groups pass; the raw-gradient check fails on the
  32-query tail, with norm ratio **0.2499999173**. The three gradient observations are retained.
  This establishes that the test rejects a constant-divisor workaround; it does not change any
  production method or relax a tolerance.
- `unit-tests-v1.xml` records 24 initial guard checks. Its text archive has one added terminal
  newline compared with the original pytest XML; test values are unchanged. `unit-tests-v2.xml`
  is directly emitted by pytest and records **36** passing guard checks, including the full-model
  configuration guard. Neither unit suite executes the standalone four-process diagnostics.
- `full-tests.xml` records **1,256** passing unit/regression tests (zero failures/errors/skips),
  before the subsequent handoff-document refresh. The standalone distributed receipts are separate.
- `full-tests-final.xml` repeats all **1,256** checks after the handoff refresh, again without
  failures, errors, skips or exclusions. [`validation.json`](validation.json) binds the executed
  source and verifies the distributed receipts, preserved predecessor evidence and current state.

`observed-source/` preserves the exact candidate, small/full diagnostic source and initial
unit-test source. `before/` preserves the preceding handoff and progress snapshot. The original
failed production-path diagnostic and its receipts remain unchanged.

## Real full-model extension — passed at 07:07 UTC

`scripts/audit_dense_full_model_normalization.py` applies the same actual three-step audit to
the independently downloaded NorMuon 3e-4 checkpoint 3126. All 18 downloaded files are verified
against the trusted ten-run HF audit **before** model loading. The expected model has 134
parameter tensors, 88 hidden matrices and 768 embedding dimensions. Each optimizer starts
fresh from the same downloaded weights; no saved optimizer is resumed or primary output changed.

Only the diagnostic fixture loader is substituted, within the diagnostic process. The ordinary
Trainer training step, production loss and optimizer builder remain unchanged, and the explicit
candidate owns the normalization repair. Configured context remains 8192, but the synthetic
inputs are short: this does not test maximum-length execution, GPU/NCCL, BF16/FlashAttention,
data-loader/rank-RNG restoration, retrieval quality or scientific replication validity.

`full-model-candidate.json` verifies all three optimizers, all nine raw/clipped gradient checks
and all 134 parameter tensors per check. Maximum raw element error is **4.054e-6**, maximum clipped
error **1.139e-6**, under the same pre-existing tolerances. Each case consumes all 288 distinct rows
once, has exact cross-rank final weights and the expected LR cadence. AdamW's final reference norm
is small (about 4.84e-4), so its roughly 0.25% relative norm difference is reported rather than
claiming uniformly tiny relative errors; all fixed raw-gradient and norm comparisons pass.
All 18 files / 1,351,464,827 bytes are unchanged after the full audit. This is a three-step CPU
fixture from one trained checkpoint, not a measurement of the primary-training error's impact.

The completed full-model attempt used `/tmp/dense-full-normalization.637tWc` and the isolated
checkout's explicit `src` path. The five core training/loss/optimizer/collator/config files are
byte-identical to live main. Its final receipt has `complete_for_full_model_cpu_fixture=true`;
this does not change the failed verdict on the unchanged production path.

## Reproduce

These commands require local process-to-process sockets. They hide all GPUs and use only
new temporary directories. They neither pause training nor modify live files.

From the isolated checkout, for the small-model candidate:

```bash
candidate_audit_dir=$(mktemp -d /tmp/dense-normalization-audit.XXXXXX)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ACCELERATE_USE_CPU=true \
PYTHONPATH=/root/embedding-optimizer-study/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
-m scripts.audit_dense_trainer_normalization --repository /root/embedding-optimizer-study \
--workdir "$candidate_audit_dir"
```

Repeat in another fresh directory with `--negative-control-constant-tail` for the expected
tail-only failure. Do not reuse the successful directory or modify its receipt.

For the full model, expose the isolated checkout explicitly because its authentication helper
depends on `embed_optim.artifact_inventory`, which is not yet published on live main:

```bash
full_audit_dir=$(mktemp -d /tmp/dense-full-normalization.XXXXXX)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ACCELERATE_USE_CPU=true \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
-m scripts.audit_dense_full_model_normalization \
--repository /root/embedding-optimizer-story-refactor --workdir "$full_audit_dir" \
--download-root /tmp/dense-checkpoint-restore.Og1kZs/download \
--audit reports/experiment-integrity/huggingface-digest-audit-ten-run.json \
--audit-sha256 9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6
```

The download root must already have been independently materialized according to
`docs/checkpoint-restoration.md`. This command does not download or repair missing content.

## Diagnostic launch history and remaining authority

The first small-model launch was denied localhost sockets by the sandbox, before any ranks
executed the audit. Its own tool session was interrupted (exit 130), then a fresh CPU-only
launch with the necessary socket permission succeeded. The first full-model launch selected
live `src` and exited at import because the isolated artifact helper was absent; it never loaded
a checkpoint. The subsequent attempt selects the isolated `src` with verified identical core
training bytes. These are diagnostic launch failures, not primary optimizer failures.

The owner has not yet approved pausing the final two primary jobs/automatic post-processing,
runtime deployment, re-training, WIP source publication or the rejected HF withdrawal scope.
The original matrix, installed libraries, live controllers, HF content/history and `gpu.py`
remain unchanged. Independent checkpoint backup continues. CPU repair checks do not resolve
the scientific impact of earlier updates and do not authorize silently relabelling existing runs.
