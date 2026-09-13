# Code correctness audit — September 5, 2026 (UTC)

Engineering evidence only. **The training stack is not cleared for scientific use.**
This responds to the owner's request to verify code first; it is not a primary experiment,
production fix, approval to stop evaluation, or manuscript material.

## Current verdict

| Surface | Evidence-backed verdict | Boundary |
| --- | --- | --- |
| Live accumulated Trainer | **Failed** the independent global-mean gradient check | Earlier four-rank CPU audit reproduces a 0.25000001 gradient scale; unchanged live source still contains the double normalization |
| Isolated normalization candidate | Tiny and full DenseOn four-rank CPU gradients passed for all three optimizers | Not deployed; no GPU/BF16/FA2/8192-token acceptance |
| NorMuon fidelity to pinned upstream | **Not conformant across the tested small-gradient envelope:** 44 of 96 updates differ | Synthetic small matrices; not proof of impact on actual training or retrieval |
| CPU checkpoint-entry restoration | Exact for 24 four-rank resume comparisons, including adversarial dropout | Requires the disclosed CPU transport adapter; fresh Trainers in one process group, not a new-host/process restart |
| Subsequent bitwise replay | **Not passed:** none of the 96 rank-level replay checks is bitwise identical | Small arithmetic differences must not be relabelled exact recovery |
| Regression suite | 1,281 passed; 0 failed, errored or skipped; 25 new diagnostic guards | Unit-test success does not override the failed numerical audits |
| Execution and durability | 12/12 runs completed; 60/60 resumable checkpoints; backup supervisor reports complete | Execution and backup are not scientific validity |

The existing evaluation controller remains in its `decontaminated-beir` step under contract
`4152531e...`. The sealed backup supervisor reports all 60 stages covered. The factorial controller
still waits for main completion under `6605090d...`. No controller, primary source, checkpoint or
external repository was modified by this audit. GPU verification awaits the owner's answer to the
request to pause evaluation safely. Nothing related to `gpu.py` was inspected or touched.

## 1. The existing normalization failure remains decisive

The [original failure receipt](../dense-ddp-accumulation-v1/README.md) and
[isolated candidate receipt](../dense-normalization-candidate-v1/README.md) are unchanged.
Both checkouts still use live `train.py` SHA-256
`e52cfcb5857aa64d4fb826c1f0b12eabe547506de25241a93b88ef1969d1720d`.
The candidate does not enter the live launcher. Passing checkpoint restoration or optimizer
serialization cannot repair the earlier gradient scale or establish its effect on retrieval.
Do not describe the issue as simply a fourfold learning-rate change: clipping and state updates
must be considered, and the counterfactual model-quality impact is unmeasured.

## 2. A newly reproduced NorMuon definition discrepancy

[normuon-reference-edges.json](normuon-reference-edges.json) executes the actual two update
functions from [official commit c6989a8](https://github.com/zichongli5/NorMuon/blob/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py),
after verifying the complete 13,062-byte source against SHA-256
`706c1a35fb35342f6ff207f8310b814a1c8f55c6ffba0e843537db434a696141`.
This does not use the repository test's hand-transcribed reference as its oracle.

Local Newton–Schulz divides by `norm.clamp_min(1e-7)`; the pinned official NorMuon divides by
`norm + 1e-7`. The audit uses shapes 8×4, 4×8, 8×8 and 64×32; elementwise gradient scales
1, 1e-3, 1e-5, 1e-7, 1e-9 and zero; and four sequential updates from zero state for each case.
The fixed reference comparison is `atol=1e-5, rtol=1.3e-6`. All 48 ordinary-scale/zero-gradient
updates agree exactly. Forty-four of the 48 small-gradient updates exceed those tolerances.
The largest measured relative update difference is approximately 0.411 at scale 1e-7 on 8×4.

A paired control changes **only the single denominator statement** in an isolated copy of the
authenticated upstream function. It matches local updates, momentum and row-second-moment state
bit-for-bit in all 96 cases. The original reference and production implementation are not changed.
This localizes the discrepancy; it does not determine whether the real model's actual update inputs
enter this envelope, prove instability, or measure retrieval impact. The intended optimizer
definition must be resolved explicitly before calling the implementation a faithful pinned NorMuon.

## 3. Actual Trainer checkpoint save/resume

The test uses the production collator, explicit loss, optimizer factory, fractional checkpoint
callback and inherited Trainer save/load/data-skip/RNG paths. A two-layer random ModernBERT uses
288 distinct synthetic query groups, 4 ranks × 8 microbatch × 4 accumulation, and groups 128/128/32.
An uninterrupted three-step baseline saves real checkpoints at steps 1, 2 and 3. Fresh Trainer
instances resume from each of the first two checkpoints to the epoch tail. Both live and isolated
candidate Trainers, all three optimizers, and dropout 0/0.1 fixtures are covered: 24 continuations,
96 rank-level comparisons. Dropout 0.1 is an explicit adversarial fixture, not the formal model.

The first attempt [failed before any resumed update](resume-cpu-map-failure.json) because the
installed Trainer passes an indexed CPU device (`cpu:0`) to optimizer-state deserialization, which
this Torch runtime rejects. An optional adapter rewrites only that mapping to `cpu`, and only for
the exact owned synthetic `optimizer.pt` file. It does not change model weights, optimizer math,
data selection, rank RNG or installed source. The [second attempt](resume-bitwise-failure.json)
then completed its first continuation but correctly failed strict bitwise comparison. Both executed
source versions are preserved under `initial-source/`.

The expanded [measurement receipt](resume-measurements.json) retains that strict requirement,
runs every case, and separately measures the differences. In all 96 rank-level comparisons:

- Model weights, complete optimizer state and scheduler are **exactly** restored before the first
  resumed microbatch; consumed row IDs, current accumulation counts, CPU/Python/NumPy RNG traces,
  learning-rate schedules and final step/epoch metadata agree exactly.
- Final weights are identical across ranks within each continuation. This is distinct from being
  identical to the uninterrupted branch; none of the strict replay comparisons passes.
- Maximum raw/clipped gradient element difference across all cases is 9.537e-7. Every measured
  gradient tensor satisfies the earlier gradient-audit tolerance `atol=5e-6, rtol=5e-4`.
- Eleven of the twelve Trainer/optimizer/dropout combinations have maximum final-weight element
  error at most 3.726e-9. Live NorMuon with zero dropout, resumed from step 1, reaches 3.381e-6.
  Its first hidden matrix's row-second-moment state differs by at most 4.323e-5 (relative L2 0.00568),
  exceeding those prior gradient tolerances. The four matching rank reports are one case, not four
  independent incidents. The candidate NorMuon cases do not reproduce that larger difference here.

Those gradient tolerances are additional measurements, **not newly adopted acceptance thresholds
for weights or optimizer state**. The bitwise flag stays false and the diagnostic exits 1. Small
fresh-DDP floating-point reduction differences are a possible explanation, not an experimentally
isolated cause. Exact restored entry state rules out a serialization mismatch in these fixtures;
it does not establish GPU replay, long-horizon equivalence, or scientific outcome robustness.

## 4. All-run data and checkpoint structure

The unchanged read-only auditor completed at 18:09 UTC:
[primary-audit-twelve-run.json](primary-audit-twelve-run.json) verifies all **500,000 stored row
identities and all 12 runs / 60 checkpoints**. There are zero duplicate sample IDs, row-identity
mismatches or invalid negative groups; all frozen source bindings match. Deep model, optimizer,
scheduler, Trainer and rank-RNG payload checks pass. Upstream raw texts were not re-downloaded,
and the separate independent HF content audit still has its explicitly documented ten-run scope.
The automatic backup supervisor's 60/60 coverage is not silently substituted for that audit.

## Reproduction

Use this isolated checkout's absolute `src` and repository in `PYTHONPATH`; the installed package
otherwise resolves to live main. Hide GPUs and disable remote logging. The directories below must
be newly created and empty. No command is a primary training launch.

```bash
audit_work=$(mktemp -d /tmp/dense-trainer-resume.XXXXXX)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ACCELERATE_USE_CPU=true \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  -m scripts.audit_dense_trainer_resume \
  --repository /root/embedding-optimizer-story-refactor --workdir "$audit_work" \
  --canonical-cpu-map-location --collect-numerical-differences

audit_edges=$(mktemp -d /tmp/normuon-reference-audit.XXXXXX)
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m scripts.audit_normuon_reference_edges --output "$audit_edges/result.json"
```

Both commands intentionally exit 1 for their retained nonconformance verdicts. Read the receipt;
do not turn that exit code into a passing gate. The NorMuon diagnostic needs only a read-only HTTPS
fetch of the exact pinned source; model loading remains local/offline.

## Required next decisions

1. Obtain a safe evaluation/GPU handoff before GPU/BF16/FA2/full-context gradient and resume checks.
2. Resolve the intended NorMuon denominator, then validate any isolated correction against the
   pinned upstream reference without weakening the tests.
3. Only after acceptance, decide deployment and whether new training is required; preserve the
   current 60 checkpoints and all failed engineering evidence. Do not silently relabel old runs.
4. Revisit the separate post-processing handoff repair: its old migration requires an inactive
   **zero-step** main ledger. Main now has four completed steps plus evaluation; that migration's
   old preconditions no longer hold and it must not be applied unchanged.

No implementation incident belongs in the paper. No source publication, HF deletion, runtime
takeover or retraining is authorized by a request to verify code. The existing manuscript release
gate and pending WIP-publication approval remain in force.
