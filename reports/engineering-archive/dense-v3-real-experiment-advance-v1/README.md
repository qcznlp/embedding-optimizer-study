# Real calibration, complete vectors and actual four-GPU continuations

Dated evidence through **2026-09-12 15:10 UTC**, archived at 15:13 UTC.
This is real experimental progress, not a claim that the NAACL paper is complete.
The direct owner message “你有权做一切事情，目标是尽快完成任务” supplies the
scoped execution authority. It is not an automatic-continuation approval.

**Later exact update, 15:18:33 UTC:** [training](updates-1518/training.json) is
at **181/391 and 178/391**, with both coordinators/all eight ranks live and
step-79/157 seals present. [CPU features](updates-1518/features.json) are
**25/61 native-verified**, same exact child live/R. The additional deep
checkpoint-read proof remains the step-79 pair; no extra step-157 tensor-read
or full-run completion is asserted. These new observation files are an addendum,
not a rewrite of the original 881-file archive manifest below.

## New real measurements and execution

- Both genuine source-state GPU calibrations finished at **14:37:13 UTC**.
  Both GPU workers exited zero and both fresh native CPU readers exited zero.
  Each independently verified eight fresh gradient shards, 88 hidden-matrix norm
  rows and the original fixed-probe global update/weight calibration.
- The complete **61-state vector matrix** passed original full native readback
  at **14:43:34 UTC**: sixty new exit-zero checkpoint encodings plus the one
  origin-preserving pretrained reuse. No encoding remains to repeat.
- Two actual four-GPU continuations started at **14:55:54 / 14:56:25 UTC**.
  At 15:10:28 UTC their native progress bars are **114 / 110 of 391 steps**;
  both exact coordinators and all eight exact ranks are live. **0/12** full
  branches are complete at this snapshot. The two fixed six-run queues cover
  all twelve cells with balanced reset operators/source states across pools.
- Their first genuine **step-79 checkpoints both pass independent native CPU
  inspection**, completed at **15:09:21 UTC**. Every one of 134 FP32 model
  tensors, all 134 named optimizer states in three groups, the scheduler and
  all four rank-local RNG payloads pass. Wrong external seal anchors are refused.
  This is actual GPU save/CPU readback, **not GPU resume-equivalence evidence**.
- The separate new CPU feature entry started at **15:04:14 UTC**. At the exact
  15:10:28 UTC observation, **10/61 states** have both their original numerical
  computation and fresh raw-vector recomputation verified. No functional
  inferential result is claimed from that incomplete matrix.

## Fixed continuation design

Sources are the genuine AdamW 3e-5 and Muon 3e-4 primary step-2345 weights.
Each receives a reset routed AdamW or Muon under order seeds 314159, 271828,
161803. Every branch uses the same intact 50K groups, one positive/seven
negatives, no in-batch negatives, temperature .02 and context 8192; four GPUs,
microbatch 8, accumulation 4, global batch 128, final 80-group tail, 391 steps,
40 warmup steps and five checkpoints at 79, 157, 235, 313, 391.

| Source state | Calibrated AdamW hidden LR | Calibrated Muon hidden LR |
| --- | ---: | ---: |
| AdamW source | 0.00015039350105964107 | 0.0018331886661728941 |
| Muon source | 0.00015137826846307303 | 0.0018296038299112601 |

These rates target a hidden update/weight ratio of **5e-4 on the fixed calibration
probe**, excluding weight decay. They do not assert equal updates at every later
training step or optimizer-quality superiority. Auxiliary AdamW LR is 3e-6,
weight decay .01, clipping 1. FP32 parameters and the original BF16/FA2 numerical
policy are unchanged. Real hardware is eight L20Z GPUs, not literal H100s.

The accepted 70-file worker assembly remains byte-identical. The new outer entry
only supplies owner/source/runtime admission, inherited dual-namespace leases,
real four-rank NCCL initialization, direct-child failure handling and fresh CPU
native whole-run verification. It does not call or migrate the stopped historical
controllers, waive old release gates, alter kernels, or implicitly resume runs.
Only an exact failed worker's directly owned siblings may be terminated; no
external process or helper is inspected or signalled.

## Preserved failures and bounded tests — engineering only

The first genuine calibration attempt produced all gradient shards but its
native reader rejected the exporter/reader tensor-name disagreement. The v2
outer adapter validates the raw checkpoint-name mapping for all 88 matrices,
retains identical parameter objects, and regenerates both histories in new
directories. The unchanged native tensor/gradient/direction checks then pass.
The GPU loader also uses the unchanged primary runtime input-configuration
function before native verification. None of these engineering incidents
belongs in the manuscript. The failed v1 outputs are retained, not relabelled.

The completed vector producer's first CPU feature attempt exited one at
14:45:18 UTC after its accepted pretrained feature. The native serializer
correctly required an existing nested parent. No vector was lost. A new-only
CPU feature entry prepares those parents; its numerical/recomputation loop is
AST-identical after removing only the new path-preparation/record-path statements.
An initial v2 entry failed before any child/output creation because an adjacent
source-bound import was not on its path. That source, eight-test result and
failure remain intact. The v3 entry adds the exact original helper directory
and a successful actual native-parent import/admission test; no native guard is
replaced and no vector is re-encoded.

The calibration outer entry has **15** final bounded tests; the training outer
entry has **18**; the feature successor has **9**, including actual native
serialization of an explicit tiny fixture and actual complete parent admission.
Positive mocked process/GPU lifecycle tests remain labelled as mocks, not runs.
Earlier versions and synthetic fixtures are preserved separately. No repeated
test count is aggregated into experimental replication.

## Provenance and exact handoff

[verification.json](verification.json), SHA-256
`b512bc3c38c134f44f4b7a99d36591e16c5c96709798f6b4038dea493ab0e869`,
checks **881 copied files**, **123 unchanged protected native dependencies** and
119 retained binary references. Binary payloads remain at their explicitly
indexed experiment paths; this snapshot does **not** claim a new HF backup or
source publication. First-checkpoint binary references originate in the actual
native deep read, not an additional tensor hash by the archive copier.

The original operational documents and unchanged manuscript are in [before/](before/).
The current concise handoff supersedes those dated counts; original evidence is
not edited. Real gradient/feature values are not synthetic fixtures.

- Calibration: [v2 actual completion](calibration/factorial-calibration-v2/run/completed.json)
  and [actual GPU/fresh-reader observer](calibration/preparation-history/observation-first.json).
- Vectors: [native complete matrix](functional/vector-production-and-failed-feature-entry/run/vectors.completed.json).
- First two real checkpoint reads:
  [native result](training/preparation-and-observations/first-checkpoints-native-readback.json),
  SHA-256 `be552d0e556c74bbea4648ce77dcce21bbf4517d025222fb8ba3c52a59f9c887`.
- Training: [exact live observation](training/preparation-and-observations/observation-second.json)
  and [source-bound observer](training/preparation-and-observations/observe_training.py).
- CPU features: [exact live observation](functional/feature-preparation-history/observation-first.json)
  and [source-bound observer](functional/feature-preparation-history/observe_features.py).

Current owned sessions: training pools **57000 / 45053**, CPU feature coordinator
**47384**. Vector coordinator 27585 is terminal/exit 1 (`848f6e`) with all vectors
complete. Genuine calibration v2 has native completed/exit-zero child evidence;
its consumed terminal handle must not be polled or restarted. Native first-save
reader 83999 is terminal/exit 0 (`1bc2d7`). Archive call 30054 is terminal/exit 0.

Next finish real features and full continuations, measure all required outcomes,
back up new data/checkpoints, compute the fixed scientific contrasts and functional
prediction, and finish portable reconstruction/manuscript release. No preferred
optimizer story, complete paper, cross-seed primary robustness or mediation is
claimed. Do not restart completed primary training/840-task evaluation, touch the
protected helper, resume old controllers or retry any external safety denial.
