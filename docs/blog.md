# Can Muon train embedding models better than AdamW?

> A controlled comparison on DenseOn and LateOn, with identical data, five-point training dynamics,
> and decontaminated BEIR evaluation.

**Experiment status:** training matrix in progress. This document already records the frozen protocol;
the results sections are populated only from the checked-in aggregation artifacts after coverage reaches
1,680/1,680.

## Why this comparison

Most modern embedding models are still fine-tuned with AdamW. [Muon](https://kellerjordan.github.io/posts/muon/)
instead orthogonalizes updates for hidden weight matrices and often tolerates much larger learning
rates. NorMuon adds a row-wise second
moment normalizer to Muon's orthogonalized update while preserving its overall update norm. Language
model results are promising, but they do not answer whether the optimizers behave well under retrieval
losses, where temperatures are low and representation geometry matters directly.

This study asks three questions:

1. Do Muon or NorMuon reach useful retrieval quality earlier in training?
2. Are they competitive with a well-tuned AdamW after one epoch?
3. Is the conclusion stable across dense and late-interaction architectures and across learning rates?

## Frozen protocol

We start from
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised) and
[LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised), the pretrained but not
supervised-fine-tuned checkpoints from the
[DenseOn/LateOn work](https://arxiv.org/abs/2607.27178). Every run sees the same ordered 500,000 query
groups for one epoch at global batch 128.

The DenseOn revision is `0edbd55684eb782bce55ee74c95b25c97cbe7f43`; the LateOn revision is
`1047071849a708b9b3ee4dccdc60186c185224a7`.

Each group contains one positive and seven hard negatives. We scan the source model's ranked candidates,
retain candidates whose score is below `0.95 × positive_score`, take the first ten qualifying candidates,
and sample seven without replacement. All query sampling and negative choices derive their RNG seeds
from seed 42 and stable BLAKE2 hashes. We intentionally disable in-batch and cross-device negatives;
the logit tensor is exactly `[batch, 8]` for both architectures.

DenseOn uses cosine InfoNCE at temperature 0.02. LateOn uses L2-normalized token vectors,
MeanMaxSim, and temperature 0.001 with query expansion disabled. Both query and document truncation
limits are 8,192 tokens, matching the paper's supervised fine-tuning context setting. Dynamic padding
means short batches do not pay the full 8,192-token cost.

This is deliberately an optimizer-isolation objective rather than a full reproduction of the paper's
supervised stage. In addition to removing in-batch negatives as specified for this study, we do not use
the paper's cross-encoder knowledge distillation or DenseOn Matryoshka auxiliary loss. Both families
therefore optimize only their explicit eight-document group contrastive loss.

### Dataset allocation

The public source contains 1.22M query rows, but not every row has a mined score record. We allocate
proportionally over the 1,046,024 query IDs present in both tables:

| Source | Scorable queries | Selected |
| --- | ---: | ---: |
| FiQA | 5,500 | 2,629 |
| HotpotQA | 85,000 | 40,630 |
| MS MARCO | 502,939 | 240,405 |
| NQ | 152,145 | 72,725 |
| FEVER | 109,810 | 52,489 |
| SQuADv2 | 130,217 | 62,244 |
| TriviaQA | 60,413 | 28,878 |
| **Total** | **1,046,024** | **500,000** |

The materialized dataset records source revisions, quotas, every selected document ID, the Hugging Face
Dataset fingerprint, and a canonical row-manifest SHA-256. The dataset used for this study has on-disk
fingerprint `8a489098f9729d86` and row-manifest SHA-256
`735ef35b7195f3dae3172496b5bc534d39f2b7594d216c685eaebb37134fc347`. Loading that artifact and
selecting the fixed nine text fields plus the length field produces training-view fingerprint
`cc0598ffd4f5454f`; every run records this second fingerprint so the final audit can prove that both
model families saw the identical view.

### Optimizer sweep

| Optimizer | Hidden-matrix learning rates | Auxiliary learning rate | Other settings |
| --- | --- | ---: | --- |
| AdamW | 1e-6, 3e-6, 1e-5, 3e-5 | same as hidden | β=(0.9, 0.999), wd=0.01 |
| Muon | 1e-4, 3e-4, 1e-3, 3e-3 | 3e-6 AdamW | momentum=0.95, 5 NS steps, wd=0.01 |
| NorMuon | 1e-4, 3e-4, 1e-3, 3e-3 | 3e-6 AdamW | momentum=0.95, β₂=0.95, 5 NS steps, wd=0.01 |

For Muon-family runs, only 2-D transformer hidden matrices use the matrix optimizer. Embeddings,
LateOn projection layers, norms, and biases use AdamW. This avoids applying orthogonalized updates to
parameters for which Muon is not designed. The auxiliary AdamW uses β=(0.9, 0.999), ε=1e-8, and the
same decay/no-decay routing as the all-AdamW baseline. NorMuon is pinned to official implementation
commit `c6989a8354730695d9f5a9faa6c55eeb24865209`; a numerical regression test checks the update,
momentum buffer, and row-wise second moment against that reference.

The routing was enumerated from the instantiated checkpoints before training:

| Family | Muon/NorMuon hidden matrices | AdamW, decayed auxiliary | AdamW, no decay |
| --- | ---: | ---: | ---: |
| DenseOn | 110,297,088 params / 88 tensors | 38,682,624 / 1 | 34,560 / 45 |
| LateOn | 110,297,088 params / 88 tensors | 43,501,056 / 6 | 34,560 / 45 |

For the AdamW baselines, the hidden and decayed-auxiliary columns instead form one swept-LR AdamW
group; the no-decay column uses that same learning rate with zero weight decay. Muon calls PyTorch's
native functional update with Nesterov momentum, five Newton–Schulz steps, coefficients
`(3.4445, -4.7750, 2.0315)`, ε=1e-7, and `adjust_lr_fn="original"`. NorMuon uses the same
orthogonalization, then applies the official β₂=0.95 row-wise second-moment normalization, restores
the pre-normalization Frobenius norm, and applies the matrix-aspect-ratio correction.

We use linear decay with a 10% warmup, bfloat16 autocast, TF32, FlashAttention-2, non-reentrant
gradient checkpointing, and gradient clipping at 1.0. Each run uses four GPUs, a per-GPU micro-batch
of 8, and four gradient-accumulation steps, yielding the shared global batch of 128. Each run saves
complete checkpoints at 20%, 40%, 60%, 80%, and 100% of its realized optimizer steps.
The 500,000 examples form 15,625 four-GPU microbatches, so steps 1–3,906 contain 128 examples and the
final partial accumulation contains 32; no training examples are duplicated or dropped.

## Evaluation

Every checkpoint is evaluated on the 14 decontaminated [BEIR](https://arxiv.org/abs/2104.08663)
datasets released with the
[DenseOn/LateOn blog](https://huggingface.co/blog/lightonai/denseon-lateon): ArguAna,
ClimateFEVER, DBPedia, FEVER, FiQA, HotpotQA, MS MARCO, NFCorpus, NQ, Quora, SCIDOCS, SciFact,
TREC-COVID, and Touche2020. Dataset commits are pinned in source. The primary aggregate is the
unweighted mean of each task's main nDCG@10 score. Final reporting also includes the normalized
trapezoidal area under the five-checkpoint quality curve over the observed 20%–100% window,
four-learning-rate dispersion, best-configuration paired task wins, and measured training throughput
relative to AdamW. For each best-config Muon/AdamW and NorMuon/AdamW comparison, we report the mean
task delta with a deterministic seed-42, 20,000-resample paired task bootstrap 95% interval and an
exact two-sided sign-test p-value after excluding ties. These are descriptive summaries: the optimizer LR
is selected on the same suite, and the 14 heterogeneous BEIR tasks are not independent draws. We
report both the raw sign-test p-value and its Holm correction across the four family-by-optimizer
comparisons.

The exact evaluation inputs are frozen below. MTEB's standard `default` subset is used for every
task. MS MARCO is scored on `dev`; the other thirteen tasks are scored on `test`. Corpus counts were
read from the pinned Hub snapshots and are also used for longest-processing-time-first scheduling.

| Task | Hugging Face dataset | Revision | Split | Corpus rows |
| --- | --- | --- | --- | ---: |
| ArguAna | `lightonai/arguana-decontaminated` | `c19c66cb43fb9b090cc55e81c10d9b5dc70b47a7` | test | 8,546 |
| ClimateFEVER | `lightonai/climate-fever-decontaminated` | `1e73a88ba467a00e22ae814873edc3a9bb63b441` | test | 5,117,453 |
| DBPedia | `lightonai/dbpedia-entity-decontaminated` | `7689f39462841132f3345798c946481418a8b77c` | test | 1,678,309 |
| FEVER | `lightonai/fever-decontaminated` | `c39d8922d4bd04bc690a4331800018c1c44d41bd` | test | 5,117,452 |
| FiQA2018 | `lightonai/fiqa-decontaminated` | `6d053f042b0a58763f0463d4591521725c8eb1b9` | test | 47,617 |
| HotpotQA | `lightonai/hotpotqa-decontaminated` | `2a3899c49545c56f84d5e884a1723c34ed96ae61` | test | 2,314,813 |
| MSMARCO | `lightonai/msmarco-decontaminated` | `7e98709a5db3f95bbd50a7b9b53f3a2c5d69f837` | dev | 4,036,967 |
| NFCorpus | `lightonai/nfcorpus-decontaminated` | `de914702862784c9d5c937cf4736bf37bc7bbac9` | test | 912 |
| NQ | `lightonai/nq-decontaminated` | `8d4418d0bab92c5887e0f330fe9bea1e692e173f` | test | 305,674 |
| QuoraRetrieval | `lightonai/quora-decontaminated` | `a303966cc5dc0dcfb77761202d10e02c2fc67be2` | test | 413,157 |
| SCIDOCS | `lightonai/scidocs-decontaminated` | `a5f62cf5006386ed1f069b79c56fbbe18e4e778a` | test | 5,833 |
| SciFact | `lightonai/scifact-decontaminated` | `0729fa34af49875724d18ace64ce07f3e1dc0587` | test | 858 |
| TRECCOVID | `lightonai/trec-covid-decontaminated` | `9e28c1e95c3e04a8f12ea2053822d80312f15794` | test | 99,522 |
| Touche2020 | `lightonai/webis-touche2020-decontaminated` | `84c6c1ff39a87ee1e1d4356fc6f43df6d49431b3` | test | 378,223 |
| **Total** |  |  |  | **19,525,336** |

Dense retrieval is evaluated with [MTEB](https://arxiv.org/abs/2210.07316) exact retrieval. Late
interaction is encoded with [PyLate](https://arxiv.org/abs/2508.03555) and searched with
[FastPLAID](https://github.com/lightonai/fast-plaid) using 4-bit residuals, 8 IVF probes, 8,192 full
scores, and index seed 42.
The training scorer uses the fused Late Interaction Kernels implementation. The pinned corpus counts
sum to 19,525,336 documents; evaluation schedules larger corpora first across the available workers
and resumes only the missing split/subset pairs from validated MTEB result files. A launch preflight
first reconstructs the exact shared training-data view and deep-validates every selected model,
optimizer, scheduler, and per-rank RNG payload before any formal evaluation GPU work. It then uses one
Python runtime for both model families and requires its core model-library versions to match the
versions recorded during training. An immutable runtime manifest also records PyLate, FastPLAID,
Late Interaction Kernels, MTEB, and FlashAttention. It additionally records SHA-256 identities for all
eight evaluation/aggregation source files and verifies that the worker interpreter imports those same
package sources. Final aggregation recomputes the hashes and rejects results with a different
checkpoint identity, dataset revision, split/subset, scoring field, MTEB version, or runtime
provenance.
Each task result and its model metadata are atomically replaced, concurrent DenseOn workers merge the
shared MTEB run-settings sidecar under a cross-process lock, and only the main LateOn rank writes cache
metadata. The final audit also checks the run-settings field layout against the recorded MTEB version,
covering the singular 2.18 and plural 2.19+ schemas without accepting a hybrid artifact.
After FastPLAID consumes a task's corpus multivectors, the shared caller-visible embedding list is
cleared before search and the temporary on-disk index is removed after the task (also on failure).
This prevents host-memory and disk usage from accumulating across the 840 LateOn evaluations.
For the training side, the strict gate parses every safetensors tensor extent, loads each optimizer and
scheduler state with PyTorch's restricted weights-only loader, and verifies the CRC of every rank RNG
archive; a merely non-empty or partially written checkpoint is not counted as complete.

## Results

<!-- RESULTS:BEGIN -->

Results will be inserted here after `reports/coverage.json` confirms all 1,680 task/checkpoint pairs.

<!-- RESULTS:END -->

## Systems observations

<!-- SYSTEMS:BEGIN -->

Final wall-clock throughput, peak memory, optimizer-state size, and checkpoint size will be reported
from the completed W&B runs and local Trainer states.

<!-- SYSTEMS:END -->

Infrastructure note: the first DenseOn Muon run encountered two pre-checkpoint
`CUBLAS_STATUS_EXECUTION_FAILED` errors in PyTorch's native bfloat16 Newton–Schulz `addmm`, both on
physical GPU 3 (at optimizer steps 52 and 388). ECC counters remained zero, but moving that device
from the DenseOn pool to the concurrent AdamW LateOn pool let the replacement run pass both observed
positions without changing the Muon implementation, precision, hyperparameters, model state, or data
order. Those failed attempts occurred before checkpoint 782 and were discarded; the accepted run
starts from the original base model and seed.

The replacement run later encountered a third transient failure at step 2,354, nine steps after its
validated checkpoint at step 2,345. This time rank 2 (physical GPU 2 in the swapped pool) reported an
asynchronous CUDA launch failure through the NCCL watchdog, without an application traceback that
identified the originating kernel. The recovery supervisor resumed DenseOn from checkpoint 2,345 and
the concurrent LateOn run from checkpoint 3,126; both passed their interruption points.

After DenseOn produced and passed a deep audit of checkpoint 3,126, a fourth transient asynchronous
launch failure occurred at step 3,336 on rank 0 (physical GPU 0). It surfaced from the native Muon
momentum-buffer `lerp_` call, although CUDA's asynchronous reporting means that frame does not prove
which earlier kernel originated the fault. The next retry encountered a fifth asynchronous launch
failure at step 3,180 on rank 1 (physical GPU 1), reported by the NCCL watchdog. The five incidents
therefore span physical GPUs 3, 2, 0, and 1, so the evidence no longer supports a single-device
explanation and instead points to the native bfloat16 Muon/CUDA path under this runtime. A subsequent
resume passed both interruption points without changing optimizer semantics, but later encountered a
sixth asynchronous launch failure at step 3,829 on rank 2 (physical GPU 2), again reported by the
NCCL watchdog. The supervisor restarted both concurrent runs from their independently validated
checkpoint 3,126 states. That retry reached step 3,375 before a seventh failure on the same rank and
physical GPU. In this event the application traceback directly reported
`CUBLAS_STATUS_EXECUTION_FAILED` from the native bfloat16 Newton–Schulz `torch.addmm`, alongside the
NCCL watchdog and same-time Xid 13 followed by Xid 43. The supervisor again resumed the independently
validated checkpoint 3,126 states without changing the optimizer or runtime. The following resume
passed every prior interruption point and completed step 3,907. A full post-run audit loaded all five
model/optimizer/scheduler checkpoints, checked all four rank RNG states at each stage, validated the
final model and Trainer state, and reported five of five checkpoints with zero errors.

The kernel log independently correlates every interruption with an NVIDIA driver Xid on the same
physical device and at the same wall-clock time:

| Incident | Optimizer step | Physical GPU (PCI bus) | User-space reporting point | Driver Xid |
| ---: | ---: | --- | --- | --- |
| 1 | 52 | 3 (`c6:00`) | Newton–Schulz `addmm` / cuBLAS | 43 |
| 2 | 388 | 3 (`c6:00`) | Newton–Schulz `addmm` / cuBLAS | 13, then 43 |
| 3 | 2,354 | 2 (`a2:00`) | NCCL watchdog | 43 |
| 4 | 3,336 | 0 (`08:00`) | Muon momentum-buffer `lerp_` | 43 |
| 5 | 3,180 | 1 (`7e:00`) | NCCL watchdog | 13, then 43 |
| 6 | 3,829 | 2 (`a2:00`) | NCCL watchdog | 13, then 43 |
| 7 | 3,375 | 2 (`a2:00`) | Newton–Schulz `addmm` / cuBLAS and NCCL watchdog | 13, then 43 |
| 8 | 2,717 | 2 (`a2:00`) | NCCL watchdog | 43 |
| 9 | 303 | 4 (`0001:09:00`) | NCCL watchdog | 13, then 43 |
| 10 | 3,345 | 1 (`7e:00`) | Newton–Schulz `addmm` / cuBLAS | 43 |

NVIDIA classifies [Xid 13](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html)
as a graphics-engine exception: typically an application/CUDA fault such as an out-of-bounds access
or illegal instruction, while rare driver or hardware causes remain possible. Xid 43 records the
resulting software-induced channel termination and says the GPU remains healthy. The repeated pattern
across five physical GPUs, identical Xid-13 exception registers on four devices, stable concurrent
AdamW execution before fail-fast termination, and zero volatile corrected or uncorrected ECC counts
therefore favor an application/kernel-path explanation over a single-card hardware fault. They still
do not identify which asynchronously executed operation originated the exception.

The supervisor resumes both concurrent runs only from their last deep-validated checkpoints.
DenseOn's reported throughput sums its non-overlapping accepted segments through step 3,126 and the
final segment after that checkpoint; LateOn applies the same rule. Repeated post-checkpoint steps,
failed pre-checkpoint attempts, restart initialization, and downtime are retained as infrastructure
evidence but excluded from optimizer-throughput comparisons. After retaining the Python/CUDA stacks,
the recovery supervisor and its newly spawned workers disable multi-gigabyte ELF core dumps; this
changes neither training nor optimizer execution. This accounting gives the completed DenseOn Muon
`1e-4` run 3.512 hours of useful wall time; its content-addressed canonical W&B history contains 392
strictly ordered rows through step 3,907. The completed LateOn AdamW `3e-6` run likewise has 8.291
hours of useful wall time and 16.751 examples/s after excluding duplicated recovery work; its deep
audit verified all five checkpoints, and its content-addressed canonical W&B history also contains
392 rows through step 3,907. After a deep audit of DenseOn Muon `3e-4` checkpoint 1,563, a
user-directed priority handoff moved Muon ahead of the remaining AdamW runs without changing any run
configuration or the 24-run contract. The handoff also replaced the old cross-pool fail-fast process
with failure-isolated GPU pools, resumed DenseOn from the audited checkpoint, and started the formal
LateOn Muon `1e-4` run. That LateOn run passed optimizer steps 52 and 388—the two early locations at
which DenseOn had previously reported direct cuBLAS failures—and continued through step 500 without
a CUDA, NCCL, or Xid event. Meanwhile, DenseOn Muon `3e-4` deep-validated checkpoints 782, 1,563,
and 2,345 before an eighth transient failure at step 2,717. Rank 2 (physical GPU 2) reported an
asynchronous launch failure through the NCCL watchdog, and the driver recorded a same-device Xid 43;
there was no direct originating-kernel traceback and all volatile ECC counters remained zero. The
independent LateOn process continued training, demonstrating that the pool isolation works as
intended. This event also exposed a scheduler recovery bug: a failed configuration advanced to the
next configuration instead of immediately resuming its latest checkpoint. The matrix runner now
requeues a failed configuration at the front of its family queue, with an integration test that
verifies the retry occurs before later queued work.

Before that patched scheduler could be activated at a safe LateOn checkpoint boundary, the old
process advanced DenseOn to Muon `1e-3`; it failed at step 303 on rank 3 (physical GPU 4) with another
asynchronous NCCL-watchdog report. The driver logged Xid 13 and 43 on the same device, including the
same exception-register values previously observed on physical GPU 2, while ECC remained zero and
LateOn continued independently.

At LateOn step 782, the first formal checkpoint passed a deep payload audit: the model, mixed
Muon/AdamW optimizer state, scheduler, Trainer state, and all four rank RNG states loaded without an
error. The optimizer payload assigns 88 hidden matrices to Muon and 51 embedding, projection, norm,
and bias tensors to auxiliary AdamW; all state tensors are finite. A tensor-by-tensor comparison
against the pinned unsupervised base revision found that all 139 trainable tensors changed by this
checkpoint, including every backbone tensor and all three projection modules. A controlled restart
then activated the patched scheduler. It selected the two intended
unfinished runs, resumed DenseOn Muon `3e-4` from checkpoint 2,345 and LateOn Muon `1e-4` from
checkpoint 782, and the latter continued through step 789. This is an end-to-end PyLate checkpoint
recovery test rather than only a file-presence check. The same LateOn process subsequently reached
checkpoint 1,563 without a CUDA, NCCL, or Xid event; its model, optimizer, scheduler, runtime
arguments, and four RNG states passed the stricter deep audit, all 139 trainable tensors changed
again relative to checkpoint 782, and training continued beyond that checkpoint. The 88 Muon
momentum buffers and all 51 auxiliary AdamW states were finite; every auxiliary AdamW step counter
equaled 1,563, and the three saved group learning rates matched the linear scheduler exactly. The
relative L2 change from checkpoint 782 to 1,563 was 0.00383 for Muon-routed hidden matrices,
0.000566 for decayed auxiliary tensors, and 0.000136 for no-decay tensors. The stricter audit now
also verifies group algorithms, hyperparameters, scheduled learning rates, state fields and shapes,
AdamW step counters, scheduler base/last learning rates, scheduler step count, finite model tensors,
and a distinct model payload at each checkpoint. All 40 checkpoints from the eight completed runs
passed this expanded audit. The only traceback in the LateOn log is the expected `SIGTERM` record
from the controlled restart at checkpoint 782, not a PyLate, CUDA, or Muon failure. The resumed DenseOn run
subsequently reached checkpoint 3,126, whose model, optimizer, scheduler, and four rank RNG payloads
also passed the deep audit. Timing accounting retains the non-overlapping segment through step 782
and excludes duplicated post-checkpoint steps and restart initialization. That DenseOn attempt later
failed at step 3,345:
rank 1 (physical GPU 1) directly reported `CUBLAS_STATUS_EXECUTION_FAILED` from the native
bfloat16 Newton–Schulz `torch.addmm`, and the driver recorded a same-device Xid 43. The patched
scheduler immediately selected the same configuration and resumed checkpoint 3,126, while the
isolated LateOn Muon process continued uninterrupted. The accepted-time adjustment retains only the
completed 2,346–3,126 interval from the failed attempt; duplicated steps after 3,126 are excluded.
That retry completed step 3,907. All five checkpoints passed the full protocol and payload audit;
the non-overlapping useful wall time is 3.525 hours, and the canonical 392-row W&B history is stored
under content hash `5fe30f45c960`. Canonical histories exclude Trainer's resume-local terminal
runtime/loss summaries and instead publish useful wall time and throughput reconstructed from the
audited non-overlapping timing ledger.

To prioritize the higher-risk LateOn/Muon path, a subsequent controlled handoff occurred only after
LateOn Muon `1e-4` checkpoint 1,563 and DenseOn Muon `1e-3` checkpoint 782 had passed the expanded
deep audit. Work performed after those durable boundaries was deliberately discarded. The two
accepted LateOn segments through step 1,563 total 11,877.64 seconds; restart gaps and repeated steps
are excluded. Both four-GPU pools then switched to LateOn: `1e-4` resumed from step 1,563 on its
original pool while `3e-4` started from the pinned base on the other pool. The DenseOn `1e-3`
checkpoint remains resumable, and its atomic timing ledger retains the accepted `0–782` segment.

Subsequent attempts commit the maximum four-rank duration automatically in an atomic timing ledger after each
durable checkpoint. The final audit requires contiguous accepted step ranges, finite positive
durations, a matching segment sum, and the expected terminal step. Historical segments retained
before this mechanism was enabled remain backed by W&B `startedAt` values and checkpoint Trainer-state
mtimes; this also restores DenseOn Muon `3e-4` steps 1,564–2,345 that would otherwise have been
omitted from its eventual throughput denominator.

## Limitations

- This is one epoch on one pretrained backbone family; it does not establish a universal optimizer
  ranking.
- Muon and NorMuon necessarily use AdamW for non-matrix parameters, so the comparison is between
  practical optimizer recipes rather than mathematically pure single-optimizer systems.
- Four learning rates improve robustness but do not guarantee that every optimizer's global optimum is
  inside the sweep.
- PyLate 1.6 uses Late Interaction Kernels 0.4.5's reduced-precision KD backward. An adversarial
  masking test returned the exact expected score and zero gradients for masked query/document tokens,
  but random bfloat16 near-ties differed from eager PyTorch by up to 0.00252 in score and could choose
  a different sparse MaxSim argmax. Every LateOn run uses the same pinned kernel and hardware, so this
  is controlled within the study, but cross-backend bitwise equivalence is not claimed.
- Every configuration uses the same single training/data seed (42). This controls the comparison but
  does not quantify variance across independent seeds.
- Seed 42 fixes data selection, negative choices, Trainer sampling, and model RNG state, but the
  high-throughput FlashAttention/TF32 CUDA path is not configured for bitwise deterministic replay.
- No in-batch negatives makes the groups controlled and memory predictable, but the absolute scores are
  not directly comparable to recipes that use large cross-device negative pools.
- Omitting the paper's knowledge-distillation and DenseOn Matryoshka auxiliary objectives isolates the
  optimizer comparison, but it also means these scores are not a reproduction of the released models'
  full supervised training recipe.
- Approximate PLAID retrieval can introduce a small difference from exhaustive late-interaction scoring;
  index settings are held constant for every checkpoint.

## Reproduction and audit trail

All code, configs, pinned revisions, tests, and commands are in the
[project repository](https://github.com/qcznlp/embedding-optimizer-study). The repository is currently
private at the user's request and is structured for a later public release. The
[W&B project](https://wandb.ai/stevezenguom/embedding-optimizer-study) contains training curves and
resolved run configurations. Local evaluation artifacts are reduced by `embed-optim-aggregate`, which
also checks expected matrix coverage before the results section is generated.

Checkpoint resumes can make an SDK run contain repeated optimizer steps even when the local Trainer
state is correct. For the final dashboard, `embed-optim-sync-wandb` therefore publishes one immutable,
content-addressed canonical history from every completed Trainer state. Raw resume segments are kept
for system telemetry; no source run is deleted.

## References

- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101),
  2017.
- Jordan et al., [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/),
  2024.
- Sourty et al., [DenseOn with the LateOn](https://arxiv.org/abs/2607.27178), 2026.
- Li et al., [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/abs/2510.05491),
  2025.
- Thakur et al., [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval
  Models](https://arxiv.org/abs/2104.08663), 2021.
- Muennighoff et al., [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316),
  2022.
- Chaffin and Sourty,
  [PyLate: Flexible Training and Retrieval for Late Interaction Models](https://arxiv.org/abs/2508.03555),
  2025.
- LightOn,
  [FastPLAID: High-Performance Engine for Multi-Vector Search](https://github.com/lightonai/fast-plaid),
  2025.
- LightOn, [DenseOn and LateOn release blog](https://huggingface.co/blog/lightonai/denseon-lateon),
  2026.
- NVIDIA, [Analyzing Xid Errors with the Xid Catalog](https://docs.nvidia.com/deploy/xid-errors/analyzing-xid-catalog.html),
  2026.
