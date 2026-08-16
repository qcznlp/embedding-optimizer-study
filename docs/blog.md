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

## Limitations

- This is one epoch on one pretrained backbone family; it does not establish a universal optimizer
  ranking.
- Muon and NorMuon necessarily use AdamW for non-matrix parameters, so the comparison is between
  practical optimizer recipes rather than mathematically pure single-optimizer systems.
- Four learning rates improve robustness but do not guarantee that every optimizer's global optimum is
  inside the sweep.
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
