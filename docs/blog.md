# Can Muon train a better dense retriever?

> A controlled DenseOn study of AdamW, Muon, and NorMuon, from one-step update geometry to
> five-stage decontaminated BEIR dynamics.

**Study status:** the original DenseOn discovery sweep is complete. Dense-only hybrid-routing
controls, three-seed confirmation, shared-start branches, and the frozen spectrum-versus-basis
intervention are running under the post-hoc scope amendment.

## What changed, and why

This project originally included both DenseOn and LateOn. After the complete discovery sweep and
exploratory mechanism outputs were visible, the project owner asked us to stop new late-interaction
work because it was much slower and less relevant to the intended audience. That is a
**user-directed, post-hoc scope change**, not a scientific exclusion chosen before results.

All existing LateOn artifacts remain available for audit. They are not deleted, treated as a
replication, pooled into uncertainty estimates, or used in the primary conclusion. Every new
training run, evaluation, causal intervention, and paper claim is DenseOn-only. The machine-readable
decision is [dense_scope_amendment.json](../configs/dense_scope_amendment.json).

The new NAACL story is in [the Dense-only paper plan](naacl-dense-paper-plan.md). The
[original two-family plan](naacl-paper-plan.md) remains byte-for-byte frozen so that the earlier
claim protocol can still be audited.

## The question

Muon approximately orthogonalizes momentum updates for hidden weight matrices. NorMuon adds a
row-wise historical normalizer and restores the matrix-level update norm. Those operator properties
are interesting, but they are not a retrieval result. The study asks a narrower question:

> When an Adam-pretrained dense encoder is adapted with a contrastive retrieval loss, can a
> matrix-aware optimizer improve zero-shot rankings, and what optimization process explains any
> improvement?

The evidence is organized to separate four things that are often conflated:

1. the update prescribed by each optimizer at the same weights;
2. the per-query distribution of the immediate functional change;
3. the trajectory created by repeatedly changing both weights and future gradients;
4. the final retrieval ranking on unseen corpora.

## Experimental contract

### Base model

The active model is
[lightonai/DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised), pinned to
revision 0edbd55684eb782bce55ee74c95b25c97cbe7f43. It has completed pretraining but no supervised
retrieval fine-tuning.

### Training data

The public source contains roughly 1.22 million query rows. Only 1,046,024 query IDs have all
required mined-score records. We deterministically select 500,000 groups with proportional
largest-remainder allocation over the seven sources:

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

Each query has one positive and seven seeded random hard negatives. Candidate documents must score
below 0.95 times the positive score; seven are sampled without replacement from the first ten
eligible candidates. Selection and negative sampling use seed 42 plus stable BLAKE2-derived
per-example seeds.

The accepted materialization has:

- dataset fingerprint 8a489098f9729d86;
- canonical row-ledger SHA-256
  735ef35b7195f3dae3172496b5bc534d39f2b7594d216c685eaebb37134fc347;
- fixed training-view fingerprint cc0598ffd4f5454f.

Every run verifies these identities before training. No in-batch or cross-device negatives are used;
the loss always sees an explicit batch-by-eight logit matrix.

### Objective and sequence length

DenseOn uses cosine InfoNCE with temperature 0.02. Query and document truncation limits are both
8,192 tokens, following the DenseOn supervised setting; dynamic padding avoids paying that cost for
short batches. To isolate the optimizer, we omit cross-encoder distillation and the Matryoshka
auxiliary objective.

### Optimizers

| Optimizer | Swept hidden-matrix LR | Auxiliary LR | Main settings |
| --- | --- | ---: | --- |
| AdamW | 1e-6, 3e-6, 1e-5, 3e-5 | same as hidden | beta=(0.9, 0.999), wd=0.01 |
| Muon | 1e-4, 3e-4, 1e-3, 3e-3 | AdamW 3e-6 | momentum=0.95, 5 NS steps, wd=0.01 |
| NorMuon | 1e-4, 3e-4, 1e-3, 3e-3 | AdamW 3e-6 | momentum=0.95, beta2=0.95, 5 NS steps |

Muon and NorMuon act only on the 88 two-dimensional Transformer hidden matrices, covering
110,297,088 parameters. Embeddings, the pooling projection, normalization parameters, and biases
remain on auxiliary AdamW. The all-AdamW baseline uses the swept rate for the same hidden and
projection weights, with the same decay/no-decay routing.

NorMuon is pinned to upstream commit c6989a8354730695d9f5a9faa6c55eeb24865209. Numerical tests
compare its update, momentum buffer, row-wise second moment, and norm restoration with the reference.

### Training schedule

All runs use one epoch, nominal global batch 128, four GPUs, bfloat16 autocast, TF32,
FlashAttention-2, non-reentrant gradient checkpointing, gradient clipping at 1.0, 10% warmup, and
linear decay. Four gradient-accumulation steps turn per-GPU microbatches of eight into the global
batch. The last optimizer step contains the remaining 32 examples; no example is duplicated or
dropped.

Complete resumable checkpoints are retained at 20%, 40%, 60%, 80%, and 100% of the 3,907 optimizer
steps. A checkpoint counts only if the model, optimizer, scheduler, trainer state, four rank-local RNG
archives, schedule, and dataset identities all pass the deep audit.

## Evaluation

Every discovery checkpoint is evaluated on 14 pinned decontaminated BEIR tasks:

ArguAna, ClimateFEVER, DBPedia, FEVER, FiQA, HotpotQA, MS MARCO, NFCorpus, NQ, Quora, SCIDOCS,
SciFact, TREC-COVID, and Touche2020.

The exact inputs are frozen below. Every task uses MTEB's standard `default` subset. Corpus counts
come from the pinned Hub snapshots and also determine longest-processing-time-first scheduling.

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

MS MARCO uses dev; all others use test. The primary score is the unweighted mean task nDCG@10. The
five checkpoints expose the complete observed training curve. We also report four-learning-rate
dispersion, trajectory AUC over 20%–100%, paired task deltas, training throughput, useful wall time,
optimizer-state size, and peak allocated memory.

Best-LR discovery comparisons are exploratory because the same 14 tasks select and evaluate the
configuration. The final comparison therefore uses three new data-order/negative-sampling seeds with
recipes fixed by a query-disjoint validation probe. Its hierarchical bootstrap resamples seeds and
tasks. The headline interval retains the Bonferroni correction over all six comparisons frozen
before the post-hoc Dense-only scope amendment; narrowing the displayed model family does not make
the statistical gate easier.

## Discovery results

<!-- RESULTS:BEGIN -->

All 840 planned task/checkpoint evaluations completed. Scores below are the unweighted mean nDCG@10 across the 14 tasks.

### Final quality and learning-rate robustness

| Family | Optimizer | Same-suite BEIR-best LR | Exploratory BEIR-best final | 4-LR mean | 4-LR median | SD | Range | 4-LR trajectory AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | adamw | 3e-5 | 0.5899 | 0.5816 | 0.5858 | 0.0099 | 0.5650–0.5899 | 0.5779 |
| dense | muon | 3e-4 | 0.5923 | 0.5833 | 0.5901 | 0.0131 | 0.5608–0.5923 | 0.5776 |
| dense | normuon | 3e-4 | 0.5934 | 0.5847 | 0.5910 | 0.0123 | 0.5634–0.5934 | 0.5800 |

- **Dense:** best same-suite BEIR-selected final score is normuon at 3e-4 (0.5934); the highest four-LR mean is normuon (0.5847); the highest mean observed-window AUC is normuon (0.5800). Same-suite BEIR-selected paired muon beats AdamW on 10/14 tasks, mean Δ=+0.0024 (95% CI [+0.0006, +0.0047]); normuon beats AdamW on 11/14 tasks, mean Δ=+0.0036 (95% CI [+0.0015, +0.0059]).

![Dense training dynamics](../reports/dense-discovery/figures/dense-training-dynamics.png)

### Five-checkpoint dynamics for every learning-rate run

Each panel below shows all four LR configurations rather than an optimizer-level average; every curve contains the formal 20%, 40%, 60%, 80%, and 100% checkpoints.

![Dense per-run training dynamics](../reports/dense-discovery/figures/dense-training-dynamics-by-run.png)

### Dynamics after selecting each optimizer by final nDCG@10 on this same BEIR suite

| Family | Optimizer | 20% | 40% | 60% | 80% | 100% | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | adamw | 0.5850 | 0.5892 | 0.5881 | 0.5880 | 0.5899 | 0.5882 |
| dense | muon | 0.5882 | 0.5921 | 0.5912 | 0.5913 | 0.5923 | 0.5912 |
| dense | normuon | 0.5881 | 0.5922 | 0.5925 | 0.5929 | 0.5934 | 0.5921 |

### Paired effects after same-suite BEIR selection

| Family | Comparison | W/T/L | Mean Δ | Paired bootstrap 95% CI | Sign p | Holm p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense | muon − AdamW | 10/0/4 | +0.0024 | [+0.0006, +0.0047] | 0.1796 | 0.438 |
| dense | normuon − AdamW | 11/0/3 | +0.0036 | [+0.0015, +0.0059] | 0.05737 | 0.2295 |

![Dense learning-rate sensitivity](../reports/dense-discovery/figures/dense-lr-sensitivity.png)

### Per-task final scores after same-suite BEIR selection

#### Dense same-suite BEIR-selected discovery task scores


| Task | AdamW | Muon | NorMuon | Muon − AdamW | NorMuon − AdamW |
| --- | --- | ---: | ---: | ---: | ---: |
| ArguAna | 0.5600 | 0.5620 | 0.5617 | +0.0020 | +0.0017 |
| ClimateFEVER | 0.4059 | 0.4104 | 0.4140 | +0.0045 | +0.0082 |
| DBPedia | 0.2776 | 0.2790 | 0.2811 | +0.0015 | +0.0035 |
| FEVER | 0.9205 | 0.9225 | 0.9235 | +0.0020 | +0.0030 |
| FiQA2018 | 0.5421 | 0.5447 | 0.5490 | +0.0026 | +0.0069 |
| HotpotQA | 0.6750 | 0.6787 | 0.6777 | +0.0037 | +0.0027 |
| MSMARCO | 0.5680 | 0.5712 | 0.5707 | +0.0032 | +0.0028 |
| NFCorpus | 0.2874 | 0.2840 | 0.2862 | -0.0034 | -0.0012 |
| NQ | 0.9254 | 0.9396 | 0.9396 | +0.0142 | +0.0142 |
| QuoraRetrieval | 0.9118 | 0.9131 | 0.9132 | +0.0013 | +0.0014 |
| SCIDOCS | 0.1477 | 0.1470 | 0.1484 | -0.0008 | +0.0006 |
| SciFact | 0.8804 | 0.8792 | 0.8792 | -0.0012 | -0.0012 |
| TRECCOVID | 0.8304 | 0.8301 | 0.8297 | -0.0004 | -0.0008 |
| Touche2020 | 0.3259 | 0.3309 | 0.3342 | +0.0050 | +0.0083 |

The best-LR comparisons are selected on this same benchmark suite and should therefore be read as controlled exploratory results, not as an unbiased model-selection estimate. Paired intervals use 20,000 deterministic task-level bootstrap resamples; the sign-test p-value is exact after excluding ties, and Holm p controls the original four-comparison discovery family. BEIR tasks are heterogeneous and not independent draws, so these are descriptive uncertainty summaries rather than population inference. The four-LR mean, spread, and complete per-task rows are included to expose sensitivity rather than reporting only the winning point. Trajectory AUC is the normalized trapezoidal mean nDCG@10 over the observed 20%–100% checkpoint window; it measures early-to-late quality, not time before the first checkpoint.

<!-- RESULTS:END -->

<!-- TASK-DELTA-STABILITY:BEGIN -->

### Exploratory task-effect stability across checkpoints

| Family | Comparison | Stages | Same direction | Pearson r | Spearman rho |
| --- | --- | --- | ---: | ---: | ---: |
| DenseOn | Muon − AdamW | 20%→40% | 10/14 | 0.645 | 0.543 |
| DenseOn | Muon − AdamW | 40%→60% | 10/14 | 0.709 | 0.640 |
| DenseOn | Muon − AdamW | 60%→80% | 11/14 | 0.651 | 0.648 |
| DenseOn | Muon − AdamW | 80%→100% | 12/14 | 0.811 | 0.855 |
| DenseOn | NorMuon − AdamW | 20%→40% | 6/14 | 0.324 | 0.345 |
| DenseOn | NorMuon − AdamW | 40%→60% | 8/14 | 0.463 | 0.451 |
| DenseOn | NorMuon − AdamW | 60%→80% | 11/14 | 0.753 | 0.662 |
| DenseOn | NorMuon − AdamW | 80%→100% | 9/14 | 0.743 | 0.714 |

This DenseOn-only post-hoc diagnostic applies each optimizer's learning-rate run selected by final nDCG@10 on this same BEIR suite at every checkpoint and correlates its 14 paired task deltas against AdamW across adjacent stages. It is a filtered exploratory view of the already audited discovery matrix. It does not alter run selection, the primary aggregate, or the frozen Dense confirmatory family, and it carries no causal interpretation.

<!-- TASK-DELTA-STABILITY:END -->

## Systems results

<!-- SYSTEMS:BEGIN -->

Every run used 4 × NVIDIA L20Z. Values are medians over the four learning-rate configurations for that optimizer and family; CUDA memory is the maximum per rank, not the sum across ranks.

| Family | Optimizer | Median hours | Samples/s | Throughput vs AdamW | Peak allocated GiB | Optimizer state GiB | Checkpoint GiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | adamw | 3.52 | 39.43 | 1.00× | 7.41 | 1.11 | 1.67 |
| dense | muon | 3.71 | 37.41 | 0.95× | 6.98 | 0.70 | 1.26 |
| dense | normuon | 3.77 | 36.86 | 0.93× | 6.97 | 0.70 | 1.26 |

The recorded wall time includes training and five full checkpoint writes. Peak CUDA memory comes from PyTorch allocator counters inside each training process, so the independent utilization guard process is excluded. For checkpoint-resumed runs, throughput is recomputed from the sum of non-overlapping useful training segments rather than Trainer's resume-local runtime; the segment adjustment and original Trainer fields remain in the audit table. Exact per-run measurements are in `reports/dense-discovery/system_metrics.csv`.

<!-- SYSTEMS:END -->

## What the optimization analysis actually says

Spectral flattening is part of Muon's construction; observing it is not a new contribution. The
retrieval-specific evidence begins where the obvious operator property stops.

### 1. Local matched steps lose while the one-seed grid median reverses

At ten common DenseOn states, we apply AdamW, Muon, and NorMuon to the same eight-gradient history.
After matching every hidden tensor to AdamW's Frobenius norm and taking a relative 1e-3 virtual step,
the mean positive-margin gains are:

| Operator | Mean immediate margin gain | Difference from AdamW |
| --- | ---: | ---: |
| AdamW | +0.000865 | — |
| Muon | +0.000562 | -0.000303 |
| NorMuon | +0.000498 | -0.000367 |

Across the four frozen learning-rate sweep points in the single discovery seed, however, median
final BEIR is +0.00435 for Muon and +0.00521 for NorMuon relative to AdamW. The sign reverses from
the matched local intervention to this one-seed grid aggregate. A “better immediate descent
direction” cannot explain that discovery ordering.

### 2. The optimizers enter different directions, not better-aligned gradients

At the same state, the parameter-weighted cosine between AdamW and Muon directions has median 0.470;
Muon and NorMuon remain much closer at 0.972. Median cosine with the common final gradient is 0.401
for AdamW, 0.258 for Muon, and 0.244 for NorMuon.

The AdamW–Muon cosine falls from 0.537 at the pretrained state to about 0.490 along the AdamW
trajectory and 0.463 along the Muon trajectory. This is post-hoc evidence for optimizer-induced state
feedback: committing a Muon-family step changes the state on which future gradients and updates are
computed. It is not yet causal proof that this feedback improves retrieval.

### 3. DenseOn shows tail redistribution, not uniform dominance

The same-state Muon step has a worse mean margin effect, yet improves the fixed p05 margin and p95/p99
loss quantiles at all ten anchors. A symmetric cross-tail check changes the interpretation. On
AdamW's selected worst 5% queries, Muon has a -0.1366 loss-change advantage, but on Muon's own worst
tail it has a +0.0203 disadvantage. Tail-set Jaccard overlap is only 0.279. NorMuon shows the same
pattern with Jaccard 0.270.

Muon therefore moves which queries are fragile; it does not uniformly suppress one shared adverse
tail. This query-level redistribution is specific to the retrieval function and cannot be read from
the optimizer definition alone.

### 4. Flatter spectra alone do not explain the one-seed discovery contrast

For exact selected matrices, the median normalized stable-rank statistic is about 0.0195 for AdamW,
0.5772 for Muon, and 0.4153 for NorMuon. Yet across DenseOn anchors, the magnitude of spectral
flattening has little or no association with tail protection. Representation effective-rank levels
track BEIR over checkpoints, but within-run first differences are weak and inconsistent.

Those negative results matter: “Muon flattens the spectrum” is an operator fingerprint, not a
mechanistic explanation of the observed four-learning-rate-median discovery contrast.

### 5. Learning a new objective can damage pretrained rankings

Query-disjoint validation selects the largest tested Muon-family rate, 3e-3, because it optimizes the
training-style objective. The discovery-BEIR oracle lies at 3e-4. The validation-selected rates lose
0.0315 mean BEIR for Muon and 0.0300 for NorMuon relative to their own discovery oracle, while
roughly doubling score drift from the pretrained ranker.

The emerging trade-off is acquisition versus preservation: strong matrix-aware adaptation can learn
the narrow contrastive objective while eroding zero-shot ranking structure.

<!-- MECHANISM:BEGIN -->

Under the disclosed post-hoc DenseOn scope, the formal mechanism tier evaluates every optimizer transform at the same frozen weights and on the same ordered eight-gradient history. The complete historical source artifacts still pass their content-hash and cardinality audits before the renderer selects the active DenseOn slice: 10 common-state anchors, 270 basis comparisons, 180 exact spectra, 60 bridge checkpoints, and 840 retrieval evaluation units.

### Retrieval time to an AdamW reference

| Family | Optimizer | AdamW reference | LR points reaching | fastest hours | median hours | right-censored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.5858 | 2/4 | 1.407 | 1.416 | 2 |
| DenseOn | Muon | 0.5858 | 3/4 | 0.749 | 1.476 | 1 |
| DenseOn | NorMuon | 0.5858 | 3/4 | 0.756 | 1.507 | 1 |

The reference is the DenseOn median final nDCG@10 of the four AdamW learning-rate points. Passage is observed only at the five saved checkpoints; no interpolation is used, and non-reaching points remain right-censored. Checkpoint time is a step-proportional estimate from audited useful terminal wall time. This one-seed discovery analysis remains exploratory rather than a substitute for the validation-frozen three-seed confirmation.

### Same-state optimizer fingerprints

| Family | Operator | row CV / AdamW | top-1% row energy / AdamW | stable rank / AdamW | spectral norm / AdamW | cosine with AdamW |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | Muon | 2.855 | 1.586 | 28.916 | 0.015 | 0.470 |
| DenseOn | NorMuon | 0.711 | 0.979 | 22.021 | 0.018 | 0.480 |

Each cell is the median over ten frozen DenseOn anchors. Ratios use raw optimizer directions but are scale-invariant except for the explicitly reported spectral-norm ratio; the exact-spectrum intervention uses per-tensor Frobenius-matched directions. Weight decay is excluded from this comparison.

### Function-preserving basis sensitivity

| Family | Operator | mapped cosine | relative direction error | absolute norm-ratio error | predicted-descent error | Q/K spectrum error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.96940 | 0.24738 | 0.00022 | 0.00488 | 0.02540 |
| DenseOn | Muon | 0.99946 | 0.03277 | 0.00007 | 0.00034 | 0.00148 |
| DenseOn | NorMuon | 0.99832 | 0.05793 | 0.00007 | 0.00360 | 0.00946 |

Each row is the median over 90 fixed comparisons: ten common-state anchors, three QKV layers, and three seeded RoPE-commuting rotations. Query and key share each split-half plane rotation, value rows are unchanged, and every direction is inverse-mapped before comparison. The transform preserves attention logits, so this table measures implementation-level coordinate dependence rather than retrieval quality; bfloat16 Newton--Schulz rounding is retained as part of the Muon runtime.

### Exact update spectra

| Family | Operator | stable rank / rank | entropy rank / rank | condition number |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0195 | 0.6984 | 61.26 |
| DenseOn | Muon | 0.5772 | 0.9698 | 16.72 |
| DenseOn | NorMuon | 0.4153 | 0.9603 | 23.82 |

The six matrices were fixed by early/middle/final depth and attention/MLP role before formal spectra existed. Values are medians over 60 exact spectra per optimizer on the active DenseOn anchors.

### Representation and score geometry

| Family | Optimizer | training margin | unseen margin | unseen query rank | pretrained top-1 agreement | mean BEIR nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DenseOn | AdamW | 0.0997 | 0.2499 | 0.6748 | 0.9286 | 0.5858 |
| DenseOn | Muon | 0.1240 | 0.2609 | 0.6770 | 0.9040 | 0.5901 |
| DenseOn | NorMuon | 0.1250 | 0.2611 | 0.6787 | 0.8996 | 0.5910 |

Rows are final-stage medians across all four frozen learning rates, not test-selected winners. Training and unseen probes remain separate; the latter contains 224 fixed examples balanced over all 14 decontaminated tasks.

### Descriptive temporal bridge

| Family | Predictor change | Outcome change | Transitions | Spearman ρ |
| --- | ---: | ---: | ---: | ---: |
| DenseOn | weight-delta row CV | unseen margin | 48 | -0.067 |
| DenseOn | unseen margin | mean BEIR nDCG@10 | 48 | 0.531 |
| DenseOn | unseen query effective rank | mean BEIR nDCG@10 | 48 | -0.027 |
| DenseOn | trailing training loss (post-hoc) | mean BEIR nDCG@10 | 48 | -0.684 |

The first three geometry associations were fixed in the renderer and use within-run first differences across all optimizers. The final training-loss row is an explicitly post-hoc diagnostic. All four are one-seed observational summaries, not a causal mediation analysis. Same-state fingerprints identify what each update rule does; causal claims about accumulated retrieval behavior still require the matched shared-start branches and fixed-state spectral interventions.

<!-- MECHANISM:END -->

## Causal and confirmatory completion design

### Routing-matched AdamW

The Muon-family recipes route hidden matrices to one optimizer and all other parameters to auxiliary
AdamW at 3e-6. Four hybrid AdamW controls use the same routing and two learning rates—one for hidden
matrices and 3e-6 for auxiliary parameters—so that a gain cannot be attributed merely to freezing the
auxiliary learning rate.

### Shared-start accumulation

All nine DenseOn short branches start from the same 60% AdamW checkpoint. They use a fixed
50,000-example subset, three independent order seeds, the three optimizer operators, and a common
global hidden-update-to-weight target of 5e-4. Five checkpoints expose whether a locally weaker
Muon-family direction catches and passes AdamW as feedback accumulates.

The prospective endpoint requires improvement in both query-disjoint loss p95 and unseen margin p05;
a mean-only win is insufficient.

### Three-seed confirmation

Recipes are selected without BEIR using the frozen validation probe, then retrained with seeds
314159, 271828, and 161803. Final checkpoints are evaluated on all 14 decontaminated tasks. The
hierarchical seed-by-task bootstrap reports nominal and Bonferroni familywise intervals for AdamW,
Muon, and NorMuon contrasts. The correction still covers the original six prespecified comparisons,
although only the three DenseOn rows are reported. If a familywise interval crosses zero, the
headline is inconclusive.

### Spectrum-versus-basis transplant

At ten frozen common states, each Muon-family update is decomposed as U diag(s) V-transpose. We
construct Adam basis/Muon spectrum, Muon basis/Adam spectrum, and head/middle/tail band transfers,
then Frobenius-match the interventions before measuring margins and tail losses. This is a post-hoc
causal decomposition whose design was frozen before its outputs existed.

<!-- OUTCOMES:BEGIN -->

### Completion outcome

Pending Dense-only hybrid, shared-start, three-seed, and transplant outputs. The discovery estimates
above remain exploratory regardless of completion. Regeneration reports the separate outcomes from
the validation-frozen recipes and three new seeds.

<!-- OUTCOMES:END -->

## Weight-space audit trail

The complete trajectory tables and figures are retained for reproducibility:

- [matched Muon/NorMuon trajectory figure](../reports/weight-space/optimizer_pair_contrast_trajectory.svg);
- [all-optimizer geometry phase figure](../reports/weight-space/optimizer_geometry_phase.svg);
- [weight-space report and exact definitions](../reports/weight-space/README.md).

NorMuon and Muon end at nearly identical pretrained-reference displacement scales, while NorMuon has
substantially lower row-norm CV and top-1%-row energy concentration. This is useful separation of
their operators, but it is not causal retrieval evidence.

## Engineering and fault recovery

The first formal Muon implementation used PyTorch's native bfloat16 Newton–Schulz addmm path. During
long distributed runs it produced repeated asynchronous CUDA/cuBLAS failures and NVIDIA Xid 13/43
events across multiple physical GPUs. ECC counters stayed clean, and controlled replays moved the
failure across devices, which argued against one bad card.

Two alternatives completed synchronized stress tests on all eight devices: FP32/TF32 addmm and an
algebraically equivalent bfloat16 decomposition using matrix multiplication plus elementwise
combinations. We selected the latter to preserve Muon's bfloat16 representation, polynomial,
coefficients, momentum, learning-rate adjustment, and update norm. Only the operation decomposition
changed. It is recorded as unfused-bfloat16-v1.

All native-addmm Muon histories were quarantined. Every accepted Muon and NorMuon run restarted from
the pinned base state and records the new implementation ID. No pre- and post-mitigation history is
mixed. Resumption uses only deep-validated checkpoints, and useful-time ledgers exclude repeated work,
initialization, pauses, and failed segments.

This engineering result is practical rather than algorithmic: changing algebraically equivalent
bfloat16 operations can change rounding, so the accepted implementation is fully pinned and covered
by numerical and distributed stress tests.

## Reproduce the DenseOn study

The repository is currently private during completion. Credentials are never stored in source.

### Install

~~~bash
git clone https://github.com/qcznlp/embedding-optimizer-study.git
cd embedding-optimizer-study
uv sync --extra dev --extra eval --extra analysis
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
source .venv/bin/activate
embed-optim-verify-runtime --spec configs/formal_runtime.json
~~~

### Prepare and audit the shared 500K data

~~~bash
embed-optim-prepare
embed-optim-prepare --audit-only
~~~

### Run the 12-run DenseOn discovery sweep

~~~bash
embed-optim-matrix \
  --matrix configs/experiment.yaml \
  --families dense \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
~~~

The historical matrix still contains archival LateOn definitions. The explicit family flag is
mandatory for the active Dense-only study.

### Evaluate all five discovery checkpoints

~~~bash
embed-optim-evaluate \
  --matrix configs/experiment.yaml \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --stages 1 2 3 4 5 \
  --gpus-a 0,1,2,3,4,5,6,7 \
  --gpus-b 4,5,6,7
~~~

### Run the frozen Dense completion queue

After generating the confirmatory and short-branch matrices, the two resumable queues can run on
independent four-GPU pools:

~~~bash
embed-optim-family-training-queue --pool a --gpus 0,1,2,3 --port 30110
embed-optim-family-training-queue --pool b --gpus 4,5,6,7 --port 30120
~~~

After both ledgers are complete, the Dense completion pipeline audits training, evaluates hybrid and
confirmatory checkpoints, runs the short-branch probes, and performs the spectral transplant:

~~~bash
embed-optim-dense-completion \
  --scope-amendment configs/dense_scope_amendment.json \
  --training-plan configs/dense_training_queue.json \
  --training-ledgers \
    logs/dense-only-runtime/training-queue-a.json \
    logs/dense-only-runtime/training-queue-b.json \
  --workdir "$PWD" \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-b 4,5,6,7 \
  --include-validation \
  --resume
~~~

All commands are resumable and content-addressed. Do not edit the frozen queue or scope files during
an active run. Each pool is protected by an exclusive lease and writes `complete=false` before it
waits or resumes. Existing terminal runs are skipped only after the same uncached deep checkpoint
payload audit used by the downstream watcher. Failed completed artifacts are preserved under
`.invalid-completed-runs/` before a clean rerun, while a 24-hour process-group watchdog prevents a
stalled matrix command from holding a pool indefinitely.

If completion is launched before the queues exit, add their exact process IDs as
`--wait-pids POOL_A_PID POOL_B_PID`. That wait does not replace the ledger gate: completion requires
exactly two unique, complete Dense-only ledgers for pools `a` and `b`, verifies their nine jobs and
shared frozen-plan hash, and hashes both ledger contents. To recover after interruption, finish or
repair both queues first and rerun this same command with `--resume`. Resume reconstructs the current
plan, scope, ledger-content, source, and step-command contract and reruns orchestration from step 1;
it never accepts a previously completed prefix. Individual evaluation units are reused only when
their own content-addressed checkpoint/runtime/result audits pass. This full rerun also upgrades
legacy completion ledgers instead of grandfathering them into the stricter contract.

### Render the Dense paper and blog

After the Dense completion ledger passes, use the canonical resume-safe finalizer. It regenerates all
scoped blog blocks and reports before running the paper, test, build, and distribution gates:

~~~bash
embed-optim-dense-finalize \
  --scope-amendment configs/dense_scope_amendment.json \
  --completion-ledger logs/dense-completion-pipeline/pipeline-ledger.json \
  --workdir "$PWD" \
  --resume
~~~

Use `--wait-pid COMPLETION_PID` only when the exact completion process is still running. After a
failure, rerun the finalizer with `--resume` only once the completion ledger is clean and complete;
even an already-complete finalization ledger is accepted only after its current completion-ledger,
training-plan, pool-ledger, scope, and step-contract provenance is revalidated, after which the full
canonical finalization orchestration is rerun from the beginning.
If the completion ledger predates those bindings, upgrade it with completion `--resume` before
retrying the finalizer.

Its reporting sequence begins with the scoped discovery aggregate and retrieval dynamics; these two
steps regenerate the `RESULTS`, `SYSTEMS`, and `TASK-DELTA-STABILITY` blocks that the later renderers
consume and audit:

~~~bash
embed-optim-aggregate \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --strict

embed-optim-summarize-retrieval-dynamics \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

embed-optim-render-mechanism-report \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

embed-optim-render-outcome-report \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

embed-optim-render-paper-results \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

embed-optim-audit-paper \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --strict

make -C paper
uv build
embed-optim-audit-distribution
~~~

## Limitations

- The scope reduction occurred after discovery and exploratory mechanism results were visible.
- Discovery best-LR estimates use the evaluation suite for selection and are not confirmatory.
- The formal confirmation covers one DenseOn base checkpoint, one epoch, one dataset construction,
  and three new sampling seeds; it is not evidence for all dense retrievers.
- The 14 BEIR tasks are heterogeneous benchmarks rather than independent population draws.
- Common-state virtual steps identify immediate direction effects, not full optimization paths.
- Weight-space and representation correlations are descriptive unless an intervention changes the
  retrieval function as predicted.
- The five retained checkpoints give a coarse trajectory, not a per-step path length.
- Runtime conclusions are specific to the pinned hardware/software stack.

## Current defensible conclusion

The discovery study supports a narrower descriptive claim: when the same BEIR suite selects one
learning rate per optimizer, Muon and NorMuon slightly exceed the likewise BEIR-selected AdamW point
estimate, but not because they take a better average one-step descent direction. This is not the
validation-selected comparison. Query-disjoint validation selects 3e-3 for both Muon variants, and
those discovery-BEIR scores are below AdamW. Together, the results suggest that Muon-family updates
move into a distinct, state-dependent trajectory and redistribute which queries are fragile, while
stronger adaptation can overfit the training-style objective and damage pretrained rankings.

Whether that trajectory explanation survives common-start accumulation and what final quality
contrast appears under validation-frozen recipes and new seeds remain gated on the running
Dense-only experiments. The final blog and paper will state a stronger claim only if those
prospective gates pass.
