# From Update Geometry to Retrieval Geometry

## Executive narrative

The paper's thesis is not that one optimizer always wins. It is that an optimizer is an inductive
bias on the retrieval function: its matrix transform determines which directions and neurons move,
those movements reshape single-vector or token-level representations, and the resulting score
geometry controls retrieval efficiency and out-of-domain ranking quality.

The story should unfold in five beats:

1. **Empirical tension.** Muon is reported to improve language-model pretraining efficiency, but an
   embedding model is an Adam-pretrained model being adapted with a contrastive ranking objective.
   It is not known whether Muon's pretraining advantage transfers to this regime.
2. **Controlled observation.** On identical data, negatives, architectures, and checkpoint
   fractions, compare AdamW, Muon, and NorMuon in quality, time-to-quality, memory, learning-rate
   tolerance, and failures. This establishes what happens without yet claiming why.
3. **Geometric fingerprint.** At common weights and on common batches, separate three mechanisms:
   AdamW's coordinate-wise adaptation, Muon's singular-direction equalization, and NorMuon's added
   row-wise adaptation. Show the fingerprints in actual updates rather than inferring them only from
   distant checkpoint displacements.
4. **Functional bridge.** Test whether an update fingerprint predicts later representation rank,
   hubness, positive-negative margins, ranking stability, and—only for LateOn—MaxSim token
   utilization. This is where the paper becomes retrieval research rather than a generic optimizer
   benchmark.
5. **Intervention and prescription.** Use scale-matched virtual updates, short common-checkpoint
   branches, component ablations, and optimizer switches to test causality. End with a decision rule
   for when AdamW, Muon, or NorMuon should be used to adapt an embedding model.

The cleanest one-sentence claim, if supported, is:

> Matrix-aware optimizers do not merely change convergence speed; they impose distinct update
> geometries whose transmission into representation and score geometry explains when they help—or
> disrupt—Adam-pretrained neural retrievers.

### Result-contingent story map

The experimental plan should remain publishable under all plausible outcomes:

| Final observation | Defensible main story |
|---|---|
| Muon improves time-to-quality and final retrieval | Spectral update conditioning transfers from language-model training to retrieval, subject to optimizer-mismatch controls. |
| Muon converges faster but ties final retrieval | Muon is a systems/optimization win, while the retrieval function is insensitive to much of the changed weight geometry. |
| Muon lowers loss but hurts zero-shot retrieval | Optimizer mismatch causes excessive functional drift; training loss is a poor proxy for retained retrieval knowledge. |
| NorMuon consistently beats Muon, especially for LateOn | Row-wise adaptation complements orthogonalization and improves token utilization in late interaction. |
| Weight geometry changes but margins/rankings do not | Parameter-space differences are largely functionally redundant; the negative mechanistic result constrains optimizer claims. |

This map must be fixed before the final BEIR matrix is inspected. It prevents selecting a causal
narrative post hoc.

## Proposed NAACL story

This paper should not be framed as an optimizer leaderboard. Its central question is:

> How does matrix-aware optimization change the path taken from an Adam-pretrained language model
> to a neural retriever, and which geometric changes explain optimization speed, robustness, and
> zero-shot retrieval quality?

The causal story to test is:

```text
optimizer transformation
        -> update/weight geometry
        -> representation and score geometry
        -> retrieval dynamics and out-of-domain quality
```

Muon orthogonalizes momentum updates, AdamW rescales coordinates, and NorMuon adds neuron-wise
normalization after orthogonalization. These rules make distinct, falsifiable predictions about
matrix spectra and row balance. Dense and late-interaction retrieval provide two useful tests of
whether those differences survive into a single-vector space or a token-level MaxSim space.

This is a stronger and safer claim than “Muon is better.” A publishable result can be positive,
conditional, or negative:

- Muon may reach the same retrieval quality with less time or optimizer memory.
- NorMuon may improve row balance and token utilization, especially for late interaction.
- Muon may lower training loss while harming zero-shot retrieval because switching from an
  Adam-pretrained solution disrupts useful features.
- The optimizer may change weight spectra without producing a meaningful functional difference.

The experiments must distinguish these outcomes rather than selecting a narrative afterward.

## Research questions and preregistered hypotheses

### RQ1: Does optimizer choice matter for neural retrieval?

Compare AdamW, Muon, and NorMuon over the complete learning-rate sweep, all five checkpoints, both
model families, and the 14-task decontaminated BEIR suite. Report quality versus optimizer steps,
examples, wall time, and peak memory.

**H1.** Muon-family optimizers have a wider stable learning-rate range and improve early quality per
unit wall time, but the final-quality advantage may depend on model family.

### RQ2: How do their updates and trajectories differ?

At identical parameter states and on identical batches, construct the update prescribed by each
optimizer without committing it. Measure layer-wise update spectra, row balance, direction, and
relative scale. Track the same quantities along actual trajectories.

**H2.** Muon produces better-conditioned, higher-effective-rank matrix updates than AdamW, while
NorMuon reduces the dispersion of per-row update norms relative to Muon.

### RQ3: Which weight-space differences reach the retrieval function?

Track hidden-state, output-embedding, and query-document score geometry on fixed probe sets. Relate
these quantities to positive-negative margins, hubness, ranking stability, and BEIR performance.

**H3.** Balanced high-rank updates preserve or increase representation diversity and improve
positive-hard-negative margins. For late interaction, row balance should also reduce concentration
of MaxSim evidence in a small number of query or document tokens.

### RQ4: Are the geometric differences causal?

Use one-step counterfactual probes, short common-checkpoint branches, optimizer switches, and
component ablations. Correlation across completed runs is supporting evidence, not causal evidence.

**H4.** Removing orthogonalization erases Muon's spectral signature; adding row normalization to an
otherwise matched rule specifically changes neuron/token utilization. Geometry changes should
precede, and predict, later changes in margins and retrieval quality.

### RQ5: When is Muon a good fine-tuning optimizer?

Both base models were pretrained with an Adam-family optimizer. Measure feature retention and
optimizer-mismatch sensitivity as a function of update strength, layer, model family, and training
time.

**H5.** Excessively strong Muon updates cause larger functional drift from the pretrained model;
moderate learning rates or constrained updates retain pretrained knowledge while preserving Muon's
conditioning benefit.

## Evidence already supplied by the current study

The current experiment is the broad empirical backbone:

- two retrieval architectures: DenseOn and LateOn;
- one deterministic, shared 500,000-query order with seven explicit hard negatives;
- no in-batch negatives and a fixed 8,192-token context limit;
- AdamW, Muon, and NorMuon, with four learning rates each;
- 24 complete one-epoch runs and five complete checkpoints per run;
- 120 checkpoints with model, optimizer, scheduler, trainer, and rank-local RNG state;
- 1,680 planned checkpoint-task evaluations on 14 decontaminated BEIR tasks;
- training time, loss, gradient norm, memory, checkpoint size, and failure/recovery provenance.

These checkpoints support most retrospective geometry analyses without retraining. The broad sweep
should be described as the discovery phase; selected configurations must be confirmed with new
seeds rather than treating learning-rate-selected test scores as unbiased estimates.

### Artifact-aware implementation boundary

The audited DenseOn and LateOn checkpoints each contain 134 model tensors. Eighty-nine are
two-dimensional, but the training-time optimizer partition identifies exactly 88 transformer hidden
matrices (110,297,088 parameters) in both families; the remaining 2-D tensor is the token embedding
matrix and was routed to auxiliary AdamW. Geometry tooling must reconstruct and verify this declared
partition rather than selecting every 2-D tensor. This prevents an embedding matrix that Muon never
updated from dominating a nominal “Muon weight spectrum” comparison.

Run the retrospective checkpoint analysis in two tiers. Stream every selected tensor from
`model.safetensors` and compute inexpensive exact quantities (Frobenius norm, row/column balance,
checkpoint displacement, and trajectory length) for all 120 checkpoints. Compute full singular
spectra only for a preregistered layer/checkpoint subset; use a deterministic low-rank sketch for the
remaining tensors and report its captured Frobenius-energy fraction. Store one record per tensor and
checkpoint with the input model digest, tensor name, optimizer partition, shape, algorithm, seed, and
approximation settings. A difference between distant checkpoints is a trajectory displacement, not
an optimizer step; raw gradients and actual single-step updates require the common-state probes below.

The repository implements this retrospective tier as `embed-optim-geometry`. It streams both the
root Transformer and SentenceTransformers/PyLate module safetensors, validates the reconstructed
88/auxiliary partition against each run's completion record, hashes every input and atomic JSONL
output, and resumes only when the complete analysis manifest matches. Passing the pinned pretrained
snapshot through `--reference` adds initialization displacement; consecutive checkpoint displacement
is recorded automatically. The tool deliberately labels randomized spectrum and displacement fields
as approximations rather than presenting them as actual optimizer steps. The companion
`embed-optim-summarize-geometry` command enforces the full matrix, revalidates record hashes and
finite values, and emits checkpoint- and run-level trajectory tables; `--verify-inputs` additionally
rehashes every source model tensor file.

### Initial descriptive signal from the completed trajectories

The exact-statistics tier already gives the mechanism section a concrete preregistered target. At
step 3,907, all eight Muon/NorMuon pairs that share a model family and nominal learning rate have a
NorMuon-to-Muon pretrained-reference displacement ratio between 1.000668 and 1.003879. Despite this
nearly identical aggregate scale, NorMuon's parameter-weighted row-norm CV is only 0.232758–0.463783
of Muon's, and its top-1%-row energy share is 0.659264–0.730320 of Muon's. The direction repeats for
all four learning rates in both DenseOn and LateOn.
The strict source table is
[`optimizer_pair_contrasts.csv`](../reports/weight-space/optimizer_pair_contrasts.csv).
The corresponding
[`optimizer_pair_contrast_trajectory.csv`](../reports/weight-space/optimizer_pair_contrast_trajectory.csv)
contains all 40 checkpoint pairs: displacement ratios remain 0.995607–1.003879, row-norm CV ratios
remain 0.166608–0.463783, and top-1%-row energy ratios remain 0.585108–0.730320. Thus the direction
does not emerge only at the final checkpoint.

![Matched Muon and NorMuon checkpoint geometry](../reports/weight-space/optimizer_pair_contrast_trajectory.svg)

This pattern supports a precise working hypothesis: NorMuon changes how trajectory energy is
distributed across neurons without primarily changing total displacement. It is still one-seed,
integrated-trajectory evidence. The paper must test the same signature on individual common-state
updates and connect it to token utilization or retrieval behavior before using causal language.
AdamW uses a different native learning-rate range, so a post-hoc nearest-displacement comparison is
not an adequate control; use matched-scale virtual updates and hybrid AdamW instead.

The overlapping observed displacement range also prevents an overly simple Muon story. For DenseOn,
the final Muon `1e-4` point has displacement/weight 0.007359 and row CV 0.1972; the nearest AdamW
point (`3e-5`, stage 2) has 0.008040 and 0.0951, while NorMuon `1e-4` has 0.007365 and 0.0894. LateOn
shows the same descriptive pattern: 0.007895/0.2012 for Muon, 0.008193/0.1011 for the nearest AdamW
point, and 0.007917/0.0933 for NorMuon. These are post-hoc checkpoint matches, not fair causal
comparisons, but they rule out framing Muon's expected benefit as neuron-wise row balancing. The
Muon hypothesis should instead be tested in singular-spectrum conditioning; row balancing is the
specific NorMuon hypothesis.

![All-optimizer checkpoint geometry by displacement scale](../reports/weight-space/optimizer_geometry_phase.svg)

## Weight- and update-space analysis

Use only hidden 2-D matrices for direct Muon/AdamW geometry comparisons, and report attention and
MLP projections separately. Embeddings, norms, biases, and heads should be analyzed as auxiliary
AdamW parameters rather than pooled with Muon-routed tensors.

For each layer, checkpoint, and common-batch probe, record:

1. Singular-value spectrum of the raw gradient, momentum, optimizer update, weight delta, and
   current weight.
2. Stable rank, entropy effective rank, numerical rank, condition number on non-negligible singular
   values, and spectral/Frobenius/nuclear norms.
3. Row- and column-norm coefficient of variation, Gini coefficient, maximum-to-median ratio, and
   fraction of update energy carried by the largest 1% and 10% of rows.
4. Update-to-weight ratios in both Frobenius and spectral norms.
5. Angles between gradient, momentum, optimizer update, current weight, and displacement from the
   pretrained initialization.
6. Layer-wise path length, final displacement, and path efficiency
   `||W_T-W_0|| / sum_t ||W_t-W_{t-1}||`.
7. Functional sensitivity along each update direction: training loss, held-out contrastive loss,
   score-margin change, and top-k ranking overlap for `W + alpha * Delta W`.

Raw Euclidean distance in parameter space is not sufficient because neural networks have
permutation, scaling, and rotation symmetries. Claims should therefore rely on layer-normalized
metrics, singular values, aligned comparisons, and function-space probes. Use CKA or orthogonal
Procrustes only as descriptive alignment tools; top-k scores and rankings are the decisive functional
measurements.

## Representation- and retrieval-space analysis

Create fixed, versioned probe sets containing training-distribution validation examples and unseen
BEIR queries/documents. Cache tokenization and sample IDs. At every checkpoint measure:

- alignment of query-positive pairs and separation from each of the seven hard negatives;
- mean positive-hard-negative margin and the full margin distribution;
- embedding covariance spectrum, effective rank, mean-vector norm, isotropy, and uniformity;
- nearest-neighbor hubness: k-occurrence skew, Gini coefficient, and dominant-document frequency;
- top-k stability relative to the pretrained checkpoint and between optimizers;
- representational drift by layer using linear CKA and centered Gram-matrix distance;
- robustness to query typos, word deletion, paraphrases, length buckets, and domain shift.

For LateOn, add token-level measures:

- effective rank and isotropy of token embeddings;
- entropy/Gini of per-query-token MaxSim contributions;
- fraction of document tokens selected by at least one query token;
- document-token hubness and repeated-token dominance;
- lexical versus semantic match attribution, split by IDF/frequency bucket.

The strongest analysis links a change at layer `l` to a downstream change in score margin or ranking,
not merely to another intrinsic geometry scalar.

The repository now implements the metric half of this protocol as
`embed-optim-analyze-probe`. It consumes a hashed, fixed `.npz` export with explicit sample IDs,
positive-first candidates, and LateOn token masks. Dense exports are scored with cosine similarity;
LateOn exports are scored with the same MeanMaxSim reduction used for training. The output records
margin distributions, rank/isotropy summaries, optional reference-ranking drift, and LateOn token
evidence concentration/coverage. Probe selection and checkpoint encoding must produce their own
versioned manifest: the analyzer intentionally does not select examples, which keeps selection fixed
and auditable across optimizers.

For the in-distribution tier, `embed-optim-prepare-probe --spec
configs/representation_probe.json` freezes 1,024 complete training groups before any representation
result is inspected. The specification fixes a balanced seven-source allocation, BLAKE2b selection
seed, source-manifest digest, exact selected-ID/ledger digests, and output Dataset fingerprints. This
tier is explicitly training-seen and cannot support a held-out generalization claim. A separately
versioned unseen-BEIR probe remains required for the functional bridge and recipe selection.

`embed-optim-export-probe` supplies the checkpoint-encoding half of the contract. It uses the exact
Dense query/document prefixes or the pinned PyLate query/document/skiplist behavior, then writes
positive-first fp16 arrays. LateOn uses packed ragged token arrays with explicit offsets: the frozen
probe's pre-encoding tokenizer audit found 1,196,149 document tokens but a 3,683-token maximum, so
global padding would inflate each estimated checkpoint payload from 0.29 GiB to 7.19 GiB (25.2x)
before container overhead. The exact post-skiplist arrays are recorded in every export manifest.
The analyzer retains legacy padded-mask compatibility while canonical exports use the packed layout.
A sidecar binds the archive to all model JSON/safetensors inputs, the frozen probe manifest/spec,
package versions, context length, prompts, and encoding hardware. This makes representation metrics
comparable across checkpoints without trusting filenames or an implicit tokenizer/model state.
`embed-optim-probe-matrix` turns this contract into a resumable eight-GPU matrix: it deduplicates the
two pretrained references, gates dependent checkpoint analyses on the corresponding reference
archive, and skips a unit only after both its export and metric provenance pass content-hash audits.

## Causal controls and additional runs

### Required fairness control

Current AdamW applies its swept learning rate to every trainable parameter, whereas Muon/NorMuon
apply the swept rate only to hidden matrices and use AdamW at `3e-6` for embeddings, heads, norms,
and biases. Add a **hybrid AdamW** control with exactly the Muon parameter partition and auxiliary
rate. Tune only its hidden-matrix rate. This separates matrix update rule from parameter grouping.

### Common-state virtual updates

At the pretrained model and selected 20%, 60%, and 100% checkpoints:

1. load one common model state;
2. compute a gradient on the same fixed batch;
3. apply each optimizer transform in memory;
4. normalize updates under several fair budgets (same Frobenius norm, same spectral norm, and same
   first-order predicted decrease);
5. compare spectra and measure the immediate functional effect without saving a new checkpoint.

This is the cleanest way to show that differences come from the update rule rather than different
points on different trajectories.

Optimizer state needs its own control. Report two explicitly separate protocols:

1. **Cold-start transform:** initialize every optimizer state to zero at the common weights and
   compare the first prescribed update. This is exactly reproducible but measures initialization
   behavior.
2. **Frozen-weight state warm-up:** hold weights fixed, feed every optimizer the same ordered sequence
   of probe-batch gradients to build momentum/second-moment state without applying updates, then
   compare the update on a held-out next batch. This isolates the stateful transform under a shared
   gradient history.

Loading each optimizer's native checkpoint moments would reintroduce trajectory history and must not
be described as a common-state comparison. Native-state probes can be reported separately as an
ecological description of what each trained run would do next. In all scale-matched conditions,
match both global and per-layer budgets so a change in layer allocation is not mistaken for a change
in within-matrix geometry.

The repository implements the frozen-weight state-warm-up protocol with
`embed-optim-export-gradients` and `embed-optim-analyze-updates`. The preregistered settings in
`configs/common_state_probe.json` select 32 source-balanced examples, average them into eight ordered
gradient states, use micro-batches of one, apply the formal global gradient-clipping threshold, and
never advance the checkpoint weights. Parameters and accumulated gradients remain float32 while the
forward pass uses the same bfloat16 autocast and non-reentrant gradient-checkpointing policy as
training. The model remains in training mode so checkpointing is operational; each shard resets a
fixed RNG seed, and the resulting cached dropout realization is shared by every optimizer replay.
Shards are committed and hashed
independently so interrupted GPU exports resume without changing their sample/RNG sequence. The
analyzer replays the exact AdamW, Muon, and NorMuon state equations used by training, including CUDA
bfloat16 Newton–Schulz, records raw direction geometry, and exports per-layer Frobenius-matched
directions for downstream interventions. Weight decay is kept separate;
otherwise its checkpoint-dependent `lambda W` term would contaminate the comparison of data-gradient
preconditioners. A one-gradient export is the cold-start diagnostic, but it is not interchangeable
with the preregistered eight-gradient stateful result.

The formal anchor matrix is also frozen before the complete BEIR matrix is available. The immutable
spec records that 98 of 1,680 strict units and partial scores had already been observed at freeze
time; this is therefore a prospective completion lock, not a claim of preregistration before any
outcome inspection. It includes the pretrained state and the 20%, 60%, and 100% checkpoints from
`adamw-lr1e-5`, `muon-lr1e-3`, and `normuon-lr1e-3` for each family: 10 anchors per family and 20
total. These nominal interior-rate trajectories are mechanism anchors, not task-suite winners. The
`embed-optim-common-state-matrix` dispatcher materializes this exact grid across eight GPUs, resumes
only provenance-compatible gradient shards, and runs an independent hash audit over all gradient,
metric, and matched-direction artifacts. The matrix intentionally samples states visited by all
three optimizer families; every counterfactual transform at an anchor still receives the same
weights and the same ordered gradient history.

The spec also preserves one pre-execution amendment: `model_mode` changed from evaluation to
training before any common-state GPU artifact existed, because Transformers gates gradient
checkpointing on `module.training`. This correction changes the dropout realization but not the
shared-gradient comparison; the per-shard RNG sequence is fixed and recorded.

### Short counterfactual branches

From the same pretrained, 20%, and 60% checkpoints, train each optimizer for a short fixed sequence
of batches. Include optimizer switches in both directions. Evaluate held-out loss, margin, feature
drift, and a small retrieval probe before any full BEIR run.

### Component ablations

Use a compact decomposition rather than a large optimizer zoo:

- matched hybrid AdamW;
- momentum update without matrix orthogonalization;
- Muon: momentum plus orthogonalization;
- row-normalized momentum without orthogonalization;
- NorMuon: orthogonalization plus row normalization.

Match update scale at the first step and report both matched-scale and native-recipe results. This
tests which effect comes from orthogonalization and which from neuron-wise adaptation.

### Confirmatory multi-seed runs

After the exploratory sweep, freeze one configuration per optimizer and family using a validation
criterion that does not inspect BEIR test labels. Run at least three seeds (preferably five) with the
same data IDs but independently seeded negative selection and example order. Report hierarchical
bootstrap confidence intervals over seeds, tasks, and queries. Use paired randomization/bootstrap
tests at query level and correct the small set of preregistered optimizer comparisons.

## Selection protocol and statistical claims

Do not choose the best learning rate on the same 14 BEIR test tasks used for the headline number.
Use the current four-rate grid as exploratory evidence, then select recipes with a held-out source
validation set or a separately declared development suite. Lock recipes before confirmatory BEIR
evaluation.

Report three complementary comparisons:

1. **Recipe comparison:** each optimizer at its independently tuned best configuration.
2. **Compute-matched comparison:** best quality reached at equal GPU-hours or wall time.
3. **Geometry-matched comparison:** common-state updates matched by update norm or predicted loss
   decrease.

Treat BEIR datasets and random seeds as sampling levels. Report the per-task table even when an
aggregate is favorable; optimizer benefits that come from only one large dataset are not a robust
result.

## Claim-evidence firewall

| Intended claim | Minimum supporting evidence | Evidence that is insufficient by itself |
|---|---|---|
| Faster or more memory-efficient training | Audited useful wall time, examples/second, time-to-quality, and peak memory on matched hardware | Lower loss at the same step |
| Better retrieval recipe | Frozen validation-selected recipe, confirmatory seeds, paired query/task uncertainty, and full per-task results | Best learning rate selected on BEIR test scores |
| NorMuon balances update energy | Common-state individual updates with matched global/per-layer scale and repeated layers/batches | Lower row CV in distant checkpoint displacement |
| Geometry explains retrieval behavior | Geometry change precedes margin/ranking change and survives short-branch or intervention controls | Cross-run correlation between two intrinsic metrics |
| More robust optimization | Prespecified stability criteria across seeds, learning rates, batches, and perturbations | One wide learning-rate sweep with a single seed |

The abstract and conclusion should contain only claims that cross the corresponding evidence bar.
Everything else should be labeled descriptive or exploratory.

## Main paper figures

1. **Quality-efficiency frontier:** mean decontaminated BEIR nDCG@10 versus wall time for all five
   checkpoints, with stable learning-rate regions and peak memory.
2. **Update geometry fingerprint:** layer-by-depth heatmaps of update effective rank, condition
   number, and row-norm CV for identical-batch counterfactual updates.
3. **The bridge:** update geometry, representation effective rank/hubness, margin, and future BEIR
   quality on a shared training timeline.
4. **Causal intervention:** short branches from a common checkpoint and optimizer-switch results.
5. **Dense versus late interaction:** single-vector geometry against token-level MaxSim utilization.
6. **Mismatch and robustness:** pretrained-feature drift versus retrieval quality under native and
   matched update strengths.

The main results table should include final quality, best intermediate quality, time-to-quality,
optimizer-state memory, and run stability. Full learning-rate and task matrices belong in the
appendix.

## Suggested paper structure

1. **Introduction:** AdamW is the retrieval default; matrix-aware optimizers change the geometry of
   learning, but their effect on neural retrievers is unknown.
2. **Optimizer geometry:** explain coordinate-wise AdamW, Muon orthogonalization, NorMuon row-wise
   adaptation, and predictions for embeddings.
3. **Controlled retrieval benchmark:** models, shared data, negatives, optimization grid, five-point
   dynamics, decontaminated evaluation, and systems measurements.
4. **Do matrix optimizers help?** quality, speed, memory, robustness, and learning-rate sensitivity.
5. **From updates to representations:** common-state update spectra, trajectory analysis, output
   geometry, hubness, margins, and token utilization.
6. **Causal tests and optimizer mismatch:** matched hybrid AdamW, scale matching, short branches,
   component ablations, and switches.
7. **Implications:** when to use Muon/NorMuon for full fine-tuning and what retrieval-specific
   optimizer design should target.
8. **Limitations:** two related base models, one training corpus, Adam-pretrained initialization,
   finite hyperparameter budget, and intrinsic metrics that do not by themselves establish cause.

## Candidate title and one-sentence contribution

**Title:** *From Update Geometry to Retrieval Geometry: Understanding Muon for Dense and
Late-Interaction Text Embeddings*

**Contribution sentence:** We provide the first controlled study connecting matrix-aware optimizer
updates to weight spectra, embedding geometry, token-level interaction patterns, and zero-shot
retrieval dynamics, with causal controls that separate update geometry from learning rate,
parameter routing, and optimizer-pretraining mismatch.

## Novelty boundary

Prior work already argues that Muon acts under spectral-norm constraints, that NorMuon balances
neuron-wise updates, that optimizer choice can change learned spectral structure, and that Muon may
behave differently when fine-tuning Adam-pretrained models. Prior NLP work also connects contrastive
learning and retrieval to isotropy. The paper should cite these directly and avoid claiming that
effective-rank or isotropy measurements alone are new.

The defensible novelty is their **retrieval-specific causal connection**: two retriever families,
full training dynamics, common-state interventions, token-level MaxSim evidence, decontaminated
zero-shot retrieval, and a fairness control for hybrid parameter routing.

### Related-work anchors

The paper should organize related work by the claim each source already establishes, rather than by
chronology:

- [Muon](https://kellerjordan.github.io/posts/muon/) introduces orthogonalized momentum for hidden
  matrix parameters. [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) provides
  large-scale language-model evidence and emphasizes weight decay and per-parameter update scaling.
  Neither study establishes transfer to contrastive neural retrieval.
- [NorMuon](https://arxiv.org/abs/2510.05491) identifies non-uniform neuron norms after Muon
  orthogonalization and adds neuron-wise second-moment normalization. Our row-balance measurements
  are therefore a test of an existing mechanism in a new retrieval regime, not a new optimizer
  principle.
- [Can Muon Fine-tune Adam-Pretrained Models?](https://arxiv.org/abs/2605.10468) reports that
  optimizer mismatch can disrupt Adam-pretrained knowledge and that the effect grows with update
  strength. Our contribution is to test this issue in full-rank embedding adaptation and connect it
  to retrieval margins, rankings, and architecture-specific representations.
- [HIL](https://aclanthology.org/2024.naacl-long.437/) connects representation
  isotropy/anisotropy to zero-shot dense retrieval. Isotropy is consequently an explanatory variable
  here, not a standalone novelty claim; optimizer interventions must connect it to downstream
  rankings.

The introduction should cite these sources before stating the gap. The strongest gap statement is:

> Existing work studies matrix-aware optimization in language-model training, optimizer mismatch in
> general fine-tuning, or embedding geometry in retrieval separately; no controlled study traces the
> causal path from matrix update geometry to dense and token-level retrieval behavior.
