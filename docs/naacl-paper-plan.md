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
4. **Functional bridge.** Intervene locally at fixed weights along Frobenius-matched optimizer
   directions, then test whether the same update fingerprint predicts later representation rank,
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

## Research questions and prospectively frozen hypotheses

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
NorMuon reduces the dispersion of per-row update norms relative to Muon. This is a directional
hypothesis, not an assumption that a flatter spectrum is always beneficial: the competing outcome is
that orthogonalization overweights noise-dominated directions or suppresses a useful heavy tail.

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

## Identification strategy: four spaces, three evidence levels

Use **optimization geometry**, rather than raw *weight-space distance*, as the mechanism umbrella.
Transformer parameterizations admit rotations, permutations, and rescalings that can move weights
without changing the retrieval function. The paper should therefore keep four spaces distinct:

| Space | Primary object | Main measurements | What it can establish |
|---|---|---|---|
| Operator/update | The update prescribed at the same weights from the same gradient history | singular spectrum, effective rank, row-energy concentration, update angles, update/weight scale | the optimizer rules have distinct geometric fingerprints |
| Trajectory/weight | The accumulated path from the shared pretrained initialization | normalized displacement, path length/efficiency, weight spectra, layer allocation | the fingerprints persist or disappear under training, descriptively |
| Representation/score | Query, document, and LateOn token representations and their scores | covariance rank, isotropy, hubness, margin, ranking overlap, MaxSim evidence entropy | a geometric difference reaches the retrieval function |
| Retrieval outcome | Rankings on held-out validation and decontaminated BEIR | nDCG@10, time-to-quality, task and seed uncertainty | the changed function is useful, neutral, or harmful |

The causal evidence should then escalate in three levels:

1. **Descriptive:** complete learning-rate trajectories and checkpoint geometry show what co-varies.
   These results motivate hypotheses but cannot identify the optimizer operator.
2. **Locally causal:** common-state replay and sign/scale-matched virtual steps change only the
   prescribed direction while holding weights, examples, gradients, and scale fixed. These identify
   immediate directional effects on loss, margin, and rankings.
3. **Accumulated causal:** short shared-checkpoint branches, hybrid-AdamW routing, and component
   ablations test whether the local effect survives repeated updates. New-seed confirmatory runs
   establish whether the resulting retrieval prescription generalizes.

Do not describe the observational update-representation correlations as a formal causal mediation
analysis. The defensible wording is a *mechanistic chain supported by interventions*: operator
interventions establish the first link, function-space measurements establish the second, and short
branches test accumulation over optimization time.

### Primary contrasts

Keep a small, prospectively fixed set of contrasts so the paper does not become an inventory of
geometry metrics:

1. **AdamW versus hybrid AdamW:** effect of parameter routing and auxiliary learning rate.
2. **Hybrid AdamW versus Muon:** effect of matrix orthogonalization under matched routing and update
   budgets.
3. **Muon versus NorMuon:** effect of neuron-wise adaptation after orthogonalization.
4. **DenseOn versus LateOn interaction:** whether a single-vector and token-level scoring function
   transmit the same update signature differently.
5. **Native versus scale-matched update:** whether a result comes from direction or merely update
   magnitude.

All other layer, checkpoint, and task breakdowns explain these contrasts; they are not additional
independent headline hypotheses.

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

### Initial systems signal and narrative consequence

The audited training-only tables already rule out one tempting headline. Across the four learning
rate sweep points, DenseOn median throughput relative to AdamW is 0.9489 for Muon and 0.9348 for
NorMuon; LateOn ratios are 0.9946 and 0.9860. Native end-to-end fine-tuning is therefore not faster
in these runs. The optimizer-state footprint is nevertheless materially smaller: Muon/NorMuon use
0.6299/0.6304 of AdamW's state size for DenseOn and 0.6415/0.6420 for LateOn. These are descriptive
matched-hardware measurements from
[`optimizer_system_summary.csv`](../reports/training-dynamics/optimizer_system_summary.csv), not
independent-seed estimates.

This sharpens the paper's empirical tension. The question is not whether a language-model
pretraining speed claim automatically transfers to retriever adaptation; it demonstrably does not
at the raw throughput level here. The remaining efficiency claim must be earned through
time-to-retrieval-quality: a slightly slower step can still be useful if its update geometry reaches
a target retrieval score in sufficiently fewer steps, while lower optimizer-state memory is a
separate systems benefit. Operator efficiency, realized throughput, state memory, and
time-to-quality must therefore remain four distinct quantities throughout the paper.

The formal time-to-quality renderer is `embed-optim-summarize-retrieval-dynamics`. It reconstructs
all 120 checkpoint means from the 1,680 provenance-valid task files, then defines one reference per
model family as the median final nDCG@10 of the four AdamW learning-rate points. For every one of the
24 trajectories it reports the first of the five observed checkpoints that reaches this reference;
it performs no interpolation and retains non-reaching runs as right-censored. Checkpoint time is
the run's audited useful wall time multiplied by `checkpoint_step / 3907`, explicitly labeled as a
step-proportional estimate rather than a measured checkpoint timestamp. This Muon-outcome-independent
rule avoids choosing a visually favorable absolute threshold after seeing Muon results. The rule was
frozen after 160 of 1,680 discovery evaluation units were visible, while the complete retrieval
matrix and this report were not visible. It is therefore a prospectively locked completion analysis,
not a preregistration; the manuscript must disclose that timing and keep the analysis exploratory.

![Native recipe systems trade-offs](../reports/training-dynamics/system_tradeoffs.svg)

The corresponding five-stage training-loss panel retains every learning-rate trajectory rather than
collapsing each optimizer to its winning configuration. It belongs in the appendix unless loss and
retrieval quality diverge strongly enough to become a main finding.

![All discovery training-loss trajectories](../reports/training-dynamics/training_loss_dynamics.svg)

### Artifact-aware implementation boundary

The audited DenseOn and LateOn checkpoints each contain 134 model tensors. Eighty-nine are
two-dimensional, but the training-time optimizer partition identifies exactly 88 transformer hidden
matrices (110,297,088 parameters) in both families; the remaining 2-D tensor is the token embedding
matrix and was routed to auxiliary AdamW. Geometry tooling must reconstruct and verify this declared
partition rather than selecting every 2-D tensor. This prevents an embedding matrix that Muon never
updated from dominating a nominal “Muon weight spectrum” comparison.

Run the retrospective checkpoint analysis in two tiers. Stream every selected tensor from
`model.safetensors` and compute inexpensive exact quantities (Frobenius norm, row/column balance,
checkpoint displacement, and the coarse path through retained checkpoints) for all 120 checkpoints.
The completed all-checkpoint tier uses `--sketch-rank 0`; it does not infer a singular spectrum from
disabled fields. Compute exact singular spectra only for the prespecified state--layer subset under
the common-state protocol. Store one record per tensor and checkpoint, and bind the input model
digest, optimizer partition, algorithm, seed, and analysis settings in its run manifest. A difference
between distant checkpoints is a trajectory displacement, not an optimizer step; raw gradients and
actual single-step updates require the common-state probes below.

The repository implements this retrospective tier as `embed-optim-geometry`. It streams both the
root Transformer and SentenceTransformers/PyLate module safetensors, validates the reconstructed
88/auxiliary partition against each run's completion record, hashes every input and atomic JSONL
output, and resumes only when the complete analysis manifest matches. Passing the pinned pretrained
snapshot through `--reference` adds initialization displacement; consecutive checkpoint displacement
is recorded automatically. When enabled, the tool labels randomized spectra as approximate;
checkpoint displacement remains an exact distant-state difference and is never presented as an
optimizer step. The companion
`embed-optim-summarize-geometry` command enforces the full matrix, revalidates record hashes and
finite values, and emits checkpoint- and run-level trajectory tables; `--verify-inputs` additionally
rehashes every source model tensor file.

### Initial descriptive signal from the completed trajectories

The exact-statistics tier already gives the mechanism section a concrete prespecified target. At
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
   values, spectral/Frobenius/nuclear norms, and—only where the full spectrum is computed—spectral
   tail/heavy-tail diagnostics with their fitted range and goodness of fit.
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
versioned unseen-BEIR probe is now implemented by `embed-optim-prepare-beir-probe`. Its prospective
protocol samples 16 queries per pinned task without using model outcomes, chooses a deterministic
highest-qrel positive, and uses seven qrel-excluded cross-query positives ranked by lexical overlap.
The resulting 224-row tier is balanced across all 14 tasks and held out from the 500K training view.
Its checked-in specification discloses that 98 of 1,680 BEIR units and partial scores had already
been observed. The final 224-row artifact was independently checked against the raw pinned qrels
(224 positive and 1,568 negative judgments), and a frozen rerun reproduced the ledger, Arrow data,
Dataset metadata, state, and manifest byte-for-byte. The specification records two pre-output pool
size amendments forced by the small NQ and Touche2020 qrel sets; no model output informed either
change. These lexical negatives are functional probes, not a replacement for full-corpus BEIR
evaluation or a claim that they reproduce the training hard-negative distribution.

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
`embed-optim-summarize-probes` is the strict downstream gate: it binds a summary to the frozen probe
and spec, requires two reference plus 120 checkpoint reports, checks every task/source quota and
representation role, and emits separate checkpoint, long-form representation, and group tables with
a hashed summary manifest. This prevents a partial or cross-probe directory from becoming a paper
figure through manual JSON collection.
`embed-optim-plot-representation-dynamics` then requires both complete formal tiers and renders the
shared DenseOn/LateOn panel for margin, query effective rank, and pretrained-ranking agreement. It
uses every learning-rate trajectory, shows the common pretrained point, and records the exact source
hashes in a sidecar instead of selecting a visually favorable configuration.
The companion `embed-optim-plot-late-token-dynamics` panel exposes all four prespecified MaxSim
token-utilization summaries on both tiers. Keeping this panel separate prevents LateOn-only evidence
from changing the shared cross-architecture plot definition after results are visible.
After strict BEIR aggregation, `embed-optim-build-mechanism-bridge` joins all 120 checkpoints across
weight trajectories, both probe tiers, and 14-task mean retrieval quality. It exports both raw
checkpoint levels and 96 within-run first differences, plus a fixed set of family-level and
optimizer-conditional Spearman associations. These tables test temporal co-movement and organize
the paper's bridge figure, but their manifest explicitly labels them observational; only the
common-state and short-branch interventions can support causal wording.

## Causal controls and additional runs

### Required fairness control

Current AdamW applies its swept learning rate to every trainable parameter, whereas Muon/NorMuon
apply the swept rate only to hidden matrices and use AdamW at `3e-6` for embeddings, heads, norms,
and biases. Add a **hybrid AdamW** control with exactly the Muon parameter partition and auxiliary
rate. Tune only its hidden-matrix rate. This separates matrix update rule from parameter grouping.

The repository implements this as the separate eight-run matrix in
`configs/hybrid_adamw.yaml`, with its immutable visibility and selection ledger in
`configs/hybrid_adamw_control.json`. It reuses the four original AdamW hidden learning rates and
fixes the auxiliary rate to `3e-6`; only final checkpoints receive the formal 14-task BEIR suite.
The ledger discloses that 140/1,680 discovery evaluation units and all weight trajectories were
visible at freeze time, while no hybrid, formal common-state, or formal representation output
existed. The control is therefore prospectively locked but not preregistered.

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
`embed-optim-export-gradients` and `embed-optim-analyze-updates`. The prospectively frozen settings in
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
with the prospectively frozen eight-gradient stateful result.

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

After the grid completes, `embed-optim-summarize-common-state` performs a second strict audit and
emits raw gradient/update tensor tables, anchor-level parameter-weighted summaries, pairwise
direction cosines, and same-anchor Muon/NorMuon contrasts against AdamW. This is the only supported
route from the 20 analyzer directories to paper figures: incomplete diagnostics remain visibly
marked and cannot be mistaken for the final mechanism result.

The representative full-spectrum tier is independently locked in
`configs/common_state_spectrum_probe.json`: layers 0, 10, and 21 crossed with attention input and
MLP expansion projections, all 20 anchors, both families, and all three counterfactual update
operators. The freeze ledger discloses that 110/1,680 BEIR units and the completed weight
trajectories were already visible, but no formal common-state or representation output existed.
`embed-optim-common-state-spectra` computes all 360 exact spectra, content-hashes every source and
output, and emits both per-spectrum summaries and normalized long-form singular-value curves. This
prevents choosing visually favorable layers after inspecting the update spectra.
`embed-optim-plot-common-state-spectra` then renders the main mechanism panel as medians and
interquartile bands over all ten anchors per family; no individual checkpoint is selected for visual
convenience.

The spec also preserves one pre-execution amendment: `model_mode` changed from evaluation to
training before any common-state GPU artifact existed, because Transformers gates gradient
checkpointing on `module.training`. This correction changes the dropout realization but not the
shared-gradient comparison; the per-shard RNG sequence is fixed and recorded.

### Short counterfactual branches

The minimum long-horizon intervention is now frozen in `configs/short_branch_protocol.json`. Both
families start from the 60% `adamw-lr1e-5` checkpoint (step 2,345), independently of retrieval
quality. A deterministic proportional 50,000-group subset preserves every query, positive, and
seven-negative group. AdamW uses hybrid hidden/auxiliary routing; AdamW, Muon, and NorMuon hidden
learning rates are derived from their common-state raw direction norms so that the calibration
condition has one shared global hidden update/weight target of `5e-4`. Each operator is then trained
for one 50K-subset epoch under order seeds `314159`, `271828`, and `161803` (18 runs total).

Evaluate all five branch checkpoints on the frozen query-disjoint validation and 224-query unseen
retrieval probes, including LateOn token utilization, but do not spend another full-corpus BEIR
matrix on this mechanism control. This tests whether the scale-matched local direction survives
repeated updates. Bidirectional optimizer switches from other trajectory checkpoints remain an
extension rather than a main-paper gate.

### Component ablations

Use a compact decomposition rather than a large optimizer zoo:

- matched hybrid AdamW;
- momentum update without matrix orthogonalization;
- Muon: momentum plus orthogonalization;
- row-normalized momentum without orthogonalization;
- NorMuon: orthogonalization plus row normalization.

Match update scale at the first step and report both matched-scale and native-recipe results. This
tests which effect comes from orthogonalization and which from neuron-wise adaptation.

### Basis-sensitivity diagnostic

Treat optimizer geometry at three distinct granularities: AdamW is coordinate-wise, Muon is
matrix/singular-direction aware, and NorMuon additionally privileges the output-neuron row basis.
This suggests a compact diagnostic that does not require another full training sweep. For selected
attention heads, apply a fixed seeded orthogonal rotation `R` to the query and key output coordinates
so that the attention logits are unchanged, replay the same frozen gradient history in the original
and rotated parameterizations, map each prescribed update back to the original basis, and compare
update cosine, norm, spectrum, and one-step function drift. Use the fused QKV layout only after
splitting and independently validating the Q/K slices.

This is a symmetry diagnostic, not a retrieval-quality result. It tests whether the optimizer reacts
to an arbitrary coordinate representation of the same attention function. Exact polar Muon should
be orthogonally equivariant up to numerical approximation; coordinate-wise AdamW and NorMuon's
row-specific state can retain basis sensitivity. Keep this experiment in the appendix unless its
effect predicts the main representation or retrieval results.

### Confirmatory multi-seed runs

The non-BEIR selection stage is now concretely frozen in `configs/validation_probe.json`. It selects
4,096 source-balanced queries from the unused portion of the pinned 1.22M-query source after
recomputing and excluding all 500,000 training query IDs. All 24 exploratory final checkpoints are
scored with the exact eight-way training objective. Within each optimizer and family, choose the
lowest mean contrastive loss, then the highest positive margin and lower hidden learning rate as
tie-breakers. This rule, its 24-job output, and the resulting six recipes are content-audited before
any confirmatory run is generated; BEIR scores are never an input to recipe selection.

The confirmatory randomization is now concretely frozen in `configs/confirmatory_protocol.json`.
Seeds `314159`, `271828`, and `161803` retain the exact 500,000 query, positive, sample, and source
identities. Each independently resamples seven documents without replacement from the reconstructed
first-ten eligible negative pool and supplies both the Trainer and data seed. The preparation audit
must reverse-reconstruct the seed-42 ledger, require distinct non-positive pool entries, prove zero
query/positive text drift, and require at least 98% changed negative groups for every base/new and
new/new seed pair.

After query-disjoint validation selects one recipe for each family/optimizer, generate exactly six
runs per new seed (18 total). Retain five checkpoints for restart/artifact integrity but formally
evaluate only the final checkpoint on BEIR, adding 252 confirmatory retrieval units. Report
deterministic hierarchical bootstrap intervals that independently resample seeds and tasks. The
aggregate MTEB outputs retain task scores but not per-query rankings, so no query-level test or
significance claim is available. For all six frozen family-by-optimizer contrasts, report both the
nominal 95% interval and a simultaneous familywise 95% interval obtained by applying a Bonferroni
correction across the six comparisons. Only the familywise interval may determine positive,
negative, or inconclusive headline language. The original exploratory seed must not be silently
counted as a confirmatory seed after its BEIR results were visible. Five seeds remain a useful
extension if the three-seed intervals are too wide, not a post-hoc way to reverse an unfavorable
conclusion.

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

The first geometry-matched causal test is frozen in `configs/functional_intervention.json`. At each
of the 20 common-state anchors it applies `W - alpha D` for AdamW, Muon, and NorMuon, where every
hidden tensor in `D` has the same Frobenius norm as its corresponding checkpoint tensor. The fixed
relative scales are `1e-4`, `3e-4`, and `1e-3`; a sign-reversed `1e-3` condition tests
directionality. Baseline plus 12 interventions are evaluated on all 224 examples in the separately
frozen decontaminated-BEIR probe, yielding 58,240 paired sample records. Loss, positive margin,
reciprocal rank, and top-1 accuracy use the exact eight-way DenseOn/LateOn training scorer. The lock
records that 144/1,680 retrieval units and all weight trajectories were visible, but no formal
common-state, representation, or intervention output existed. This supports only an immediate
fixed-weight causal claim; a shared-checkpoint short branch is still required before making a
long-horizon optimization claim.

Treat BEIR datasets and random seeds as sampling levels. Report the per-task table even when an
aggregate is favorable; optimizer benefits that come from only one large dataset are not a robust
result.

## Minimum publishable package versus extensions

The paper should have a hard core so the mechanism story does not expand indefinitely.

**Main-paper completion gates:**

1. Complete the current 24-run, 1,680-unit exploratory matrix and report all learning rates rather
   than only winners.
2. Complete the frozen 20-anchor common-state matrix and show at least one spectral signature that
   distinguishes AdamW/Muon plus the row-balance signature that distinguishes Muon/NorMuon.
3. Connect those signatures to fixed representation/score probes in both model families; a weight
   metric with no margin, ranking, or token-utilization consequence is a negative mechanism result.
4. Run the hybrid-AdamW routing control, the frozen scale-matched one-step intervention, and at least
   one scale-matched short branch from a shared checkpoint. Together these separate update rule,
   parameter grouping, immediate directional efficacy, and accumulated trajectory effects.
5. Freeze recipes using non-BEIR validation, then confirm the headline comparison with at least three
   independent seeds. The exploratory seed may be shown but should not silently become one of the
   confirmatory seeds after its test results influenced recipe choice.

Optimizer switches in both directions, perturbation robustness, complete activation-covariance
maps, heavy-tail fits for every layer, and five confirmatory seeds are valuable extensions. They
belong in the appendix or follow-up work unless one is needed to resolve the main causal result.
Confirmatory runs need only the final BEIR checkpoint unless the paper makes a seed-level claim about
training dynamics; the existing sweep already supplies the exploratory five-stage trajectories.

## Claim-evidence firewall

| Intended claim | Minimum supporting evidence | Evidence that is insufficient by itself |
|---|---|---|
| Faster or more memory-efficient training | Audited useful wall time, examples/second, time-to-quality, and peak memory on matched hardware | Lower loss at the same step |
| Better retrieval recipe | Frozen validation-selected recipe, confirmatory seeds, paired seed/task uncertainty, and full per-task results | Best learning rate selected on BEIR test scores |
| NorMuon balances update energy | Common-state individual updates with matched global/per-layer scale and repeated layers/batches | Lower row CV in distant checkpoint displacement |
| Geometry explains retrieval behavior | Geometry change precedes margin/ranking change and survives short-branch or intervention controls | Cross-run correlation between two intrinsic metrics |
| More robust optimization | Prespecified stability criteria across seeds, learning rates, batches, and perturbations | One wide learning-rate sweep with a single seed |

The abstract and conclusion should contain only claims that cross the corresponding evidence bar.
Everything else should be labeled descriptive or exploratory.

## Main paper figures

Keep the main-paper visual budget to four composite figures; a six-figure mechanism inventory would
crowd an NAACL-length argument and make the paper read like unrelated analyses.

1. **What happens:** mean decontaminated BEIR nDCG@10 versus wall time for all five checkpoints,
   with learning-rate stability and peak memory in aligned panels.
2. **What the optimizer does:** layer-by-depth heatmaps plus representative spectra for common-state
   update effective rank, spectral tail, condition, row-norm CV, and optimizer-pair direction angle.
3. **How it reaches retrieval:** a shared-timeline bridge from update geometry to representation
   rank/hubness and score margin, with DenseOn and LateOn/MaxSim token utilization as two panels.
4. **Why the bridge is causal:** scale-matched virtual steps, short branches, optimizer switches,
   and pretrained-feature drift summarized in one intervention figure.

The main results table should include final quality, best intermediate quality, time-to-quality,
optimizer-state memory, and run stability. Full learning-rate and task matrices belong in the
appendix. Put standalone heavy-tail fits, layer-wise activation covariance, robustness perturbations,
and the full mismatch sweep in the appendix unless one becomes the decisive explanation.

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

## Abstract skeleton

Neural retrievers are usually fine-tuned with AdamW, while matrix-aware optimizers such as Muon
promise faster and more robust optimization by changing the geometry of matrix updates. Whether
that inductive bias transfers to Adam-pretrained embedding models—and whether it affects dense and
late-interaction retrievers through the same mechanism—remains unclear. We compare AdamW, Muon, and
NorMuon in a controlled 24-run study over two retriever architectures, four learning rates, five
training stages, and 14 decontaminated zero-shot retrieval tasks. We then evaluate every optimizer
at common weight states and on common gradient histories, connect update spectra and row balance to
representation rank, margins, ranking drift, and MaxSim token utilization, and test the resulting
links with scale-matched short branches and routing controls. The final two sentences should be
filled only after confirmatory results exist: one sentence stating the strongest supported empirical
result and one stating the strongest supported mechanism or negative-mechanism conclusion.

## Reviewer-question map

| Likely reviewer question | Evidence placed in the paper |
|---|---|
| Is this only a hyperparameter sweep? | Common-state transforms, exact spectra, and intervention branches identify optimizer-specific mechanisms. |
| Is the AdamW comparison unfair? | Hybrid AdamW uses Muon's hidden/auxiliary parameter routing, plus compute- and update-scale-matched comparisons. |
| Were learning rates chosen on BEIR test data? | The full grid is exploratory; recipes are frozen on non-BEIR validation before multi-seed confirmation. |
| Do intrinsic geometry metrics matter for retrieval? | Weight/update changes are linked to margins, rankings, hubness, and LateOn token evidence, then tested by short branches. |
| Are correlations being called causal? | Checkpoint correlations remain descriptive; causal wording is reserved for common-state and intervention evidence. |
| Does Muon simply balance neuron rows? | Muon's spectral hypothesis and NorMuon's row-balancing hypothesis are explicitly separated. |
| Is a win just faster kernels or lower optimizer memory? | Quality is reported against steps and wall time alongside throughput, peak memory, and optimizer-state size. |

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
- [Optimizer-Model Consistency](https://arxiv.org/abs/2605.06654) argues that using the pretraining
  optimizer during full fine-tuning can improve the learning--forgetting tradeoff. This makes
  pretrained-ranking retention and query-disjoint loss complementary outcomes in our study: a new
  optimizer can fit the retrieval objective while still erasing useful pretrained behavior.
- [HTMuon](https://aclanthology.org/2026.findings-acl.1819/) argues that Muon can suppress
  heavy-tailed weight spectra and overemphasize noise-dominated update directions. This gives our
  spectrum analysis a genuine competing hypothesis: higher update effective rank need not imply
  better retrieval, and tail fits should be reported only for a separately frozen full-spectrum
  tier rather than inferred from the rank-64 sketches.
- [The Newton-Muon Optimizer](https://arxiv.org/abs/2604.01472) interprets standard Muon as omitting
  right preconditioning by the layer-input second moment. Dense versus late-interaction retrieval
  may expose different input-covariance regimes, so activation covariance is a useful explanatory
  measurement rather than evidence that orthogonalization alone is sufficient.
- [Adaptive Optimization via Schatten-p Norms](https://arxiv.org/abs/2605.19781) places SGD, Muon,
  and Adam-like rules inside a broader family of layer-dependent matrix geometries. It narrows our
  novelty claim from “geometry matters” to the retrieval-specific causal bridge and motivates
  reporting which layers favor which update geometry instead of seeking one universal winner.
- [Muon Learns More Robust and Transferable Features than Adam](https://arxiv.org/abs/2606.09658)
  connects Muon pretraining to larger margins, higher hidden-state effective rank, robustness, and
  transfer in language and vision classification. These are direct competing explanations for our
  representation probes, but they do not establish that the same effects survive an optimizer
  switch from Adam-pretrained weights or improve dense/late-interaction rankings.
- [The Loss Does Not See the Basis, but Adam Does](https://arxiv.org/abs/2608.05136) formalizes a
  coordinate-basis dependence of Adam in factored models and contrasts it with gauge-equivariant
  update rules including Muon. Our optional Q/K rotation diagnostic tests a narrow, function-
  preserving attention symmetry; it should not generalize that theory to the entire Transformer
  without evidence.
- [HIL](https://aclanthology.org/2024.naacl-long.437/) connects representation
  isotropy/anisotropy to zero-shot dense retrieval. Isotropy is consequently an explanatory variable
  here, not a standalone novelty claim; optimizer interventions must connect it to downstream
  rankings.

The introduction should cite these sources before stating the gap. The strongest gap statement is:

> Existing work studies matrix-aware optimization in language-model training, optimizer mismatch in
> general fine-tuning, or embedding geometry in retrieval separately; no controlled study traces the
> causal path from matrix update geometry to dense and token-level retrieval behavior.
