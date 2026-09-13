<!--
This README substantially modifies the lightonai/mdenseon-mlateon README at commit
b0db47a48f969d825446668b5b17bfc27a359fc1. See THIRD_PARTY_NOTICES.md.
-->

# Optimizers, weight trajectories, and dense retrieval

How do AdamW, Muon and NorMuon change a pretrained retriever's weights, and
which changes matter for retrieval? This DenseOn study connects **weight
trajectories, functional representation utility and held-out retrieval**.
The target is a NAACL paper with reproducible model and analysis artifacts.

<!-- FINAL-CONCLUSION:BEGIN -->
**Results pending (FINAL_CONCLUSION_PENDING).** The paper's optimizer conclusions await complete
retrieval, functional and crossed-continuation evidence. Historical results, numerical diagnostics
and passing tests do not substitute for those findings.
<!-- FINAL-CONCLUSION:END -->

This is the **local development checkout**, not a completed source release.
For current work and exact running handles, read [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md);
agents must also follow [AGENTS.md](AGENTS.md). Dated evidence is retained in
[PROJECT_STATUS.md](PROJECT_STATUS.md).

## Current progress

Snapshot: **2026-09-13**, not a perpetual heartbeat.

Current status and agent instructions now contain only the active handoff;
the full chronological records remain in the
[preserved history](reports/engineering-archive/dense-v3-portable-handoff-v1/README.md).

Read the [complete-result paper PDF](reports/paper-review/dense-v3-complete-manuscript-revision-v1/actual/paper/build/main.pdf)
and [revision notes](reports/paper-review/dense-v3-complete-manuscript-revision-v1/README.md).
This reviewed local candidate contains the complete results, revised story and
closest references: eight-page main text, 158-word abstract, all 13 pages
visually inspected. Authoritative source installation/release remains pending;
the legacy release marker above has not been retired by a source-level gate.

The complete manuscript now has a [stable source/build entry](paper/current/README.md):

```bash
make -C paper current CURRENT_OUTPUT=/absolute/path/to/new-paper-build
```

The actual Make target and an extracted-wheel rebuild both pass; the latter
loads all study modules from the wheel and reproduces the reviewed PDF text.
See [verification and remaining release findings](reports/engineering-archive/dense-v3-current-paper-entry-v1/README.md).

| Component | Verified state |
| --- | --- |
| Primary experiment | 12/12 runs, five stages each; 60 checkpoints backed up and HF hash-verified |
| Primary retrieval | 840/840 checkpoint/task evaluations; 14/14 pretrained-baseline tasks; validation-only selection complete |
| Weight and functional analysis | Complete 60-state geometry/prediction panel and 61-state functional panel including the pretrained reference |
| Crossed state-by-operator continuations | 12/12 runs and 60 additional checkpoints complete and backed up; 60/60 five-stage probes complete |
| Continuation full-corpus retrieval | **168/168 complete**, all workers and both coordinators exit zero |
| Complete continuation inference and paper | All six tables/three contrasts verified; full numerical/PDF replay passed; new complete-result prose/reference revision built and all 13 pages visually reviewed |
| Current-paper source/build | Stable `paper/current` and package entry implemented; 86 focused tests pass; actual Make and extracted-wheel builds pass |
| Complete numerical-to-reviewed-paper command | Version-isolated installed-wheel execution passes, joining all eight shared inputs; 194 integration tests pass; [run instructions](docs/versioned-paper-reproduction.md) |
| Distribution portability | Original full audit passes; real relocated input read is unchanged; historical failures and all negative controls retained |
| Primary training source | All 33 modules and 56 bindings match the actual training snapshot; all 12 run sources load in this checkout; [inspection instructions](docs/current-training-source.md) |
| Recovery and remaining delivery | Both genuine GPU restores pass exact endpoints; scoped recovery add-on and versioned paper consumers work; final release-parent transition and source publication remain |

All scientific training/evaluation and the statistical, document, numerical/PDF
reconstruction and result-backup pipelines have finished. Recovery verification
and final paper/source delivery remain. No scientific training/evaluation is
running. Both bounded [warm-reducer restoration checks](reports/engineering-archive/dense-v3-warm-reducer-recovery-v1/README.md)
now pass bitwise endpoint equality and are terminal. Do not restart completed work.
The [earlier diagnosis](reports/engineering-archive/dense-v3-resume-endpoint-diagnosis-v1/README.md)
retains both original failed comparisons; neither attempt replaces scientific states.
[Operational observations](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
retain actual process exits and current-task timing evidence.

## What the completed primary results support

Averaged over all four declared learning rates, the primary optimizer
comparisons are inconclusive. Validation-selected Muon and NorMuon have higher
retrieval point estimates than selected AdamW at all five retained stages.
At the final stage, selected NorMuon–AdamW is **+0.437 nDCG@10 points**
(simultaneous 95% interval **[+0.146, +0.729]**); selected Muon–AdamW is
**+0.317**, with its interval crossing zero. These are paired-task intervals
within each estimand family, **not independent-training-seed intervals**.
Primary and validation-selected comparisons answer different questions.
See [all six endpoint contrasts](reports/dense-v3-final-inference-v1/README.md)
and [all rates and stages](reports/dense-v3-complete-trajectories-v1/README.md).

![All twelve measured retrieval trajectories, retaining every declared learning rate](reports/dense-v3-complete-trajectories-v1/figures/retrieval_trajectories.png)

Stars mark validation selection; lines connect measured checkpoints. Crossing
AdamW's final score earlier does not establish deployable wall-time savings:
selection used full-horizon validation.

The weight/functional analyses test explanations, not just differences in
spectra. Displacement and degrading-attribution mass predict retrieval under
the original baseline, but **no tested weight or functional predictor passes
all four exploratory recipe comparators**. The Muon–AdamW helpful-participation
contrast is not stable across the sampled rotations. These findings do
not establish “Muon uses more useful dimensions” or a causal gain mechanism.
Read the [weight controls](reports/dense-v3-predictor-sensitivity-v1/README.md),
[functional inference](reports/dense-v3-functional-inference-v1/README.md) and
[functional controls](reports/dense-v3-functional-sensitivity-v1/README.md).
The complete crossed continuation now separates reached state from subsequent
update rule under its fixed 50K-query design:

| Contrast | nDCG@10 points | Marginal 95% interval |
| --- | ---: | ---: |
| Muon-source minus AdamW-source state, averaged over continuation rules | +0.322 | [+0.047, +0.678] |
| Reset Muon minus reset AdamW, averaged over source states | -0.526 | [-0.897, -0.186] |
| State-by-operator interaction | +0.208 | [-0.123, +0.555] |

These are three marginal intervals from the declared two-way seed/task
bootstrap, not simultaneous intervals or independent primary-source replications.
The result supports a distinction between the two reached weight states and
the locally preferred continuation rule; it does not establish mediation or
a universal optimizer ranking. [Complete native evidence and scope](reports/engineering-archive/dense-v3-final-evaluation-closeout-v1/README.md).

## Experiment design

| Choice | Primary experiment |
| --- | --- |
| Initialization | [DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised), revision `0edbd55684eb782bce55ee74c95b25c97cbe7f43` |
| Data and objective | Same deterministic 500,000 queries; one positive and seven hard negatives; no in-batch negatives; temperature 0.02 |
| Schedule | One epoch, global batch 128, maximum context 8192 |
| AdamW learning rates | 1e-6, 3e-6, 1e-5, 3e-5 |
| Muon / NorMuon learning rates | 1e-4, 3e-4, 1e-3, 3e-3 |
| Retained steps | 782, 1563, 2345, 3126, 3907 |
| Evaluation and selection | Fourteen pinned decontaminated BEIR tasks; separate frozen 4,096-query validation loss |
| Replication scope | One primary training seed (42); learning-rate cells are not independent seeds |
| Actual training hardware | Eight NVIDIA L20Z GPUs, split into two disjoint four-GPU pools; not literal H100 hardware |

The [v3 primary protocol](configs/dense_primary_v3_protocol.json) binds data,
model, runtime and checkpoint identities. This compares complete optimizer
recipes: the primary AdamW rate varies across all its parameters, whereas
Muon/NorMuon use auxiliary AdamW at 3e-6. AdamW's validation-selected rate is
the upper tested grid boundary, not an established optimum.

The functional probe uses 224 fixed queries across fourteen tasks, eight
candidates per query, 768-coordinate deletion with cosine re-normalization,
shared random masks and three sampled rotations. It is **not full-corpus
compressed retrieval**, and sampled rotations do not establish arbitrary-basis
invariance.

The [crossed continuation design](docs/state-operator-factorial.md) uses two
genuine source states × two reset optimizer rules × three data-order seeds:
12 continuations on the same fixed 50k queries. Initial hidden-update
calibration is not trajectory-wide norm matching, and the three order seeds
are not three independently trained source states. Only DenseOn is active;
the [scope amendment](configs/dense_scope_amendment.json) preserves historical
LateOn artifacts without new LateOn experiments or pooled architecture claims.

## Recover artifacts and reproduce analyses

| Need | Start here |
| --- | --- |
| Primary checkpoint weights and optimizer state | [Immutable checkpoint restoration](docs/checkpoint-restoration.md) and [all sixty primary locations](docs/primary-v3-checkpoints.json) |
| Continuation checkpoints and five-stage probes | [Continuation restoration guide and checkpoint index](docs/continuation-probe-restoration.md) |
| Inspect an already restored continuation save | [CPU-only authenticated checkpoint reader](docs/saved-factorial-checkpoints.md); original identities retained, not GPU-resume validation |
| Resume either of the two verified continuation saves | [Scoped restoration add-on](docs/training-restoration.md); requires the original admitted four-rank worker, not a general new-host launcher |
| Complete continuation retrieval and statistical tables | [Immutable 593-file outcome snapshot and restoration](docs/continuation-outcomes-restoration.md) |
| Full primary retrieval trajectories | [Retrieval tables, raw-score snapshots and figures](docs/retrieval-trajectories-restoration.md) |
| Weight / functional measurements | [Weight-analysis restoration](docs/weight-analysis-restoration.md), [functional/calibration restoration](docs/functional-analysis-restoration.md) |
| Complete primary + continuation numerical graph and original full PDF | [Paper-results reproduction](docs/paper-results-reproduction.md#complete-paper-replay); unchanged local closed bundle, separate reviewed-prose build |
| Complete numerical graph joined to the reviewed manuscript in one command | [Version-isolated paper reproduction](docs/versioned-paper-reproduction.md); original numerical source and current document package both actually executed |
| Manuscript and acceptance requirements | [Paper directory](paper/README.md), [completion gates](docs/completion-gates.md) |

Public data/model repositories are
[checkpoints](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints)
and [analysis artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts).
Use the guides' **immutable revisions and digests**, not mixed historical
namespaces. Some source/replay bundles remain local unpublished WIP; a clean
remote clone is not yet guaranteed to contain them. Numerical reconstruction
is not a fresh encoding, GPU-training replay or physical second-host proof.

## Development and release

Development only, in a **separate environment**:

```bash
uv sync --extra dev --extra eval --extra analysis
export PYTHONPATH="$PWD/src:$PWD"
CUDA_VISIBLE_DEVICES='' uv run python -B -m pytest -q
```

Do not install or upgrade packages in the live experiment environment.
Formal execution uses [formal_runtime.json](configs/formal_runtime.json),
[requirements-formal.lock](requirements-formal.lock) and
[requirements-formal-flash.txt](requirements-formal-flash.txt).
Current execution authority and exact source bindings are in the agent handoff;
archived controller commands are not current launch instructions.

The [reviewed current paper](reports/paper-review/dense-v3-complete-manuscript-revision-v1/actual/paper/build/main.pdf)
contains both primary and continuation findings; its prose revision, complete
visual review and document reconstruction have passed. The original full
[distribution audit now passes](reports/engineering-archive/dense-v3-portable-input-roles-v1/README.md).
Final release still requires the unified scientific-source/release-parent transition. Do not remove
pending markers, discard failed evidence or change hashes to bypass these
requirements. No engineering-error narrative belongs in the manuscript.

The [former homepage](reports/engineering-archive/dense-v3-reader-first-readme-v1/before/README.md)
preserves the detailed chronology and earlier reproduction/test records.
[AGENTS.md](AGENTS.md) retains all source, process, external-access and
publication boundaries. Tracking is available through
[W&B](https://wandb.ai/stevezenguom/embedding-optimizer-study) and the
[project issue](https://github.com/qcznlp/embedding-optimizer-study/issues/41);
neither substitutes for native completion evidence. Never put credentials in
source or released logs.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
