# Muon for dense-retriever adaptation

A reproducible study of AdamW, Muon, and NorMuon for supervised adaptation of
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised).

The project compares complete training dynamics, zero-shot retrieval quality, systems cost, and
weight/update geometry. Its central question is not whether Muon produces flatter update spectra—that
is largely built into the operator—but why a matrix-aware optimizer can lose a matched one-step
functional comparison and still finish with a better dense retriever.

The repository is private while the remaining DenseOn confirmation is running. It is structured for
a later public release under Apache-2.0.

- Full study write-up: [docs/blog.md](docs/blog.md)
- Dense-only NAACL plan: [docs/naacl-dense-paper-plan.md](docs/naacl-dense-paper-plan.md)
- Result-safe ACL manuscript: [paper/](paper/README.md)
- Live dashboard: [Weights & Biases](https://wandb.ai/stevezenguom/embedding-optimizer-study)

## Scope disclosure

The original discovery matrix contained DenseOn and LateOn. After discovery and exploratory
mechanism outputs were visible, the project owner directed all new work to DenseOn because
late-interaction training was much slower and less relevant to the intended audience.

That is a user-directed, post-hoc scope amendment. Existing LateOn configurations, checkpoints,
logs, and results are retained for audit, but they are not used for primary inference, replication,
or confirmation. The decision and expected Dense-only counts are frozen in
[configs/dense_scope_amendment.json](configs/dense_scope_amendment.json).

The earlier [two-family paper plan](docs/naacl-paper-plan.md) remains byte-for-byte frozen because it
is part of the original claim-protocol hash chain. It is historical, not the active paper plan.

## Current evidence

The complete 12-run DenseOn discovery sweep contains four learning rates for each optimizer and five
evaluated checkpoints per run: 60 checkpoints and 840 decontaminated BEIR task units.

| Optimizer | Best LR | Best final mean nDCG@10 | Four-LR final mean |
| --- | ---: | ---: | ---: |
| AdamW | 3e-5 | 0.5899 | 0.5816 |
| Muon | 3e-4 | 0.5923 | 0.5833 |
| NorMuon | 3e-4 | 0.5934 | 0.5847 |

The best observed Muon and NorMuon points are +0.0024 and +0.0036 over the best AdamW point.
These are exploratory estimates because BEIR is used for discovery selection. Three new
validation-frozen seeds provide the confirmatory comparison.

The strongest current mechanism observation is a local-to-global reversal:

- a Frobenius-matched Muon virtual step improves the mean immediate query margin less than AdamW;
- the full Muon trajectory has a better median unseen margin and BEIR point estimate;
- in the post-hoc fixed-state intervention, the adverse query tail is redistributed rather than
  uniformly dominated;
- spectral flattening alone has little or no anchor-level association with tail protection.

Shared-start branches and frozen spectrum-versus-basis transplants test whether repeated
optimizer-induced state feedback explains the reversal. See the blog for values and claim
boundaries.

## Experimental contract

| Variable | Value |
| --- | --- |
| Base checkpoint | lightonai/DenseOn-unsupervised, pinned revision |
| Optimizers | AdamW, Muon, NorMuon |
| Discovery sweep | Four learning rates per optimizer |
| Training examples | Identical deterministic 500,000-query view |
| Contrastive group | One positive plus seven seeded random hard negatives |
| In-batch negatives | Disabled |
| Context length | 8,192 query and document tokens |
| Objective | Cosine InfoNCE, temperature 0.02 |
| Epochs / nominal global batch | One / 128 |
| Checkpoints | 20%, 40%, 60%, 80%, and 100% |
| Evaluation | 14 pinned decontaminated BEIR tasks, nDCG@10 |
| Confirmation | Three new negative-sampling/data-order seeds |
| Default compute | Two independent four-GPU pools |

Muon and NorMuon operate on the 88 two-dimensional Transformer hidden matrices
(110,297,088 parameters). Embeddings, pooling projection, norms, and biases use auxiliary AdamW at
3e-6. Hybrid AdamW controls reproduce that routing so optimizer effects are not confused with the
parameter-group recipe.

## Repository layout

| Path | Purpose |
| --- | --- |
| configs/experiment.yaml | Frozen historical discovery matrix; use an explicit Dense family filter |
| configs/dense_scope_amendment.json | Active scope and strict expected counts |
| configs/dense_training_queue.json | Frozen 18-run confirmation/short-branch queue |
| src/embed_optim/ | Training, evaluation, optimizers, audits, reports, and interventions |
| scripts/eval/dense_parallel.py | Eight-GPU dense retrieval evaluator |
| docs/blog.md | Dense-only Markdown article with generated result blocks |
| paper/ | ACL manuscript and generated result tables |
| reports/ | Content-addressed summaries and publication figures |
| tests/ | Unit, integration, provenance, distribution, and numerical regression tests |

## Installation

Python 3.10–3.13 and CUDA GPUs with bfloat16 support are expected. Formal runs record Python 3.12,
PyTorch 2.9.1+cu129, SentenceTransformers 5.7, FlashAttention 2.7.4.post1, and four GPUs per job.

~~~bash
git clone https://github.com/qcznlp/embedding-optimizer-study.git
cd embedding-optimizer-study

uv sync --extra dev --extra eval --extra analysis
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
source .venv/bin/activate

embed-optim-verify-runtime --spec configs/formal_runtime.json
~~~

The checked-in uv lock is the portable development/CI environment. Formal training and evaluation
must use the interpreter that passes the pinned runtime check.

Late-interaction packages remain dependencies only so the historical discovery artifacts and code
paths can be audited. They are not used by the active DenseOn pipeline.

## Data preparation

The source contains about 1.22 million queries. The builder intersects query and score tables,
allocates 500,000 rows proportionally across seven sources, and deterministically selects seven hard
negatives from the first ten eligible candidates.

~~~bash
embed-optim-prepare
embed-optim-prepare --audit-only
~~~

The audit checks source revisions, source quotas, every selected document ID, the dataset
fingerprint, the row-ledger hash, and the exact training-view fingerprint. Every formal run repeats
the view check before loading the model.

## Run the DenseOn discovery sweep

The historical matrix still contains archival LateOn definitions. Always pass the explicit family
filter:

~~~bash
embed-optim-matrix \
  --matrix configs/experiment.yaml \
  --families dense \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7 \
  --max-retries 2
~~~

Each accepted run writes complete checkpoints at five declared steps. Audit them before evaluation:

~~~bash
embed-optim-watch-checkpoints \
  --matrix configs/experiment.yaml \
  --families dense \
  --fail-on-problem
~~~

## Evaluate all discovery checkpoints

~~~bash
embed-optim-evaluate \
  --matrix configs/experiment.yaml \
  --families dense \
  --stages 1 2 3 4 5 \
  --gpus-a 0,1,2,3,4,5,6,7 \
  --gpus-b 4,5,6,7
~~~

Dense evaluation uses MTEB exact retrieval over the 14 pinned decontaminated datasets. Results are
accepted only if checkpoint identity, dataset revision, split, score field, package runtime, and
worker-source hashes match the immutable evaluation manifest.

## Run the Dense-only completion study

Generate the confirmatory and short-branch matrices from their frozen protocols, then launch the two
resumable four-GPU queues:

~~~bash
embed-optim-prepare-confirmatory-data
embed-optim-generate-confirmatory-matrices
embed-optim-short-branch

embed-optim-family-training-queue \
  --pool a \
  --gpus 0,1,2,3 \
  --port 30110

embed-optim-family-training-queue \
  --pool b \
  --gpus 4,5,6,7 \
  --port 30120
~~~

The queue plan contains exactly nine DenseOn jobs per pool: nine confirmatory full runs and nine
50K shared-start runs in total. It is content-bound to the generated matrices and safe to resume.

After both queue ledgers complete, run the evaluation/intervention pipeline:

~~~bash
embed-optim-dense-completion \
  --scope-amendment configs/dense_scope_amendment.json \
  --workdir "$PWD" \
  --gpus 0,1,2,3,4,5,6,7 \
  --gpus-b 4,5,6,7 \
  --include-validation
~~~

The pipeline performs:

1. deep checkpoint audits for hybrid, confirmatory, and short-branch runs;
2. final-stage hybrid BEIR evaluation and summary;
3. three-seed confirmatory BEIR evaluation and hierarchical summary;
4. all five shared-start branch probes and tail summary;
5. ten-anchor spectrum/basis transplant, audit, and summary;
6. optional tests, formatting checks, and distribution build.

Every step has an atomic ledger, validated completion predicate, bounded retries, and resume mode.
Do not edit the scope amendment, queue plan, or bound protocols while a run is active.

## Render the final Dense-only deliverables

After the Dense completion ledger passes, the canonical, resume-safe finalizer regenerates every
scoped report and marker, audits the paper, runs the local quality gates, builds the paper and Python
distributions, and audits the distributions:

~~~bash
embed-optim-dense-finalize \
  --scope-amendment configs/dense_scope_amendment.json \
  --workdir "$PWD" \
  --resume
~~~

For an independent step-by-step audit, the reporting portion of that finalizer is:

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
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

make -C paper clean all
~~~

The aggregate and retrieval-dynamics steps first validate the complete historical discovery sources,
then filter DenseOn and regenerate the `RESULTS`, `SYSTEMS`, and `TASK-DELTA-STABILITY` blog blocks.
The later renderers regenerate the mechanism, outcome, and manuscript artifacts. New hybrid,
short-branch, confirmatory, and intervention manifests must themselves declare families=["dense"] and
bind the exact scope-amendment hash. A partial or mixed-scope report fails closed.

## Weights & Biases

Authenticate outside the repository:

~~~bash
wandb login
~~~

Canonical synchronization uploads only content-verified histories and reads them back before marking
them current. Existing LateOn remote runs are historical; they are retained rather than deleted.
The final Dense-only sync must use an explicit family selection and the final report should not count
historical LateOn tags as active confirmation.

~~~bash
embed-optim-sync-wandb --families dense
~~~

No API key or service credential is stored in source, logs intended for release, build artifacts, or
Git history.

## Verification

Run the full local gate:

~~~bash
pytest -q
ruff check src tests scripts/eval
ruff format --check src tests scripts/eval
make -C paper clean all
uv build
embed-optim-audit-distribution
~~~

CI repeats package build, distribution audit, tests, lint, and formatting. The distribution audit
compares the wheel and source archive against every package module, console entry point, and declared
data file.

## Reproducibility and integrity

The study uses fail-closed, content-addressed contracts:

- model and dataset revisions are pinned;
- data selection and negative sampling are deterministic;
- all formal checkpoints include model, optimizer, scheduler, trainer, and rank-local RNG state;
- checkpoint payloads are loaded and shape/finite-value checked, not merely tested for existence;
- training history is reconstructed from non-overlapping accepted segments after resumption;
- failed or superseded histories remain quarantined and never enter aggregation;
- evaluation results bind task revision, split, subset, checkpoint, runtime, and worker code;
- report manifests bind every source table and disclose whether an analysis is prospective,
  post-hoc, descriptive, or causal;
- Dense-only reports bind the user-directed scope amendment.

Muon and NorMuon use an unfused-bfloat16-v1 Newton–Schulz decomposition after the native bfloat16
addmm path produced cross-device CUDA/cuBLAS failures in long distributed runs. The replacement
preserves the polynomial, precision, coefficients, momentum, and update norm while avoiding the
failing operation decomposition. Native histories are quarantined; all accepted matrix-aware runs
restart from the common base.

## Historical LateOn archive

LateOn code and artifacts remain because deleting them would damage provenance. They may be used to
audit the completed discovery phase, but the active project does not:

- launch new LateOn training or evaluation;
- use LateOn as confirmatory evidence;
- estimate an architecture interaction;
- pool LateOn units with DenseOn uncertainty;
- present MaxSim or token-utilization findings as part of the main story.

A future maintainer who intentionally reproduces the historical two-family matrix should follow the
frozen original protocols, not the Dense-only quickstart above.

## Citation and governance

Citation metadata is in [CITATION.cff](CITATION.cff). Contributions follow
[CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security issues
should follow [SECURITY.md](SECURITY.md). Third-party licenses and pinned reference implementations
are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
