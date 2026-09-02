# Muon for dense-retriever adaptation

A reproducible study of AdamW, Muon, and NorMuon for supervised adaptation of
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised).

The project compares complete training dynamics, zero-shot retrieval quality, systems cost, and
weight/update geometry. Its central question is not whether Muon produces flatter update spectra—that
is largely built into the operator—but why a matrix-aware optimizer can lose a matched one-step
functional comparison while showing higher one-seed, four-learning-rate-median final BEIR. The
[generated completion outcome](docs/blog.md#audited-completion-status) records whether the recipe
frozen by query-disjoint validation passes its prospective gates.

The repository remains private until the project owner requests publication. It is structured for a
later public release under Apache-2.0.

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

The completion protocol also restores five-stage retrieval coverage for all 4 routing-matched
hybrid runs and all 9 validation-frozen confirmatory runs. Their 20%–80% checkpoints add 728
strictly isolated task units to the existing 182 final-stage units. The source-bound supplemental
trajectory is defined as 13 runs × 5 stages = 65 rows and 910 task units; together with discovery,
the complete design contains 1,750 DenseOn BEIR units. Stages 1–4 are descriptive dynamics only:
the hybrid and confirmatory inferential summaries continue to read their independently frozen
stage-5 roots.
Shared-start controls instead use five-stage query-disjoint and unseen probes, not BEIR inference.

| Optimizer | Same-suite BEIR-best LR | Exploratory BEIR-best final | Four-LR final mean | Four-LR final median |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 3e-5 | 0.5899 | 0.5816 | 0.5858 |
| Muon | 3e-4 | 0.5923 | 0.5833 | 0.5901 |
| NorMuon | 3e-4 | 0.5934 | 0.5847 | 0.5910 |

The best observed Muon and NorMuon points are +0.0024 and +0.0036 over the best AdamW point.
These are exploratory estimates because BEIR is used for discovery selection. The query-disjoint
validation rule instead selects AdamW 3e-5 (discovery BEIR 0.5899), Muon 3e-3 (0.5608), and NorMuon
3e-3 (0.5634). Three new seeds using recipes frozen by that validation rule provide the confirmatory
comparison.

The strongest current mechanism observation is a local-to-global reversal:

- a Frobenius-matched Muon virtual step improves the mean immediate query margin less than AdamW;
- across the four frozen learning-rate points, Muon's one-seed final median unseen margin and BEIR
  exceed AdamW by 0.0110 and 0.00435;
- in the post-hoc fixed-state intervention, the adverse query tail is redistributed rather than
  uniformly dominated;
- spectral flattening alone has little or no anchor-level association with tail protection.

Shared-start branches and frozen spectrum-versus-basis transplants test whether repeated
optimizer-induced state feedback explains the reversal. See the blog for values and claim
boundaries.

## Final result-driven conclusion

<!-- FINAL-CONCLUSION:BEGIN -->

On the validation-frozen three-seed DenseOn retrieval comparison, Muon versus AdamW was negative (mean delta nDCG@10 -0.0306; familywise 95% CI [-0.0464, -0.0138]), while NorMuon versus AdamW was negative (mean delta nDCG@10 -0.0304; familywise 95% CI [-0.0446, -0.0138]). Across DenseOn's four frozen learning rates, routing-matched hybrid AdamW minus native AdamW averaged +0.0001 nDCG@10, with 3 positive, 1 negative, and 0 zero learning-rate points. This is descriptive evidence about parameter routing as an alternative explanation; it does not by itself identify the matrix rule or prove that routing accounts for the confirmatory Muon-family contrast. The frozen shared-start tail endpoint for DenseOn concluded Muon: mixed; NorMuon: mixed. The frozen temporal spectral bridge was a claimable negative, the fixed-state dose/band chain was a claimable negative, and their joint spectral-component account was a claimable negative. This explains only the tested chain: it does not identify formal mediation or establish a universal optimizer ranking.

<!-- FINAL-CONCLUSION:END -->

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
| Full-length retrieval dynamics | 12 discovery + 4 hybrid + 9 confirmatory runs, five stages each |
| Supplemental inference boundary | Hybrid/confirmatory stages 1–4 descriptive; stage 5 formal |
| Confirmation | Three new negative-sampling/data-order seeds |
| Default compute | Two independent four-GPU pools |

Muon and NorMuon operate on the 88 two-dimensional Transformer hidden matrices
(110,297,088 parameters). Embeddings, pooling projection, norms, and biases use auxiliary AdamW at
3e-6. Hybrid AdamW reproduces that routing and measures its effect within AdamW. Its comparison with
Muon-family runs is a separately tuned, matched-routing recipe comparison—not a scale-matched
identification of orthogonalization alone.

## Repository layout

| Path | Purpose |
| --- | --- |
| configs/experiment.yaml | Frozen historical discovery matrix; use an explicit Dense family filter |
| configs/dense_scope_amendment.json | Active scope and strict expected counts |
| configs/dense_training_queue.json | Frozen 18-run confirmation/short-branch queue |
| configs/dense_retrieval_dynamics_extension.json | Source-bound five-stage hybrid/confirmation extension |
| src/embed_optim/ | Training, evaluation, optimizers, audits, reports, and interventions |
| scripts/eval/dense_parallel.py | Eight-GPU dense retrieval evaluator |
| docs/blog.md | Dense-only Markdown article with generated result blocks |
| paper/ | ACL manuscript and generated result tables |
| reports/ | Content-addressed summaries and publication figures |
| tests/ | Unit, integration, provenance, distribution, and numerical regression tests |

## Installation

Python 3.10–3.13 and CUDA GPUs with bfloat16 support are expected. The portable developer/CI
environment is installed with the checked-in `uv.lock`:

~~~bash
git clone https://github.com/qcznlp/embedding-optimizer-study.git
cd embedding-optimizer-study

uv sync --extra dev --extra eval --extra analysis
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
source .venv/bin/activate
~~~

Formal runs use a separate, hash-locked Python 3.12 / CUDA 12.9 reconstruction environment. This
also reproduces the otherwise intentional FastPlaid/Torch version override present on the experiment
host. Building FlashAttention is a second step because its build imports the already-installed
PyTorch package:

~~~bash
uv venv --python 3.12 .venv-formal
uv pip sync \
  --python .venv-formal/bin/python \
  --no-config \
  --require-hashes \
  --torch-backend cu129 \
  requirements-formal.lock
uv pip install \
  --python .venv-formal/bin/python \
  --no-config \
  --no-deps \
  --require-hashes \
  --no-build-isolation-package flash-attn \
  -r requirements-formal-flash.txt
uv pip install --python .venv-formal/bin/python --no-config --no-deps -e .

.venv-formal/bin/embed-optim-verify-runtime --spec configs/formal_runtime.json
~~~

The verifier also hashes the constraints and both reconstruction locks before checking Python,
PyTorch/CUDA, and every formal package version. Formal training and evaluation must use the
interpreter that passes this check; the ordinary `.venv` is not presented as a formal runtime.

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
  --scope-amendment configs/dense_scope_amendment.json \
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
Each pool holds an exclusive lease, resets its aggregate ledger to `complete=false` before waiting,
and deep-validates all five checkpoint payloads before accepting an existing run. A completed output
that fails that audit is atomically preserved under the sibling `.invalid-completed-runs/` directory
and retrained from a clean output path, so a shallow terminal marker cannot permanently hide damage.
Each matrix command also has a 24-hour process-group watchdog (including its bounded internal
retries); override it conservatively with `--job-timeout-seconds` when a legitimate run needs longer.
GPU lists must contain four unique canonical non-negative integer IDs.

After both queue ledgers complete, run the evaluation/intervention pipeline:

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

The pipeline performs:

1. deep checkpoint audits for hybrid, confirmatory, and short-branch runs;
2. final-stage hybrid BEIR evaluation and summary, followed by its isolated stages 1–4 dynamics;
3. three-seed final-stage confirmatory BEIR evaluation and hierarchical summary, followed by its
   isolated stages 1–4 dynamics;
4. a strict 728-unit extension audit plus a 65-row, 910-unit five-stage trajectory build and audit;
5. all five shared-start branch probes, frozen temporal-predictor extraction and audit, tail summary,
   and temporal short-branch analysis and audit;
6. ten-anchor spectrum/basis transplant, audit, and summary;
7. frozen dose/band analysis and audit; and
8. optional tests, formatting checks, and distribution build.

Every step has an atomic ledger, validated completion predicate, bounded retries, and resume mode.
Do not edit the scope amendment, queue plan, or bound protocols while a run is active.

If this command is launched while the two queue processes are still running, also pass their exact
process IDs as `--wait-pids POOL_A_PID POOL_B_PID`. The process-ID wait is only a convenience: the
completion pipeline always requires exactly two unique ledgers for pools `a` and `b`, verifies both
are complete Dense-only nine-job queues, and rehashes the shared frozen plan and both ledger files.
If a queue, completion step, or host session fails, first recover the queue until both ledgers are
clean and complete, then rerun the same completion command with `--resume`. Resume never trusts an
old completed-step prefix: it reconstructs the current input/source/command contract and executes
the orchestration again from step 1. Individual evaluators may still skip units only after their own
content-addressed audits prove the checkpoint, runtime, and result identity unchanged. This same
full rerun upgrades legacy completion ledgers to the current provenance schema.

## Render the final Dense-only deliverables

After the Dense completion ledger passes, the canonical, resume-safe finalizer regenerates every
scoped report and marker, audits the paper, runs the local quality gates, builds the paper and Python
distributions, and audits the distributions:

~~~bash
embed-optim-dense-finalize \
  --scope-amendment configs/dense_scope_amendment.json \
  --completion-ledger logs/dense-completion-pipeline/pipeline-ledger.json \
  --workdir "$PWD" \
  --include-wandb \
  --resume
~~~

When finalization is started before completion exits, pass the exact completion process ID with
`--wait-pid COMPLETION_PID`. On recovery, rerun the same finalizer command with `--resume` only after
the completion ledger is complete. The finalizer reconstructs the canonical completion commands,
revalidates the current training-plan, pool-ledger, scope, and step-contract provenance, and reruns
its full orchestration rather than trusting an old finalization prefix. If an older completion
ledger lacks those bindings, upgrade it with the completion `--resume` command first.
W&B verification is mandatory for publication completion: the finalizer cannot report a complete
release while offline or while any frozen source run is missing, unfinished, or inconsistent.

For an independent step-by-step audit, the complete ordered finalizer is:

~~~bash
embed-optim-temporal-short-branch-predictors \
  --protocol configs/short_branch_protocol.json \
  --analysis-protocol configs/causal_chain_analysis.json \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --experiment-matrix configs/experiment.yaml \
  --output-csv reports/short-branch/temporal_mechanism_predictors.csv \
  --manifest reports/short-branch/temporal_mechanism_predictors.manifest.json \
  --cache-dir reports/short-branch/temporal-predictor-cache \
  --audit

embed-optim-temporal-short-branch \
  --protocol configs/causal_chain_analysis.json \
  --scope-amendment configs/dense_scope_amendment.json \
  --predictor-csv reports/short-branch/temporal_mechanism_predictors.csv \
  --predictor-manifest reports/short-branch/temporal_mechanism_predictors.manifest.json \
  --outcome-csv reports/tail-stability/short_branch_checkpoint_tail.csv \
  --outcome-manifest reports/tail-stability/summary_manifest.json \
  --output-dir reports/temporal-short-branch \
  --audit

embed-optim-aggregate \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --strict

embed-optim-dose-band-analysis \
  --protocol configs/causal_chain_analysis.json \
  --audit

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

pytest -q
ruff check src tests scripts/eval
ruff format --check src tests scripts/eval
make -C paper release
embed-optim-audit-paper \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
embed-optim-audit-wandb-dense-sources \
  --repository "$PWD" \
  --scope-amendment configs/dense_scope_amendment.json \
  --experiment-matrix configs/experiment.yaml \
  --hybrid-matrix configs/hybrid_adamw.yaml \
  --training-plan configs/dense_training_queue.json \
  --receipt reports/wandb/dense_source_provenance_audit.json
embed-optim-sync-wandb \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
uv build
embed-optim-audit-distribution
~~~

The finalizer first re-audits the temporal predictor and temporal short-branch artifacts against the
frozen causal protocol. It then regenerates the scoped discovery aggregate before re-auditing the
dose/band analysis against that fresh Dense coverage. Retrieval dynamics regenerates the
`TASK-DELTA-STABILITY` block after the aggregate regenerates `RESULTS` and `SYSTEMS`; the later
renderers regenerate the mechanism, outcome, and manuscript artifacts. New hybrid, short-branch,
confirmatory, and intervention manifests must themselves declare families=["dense"] and bind the exact
scope-amendment hash. A partial or mixed-scope report fails closed.

## Weights & Biases

Authenticate outside the repository:

~~~bash
wandb login
~~~

Canonical synchronization uploads only content-verified histories and reads them back before marking
them current. Existing LateOn remote runs are historical; they are retained rather than deleted.
Before that update, the publication finalizer performs a read-only exact provenance audit of all 34
frozen Dense source runs: 12 discovery, 4 hybrid, 9 confirmatory, and 9 shared-start runs. Only after
their full configs, Git metadata, finished state, tags/group, and normalized histories match does it
synchronize and read back the 12 canonical discovery runs. The receipt is written to
`reports/wandb/dense_source_provenance_audit.json` before the distribution build. The final
Dense-only sync must use an explicit family selection and the final report should not count
historical LateOn tags as active confirmation.

~~~bash
embed-optim-sync-wandb \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
~~~

After reviewing a dry run, the historical LateOn canonical runs can be explicitly retired from the
active W&B view without deleting them or touching non-canonical and hybrid runs:

~~~bash
embed-optim-sync-wandb \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --retire-excluded-families \
  --dry-run

embed-optim-sync-wandb \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --retire-excluded-families
~~~

Retirement fails closed unless every selected Dense history and every excluded LateOn canonical
identity/hash exactly matches the frozen matrix. It removes `canonical-current`, adds
`canonical-historical`, records the verified scope amendment in the run summary, and verifies the
remote state afterward. Repeating the command is safe and does not update already-historical runs.

No API key or service credential is stored in source, logs intended for release, build artifacts, or
Git history.

## Verification

Run the full local gate:

~~~bash
pytest -q
ruff check src tests scripts/eval
ruff format --check src tests scripts/eval
make -C paper release
embed-optim-audit-paper \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
uv build
embed-optim-audit-distribution
~~~

CI repeats package build, distribution audit, tests, lint, and formatting. The distribution audit
compares the wheel and source archive against every package module, console entry point, and declared
data file. It also follows executable config-to-config references transitively, so the six frozen
confirmatory/shared-start seed matrices and their formal-runtime dependency cannot disappear from a
release archive. Producer-local generated manifests are immutable provenance receipts rather than
portable executable inputs; they remain in the Git repository, are reported explicitly by the
distribution audit as repository-only provenance, and are not copied into the wheel or source
archive. The Git repository is therefore authoritative for the complete historical hash chain.

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
