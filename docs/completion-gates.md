# DenseOn study completion gates

This checklist defines “complete” for the active Dense-only AdamW/Muon/NorMuon study. Existing
two-family discovery artifacts are preserved and audited first, but only the DenseOn slice can
satisfy primary, causal, or confirmatory gates. The authoritative scope is
[configs/dense_scope_amendment.json](../configs/dense_scope_amendment.json).

A running process, plausible file count, or green unit test is not sufficient. Every accepted gate
must have a complete manifest, exact cardinality, content hashes, and the same scope-amendment
identity.

| Requirement | Authoritative evidence | DenseOn passing condition |
| --- | --- | --- |
| Scope | configs/dense_scope_amendment.json | families=["dense"]; historical Late is retained, not used in primary inference |
| Shared data | materialized 500K manifest and row ledger | 500,000 rows, seven source quotas, seven distinct seeded negatives, one training-view fingerprint |
| Discovery training | configs/experiment.yaml plus completion/checkpoint audit | 12 runs, 60 checkpoints, all model/optimizer/scheduler/Trainer/RNG payloads deep-valid |
| Discovery evaluation | [reports/dense-discovery/coverage.json](../reports/dense-discovery/coverage.json) and provenance-valid MTEB files | 12 × 5 × 14 = 840 DenseOn units selected only after the full historical source contract passes |
| Training dynamics | reports/training-dynamics manifests | 12 DenseOn run rows and 60 five-stage rows after strict full-source audit and filtering |
| Retrieval dynamics | reports/retrieval-dynamics-dense/summary_manifest.json | 60 checkpoints, 840 source units, 12 first-passage rows, 8 adjacent-stage task-stability rows |
| Weight trajectories | reports/weight-space/summary_manifest.json | 12 runs, 60 checkpoints, 20 matched Muon/NorMuon checkpoint pairs |
| Common state | reports/common-state/summary_manifest.json | 10 DenseOn anchors and all three transforms at every anchor |
| Exact spectra | results/common-state-spectra/summary/summary_manifest.json | 180 spectra: 10 anchors × 3 optimizers × 6 selected matrices |
| Basis sensitivity | reports/basis-sensitivity/summary_manifest.json | 270 full-tensor and 1,620 selected-head DenseOn records |
| Functional intervention | reports/functional-intervention/manifest.json | all 10 DenseOn anchors, conditions, query pairs, scales, signs, and hashes valid |
| Hybrid routing | reports/hybrid-adamw/summary_manifest.json | 4 runs and 56 final BEIR units, families=["dense"], scope hash matches |
| Confirmatory training | generated matrices and deep checkpoint audits | 3 seeds × 3 optimizers = 9 terminal runs and 45 checkpoints |
| Confirmatory evaluation | reports/confirmatory/summary_manifest.json | 9 × 14 = 126 units; seed-by-task bootstrap and the original six-comparison Bonferroni family remain valid while only Dense rows are shown |
| Extended Dense retrieval dynamics | reports/dense-retrieval-dynamics/summary_manifest.json | 13 runs × 5 stages = 65 rows and 910 task units: 728 isolated stage-1–4 units plus 182 formal stage-5 units; stages 1–4 are descriptive only |
| Shared-start training | generated matrices and deep checkpoint audits | 3 seeds × 3 operators = 9 runs and 45 checkpoints from one frozen AdamW start |
| Shared-start probes | reports/short-branch/summary_manifest.json | 45 query-disjoint checkpoint rows plus 46 unseen-probe jobs (45 checkpoints and one pretrained reference), with the three-seed endpoint decision |
| Temporal shared-start mechanism | reports/short-branch/temporal_mechanism_predictors.manifest.json and reports/temporal-short-branch/summary_manifest.json | Predictor extraction and the scope-bound temporal analysis are each complete; both strict `--audit` steps rehash every declared input and output, and tail-stability outcomes are built before the temporal join |
| Tail summary | reports/tail-stability/summary_manifest.json | discovery tier labeled post hoc; accumulated tier satisfies the fixed joint tail rule |
| Spectrum/basis transplant | reports/spectral-transplant/summary_manifest.json | all conditions at 10 anchors; native, 2×2, path, and band summaries complete |
| Dose/band mechanism | reports/dose-band/summary_manifest.json | Dense-only, scope-bound dose/band analysis is complete; its strict `--audit` rehashes every declared input and output |
| Mechanism report | reports/mechanism-summary.manifest.json | families=["dense"], same scope hash, 10 anchors, 180 spectra, 60 bridge checkpoints, 840 retrieval units |
| Outcome report | reports/outcome-summary.manifest.json | hybrid, shared-start, spectral transplant, and confirmation are all strict and rendered into the blog |
| Blog | [docs/blog.md](blog.md) | each generated marker has exactly one pair and matches its hashed report byte-for-byte |
| Paper | reports/paper-results.manifest.json and paper/ | Dense constants are 12/60/840/20; no pending macro; strict paper audit, strict ACL release build, and the post-build strict re-audit pass |
| W&B | reports/wandb/dense_source_provenance_audit.json plus remote content-addressed histories | exact config, Git, finished-state, tags/group, and normalized history for all 34 frozen Dense source runs (12 discovery + 4 hybrid + 9 confirmatory + 9 shared-start), followed by exactly 12 Dense canonical discovery identities; historical Late runs are retained and clearly tagged |
| Distribution | wheel, sdist, tests, CI | tests/lint/format, PDF, package build, and distribution audit all pass |
| Publication hygiene | tracked tree, Git history, distributions, GitHub settings | no credentials; repository remains private until the user requests publication |

## Automated compute pipeline

After both nine-job Dense training queues complete, run:

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
  --resume
~~~

The ledger under logs/dense-completion-pipeline must finish every checkpoint audit, evaluation,
the 728-unit hybrid/confirmatory dynamics extension, its 65-row five-stage summary, short-branch
summary, temporal predictor extraction, tail summary, temporal short-branch analysis,
spectral-transplant step, and dose/band analysis. Each new build is followed immediately by its
strict `--audit`; a pending
receipt, failed audit, or incomplete step blocks final reporting. The commands and source hashes are
part of the completion ledger's step contract, and the finalizer binds its provenance to the exact
completed ledger before rendering the mechanism report, blog, and paper.

The dynamics extension is frozen in
[`configs/dense_retrieval_dynamics_extension.json`](../configs/dense_retrieval_dynamics_extension.json).
It writes stages 1–4 to separate result roots, joins them to the already formal stage-5 results only
for the descriptive trajectory, and never changes the hybrid-routing or confirmatory inference
inputs. The resulting CSV and publication figures are
`reports/dense-retrieval-dynamics/five_stage_retrieval_dynamics.{csv,pdf,svg}`.

The machine-readable integration contract is
[`configs/mechanism_analysis_integration.json`](../configs/mechanism_analysis_integration.json).
It fixes the module names, output manifests, ordering dependencies, Dense-only scope, and audit flag.

Each training-pool queue has a non-blocking exclusive lease and clears its aggregate completion bit
at entry. It may adopt or skip an on-disk run only after an uncached deep audit of all five scheduled
checkpoint payloads. If a terminal run fails that audit, the entire output is moved atomically to a
sibling `.invalid-completed-runs/` evidence directory before a clean rerun. The default per-command
watchdog is 24 hours and terminates the complete child process group after a grace period; extending
it requires the explicit `--job-timeout-seconds` option.

When the queues are still running, add the exact queue process IDs as
`--wait-pids POOL_A_PID POOL_B_PID`. This is a wait guard, not evidence of completion: the command
also requires exactly two unique ledgers, one each for pools `a` and `b`, and verifies Dense family,
nine complete jobs per pool, the common frozen-plan hash, scope identity, and exact ledger-content
hashes. For recovery, first resume the failed queue until both queue ledgers are clean and complete,
then rerun the command above with `--resume`. Resume always rebuilds the current input/source/command
contract and reruns orchestration from step 1; it never trusts a completed prefix from the old
ledger. Content-addressed evaluators may independently skip only units whose checkpoint, runtime,
and result identities still pass their strict audits. The same full rerun upgrades older completion
ledgers to the current per-step provenance schema.

## Final independent audit

Run the canonical resume-safe finalizer only after the compute ledger is complete:

~~~bash
embed-optim-dense-finalize \
  --scope-amendment configs/dense_scope_amendment.json \
  --completion-ledger logs/dense-completion-pipeline/pipeline-ledger.json \
  --workdir "$PWD" \
  --include-wandb \
  --resume
~~~

If finalization is queued behind a still-running completion process, add its exact process ID with
`--wait-pid COMPLETION_PID`. If either process fails, recover completion first and then rerun the
finalizer with `--resume`. The finalizer revalidates current upstream provenance even when its own
ledger says `complete`; every resume reruns the full canonical finalization orchestration instead of
reusing an old completed prefix.
If an older completion ledger is rejected for missing provenance, upgrade it with completion
`--resume` before retrying finalization.
W&B verification is a mandatory publication gate. The compatibility flag is shown explicitly in
the canonical command, but the finalizer cannot produce a publication-complete ledger without the
read-only 34-source audit and the 12-run canonical synchronization/readback audit.

For independent inspection, its complete ordered finalization sequence is:

~~~bash
python -m embed_optim.temporal_short_branch_predictors \
  --protocol configs/short_branch_protocol.json \
  --analysis-protocol configs/causal_chain_analysis.json \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --experiment-matrix configs/experiment.yaml \
  --output-csv reports/short-branch/temporal_mechanism_predictors.csv \
  --manifest reports/short-branch/temporal_mechanism_predictors.manifest.json \
  --cache-dir reports/short-branch/temporal-predictor-cache \
  --audit

python -m embed_optim.temporal_short_branch \
  --protocol configs/causal_chain_analysis.json \
  --scope-amendment configs/dense_scope_amendment.json \
  --predictor-csv reports/short-branch/temporal_mechanism_predictors.csv \
  --predictor-manifest reports/short-branch/temporal_mechanism_predictors.manifest.json \
  --outcome-csv reports/tail-stability/short_branch_checkpoint_tail.csv \
  --outcome-manifest reports/tail-stability/summary_manifest.json \
  --output-dir reports/temporal-short-branch \
  --audit

python -m embed_optim.aggregate \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --strict

python -m embed_optim.dose_band_analysis \
  --protocol configs/causal_chain_analysis.json \
  --audit

python -m embed_optim.retrieval_dynamics \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

python -m embed_optim.mechanism_report \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

python -m embed_optim.outcome_report \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

python -m embed_optim.paper_results \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

python -m embed_optim.paper_audit \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json

pytest -q
ruff check src tests scripts/eval
ruff format --check src tests scripts/eval
make -C paper release
python -m embed_optim.paper_audit \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
python -m embed_optim.wandb_dense_provenance_audit \
  --repository "$PWD" \
  --scope-amendment configs/dense_scope_amendment.json \
  --experiment-matrix configs/experiment.yaml \
  --hybrid-matrix configs/hybrid_adamw.yaml \
  --training-plan configs/dense_training_queue.json \
  --expected-git-remote https://github.com/qcznlp/embedding-optimizer-study.git \
  --receipt reports/wandb/dense_source_provenance_audit.json
python -m embed_optim.wandb_sync \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
uv build
python -m embed_optim.distribution_audit
~~~

Authenticate to W&B outside the repository. The source audit is read-only and runs before canonical
discovery synchronization, so provenance failure occurs before any remote update. Its secret-free,
self-hashed 34-run receipt is packaged by the subsequent distribution build. Do not delete
historical Late runs.

Finally verify:

- the GitHub repository is still private;
- CI belongs to the final commit;
- the blog, paper, manifests, built archives, and W&B histories all identify the same Dense scope;
- a secret-pattern scan of the tracked tree, reachable Git history, and distributions returns no
  credentials;
- remaining mentions of LateOn are limited to transparent historical-scope disclosure or source
  titles.
