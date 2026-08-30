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
| Discovery evaluation | reports/coverage.json and provenance-valid MTEB files | 12 × 5 × 14 = 840 DenseOn units selected only after the full historical source contract passes |
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
| Shared-start training | generated matrices and deep checkpoint audits | 3 seeds × 3 operators = 9 runs and 45 checkpoints from one frozen AdamW start |
| Shared-start probes | reports/short-branch/summary_manifest.json | 45 query-disjoint checkpoint rows plus 46 unseen-probe jobs (45 checkpoints and one pretrained reference), with the three-seed endpoint decision |
| Tail summary | reports/tail-stability/summary_manifest.json | discovery tier labeled post hoc; accumulated tier satisfies the fixed joint tail rule |
| Spectrum/basis transplant | reports/spectral-transplant/summary_manifest.json | all conditions at 10 anchors; native, 2×2, path, and band summaries complete |
| Mechanism report | reports/mechanism-summary.manifest.json | families=["dense"], same scope hash, 10 anchors, 180 spectra, 60 bridge checkpoints, 840 retrieval units |
| Outcome report | reports/outcome-summary.manifest.json | hybrid, shared-start, spectral transplant, and confirmation are all strict and rendered into the blog |
| Blog | [docs/blog.md](blog.md) | each generated marker has exactly one pair and matches its hashed report byte-for-byte |
| Paper | reports/paper-results.manifest.json and paper/ | Dense constants are 12/60/840/20; no pending macro; strict paper audit and ACL build pass |
| W&B | remote content-addressed histories | exactly 12 Dense canonical discovery identities; historical Late runs are retained and clearly tagged |
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
short-branch summary, tail summary, and spectral-transplant step. A failed or incomplete step blocks
final reporting.

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
then rerun the command above with `--resume`. Any change to either ledger, the plan, the scope, or the
completion step contract prevents reuse of the old audit/evaluation prefix.
An older completion ledger without a per-step input binding is not grandfathered in; rerun the
completion command once with `--resume` to validate the current inputs and rewrite the ledger.

## Final independent audit

Run the canonical resume-safe finalizer only after the compute ledger is complete:

~~~bash
embed-optim-dense-finalize \
  --scope-amendment configs/dense_scope_amendment.json \
  --completion-ledger logs/dense-completion-pipeline/pipeline-ledger.json \
  --workdir "$PWD" \
  --resume
~~~

If finalization is queued behind a still-running completion process, add its exact process ID with
`--wait-pid COMPLETION_PID`. If either process fails, recover completion first and then rerun the
finalizer with `--resume`. The finalizer revalidates current upstream provenance even when its own
ledger says `complete`; a changed completion ledger or finalization step contract causes the final
steps to rerun instead of reusing stale reports.
If an older completion ledger is rejected for missing provenance, upgrade it with completion
`--resume` before retrying finalization.

For independent inspection, its complete ordered finalization sequence is:

~~~bash
python -m embed_optim.aggregate \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json \
  --strict

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
make -C paper clean all
uv build
python -m embed_optim.distribution_audit
~~~

If W&B synchronization is included, authenticate outside the repository and explicitly select
DenseOn. Do not delete historical Late runs.

Finally verify:

- the GitHub repository is still private;
- CI belongs to the final commit;
- the blog, paper, manifests, built archives, and W&B histories all identify the same Dense scope;
- a secret-pattern scan of the tracked tree, reachable Git history, and distributions returns no
  credentials;
- remaining mentions of LateOn are limited to transparent historical-scope disclosure or source
  titles.
