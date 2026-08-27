# Study completion gates

This checklist defines what “complete” means for the formal AdamW/Muon/NorMuon study. A running
process, a plausible file count, or a green unit test is not sufficient by itself: each gate names
the authoritative artifact and the command that must validate its full scope.

| Requirement | Authoritative evidence | Passing condition |
|---|---|---|
| Frozen experiment | `configs/experiment.yaml` and the immutable contract in `embed_optim.aggregate` | Exactly two model families, three optimizers, four learning rates each, seed 42, and five checkpoint fractions: 24 runs total. |
| Shared training data | Materialized dataset manifest/rows plus every `completed.json` | Exactly 500,000 canonical rows, seven source quotas, seven distinct seeded negatives per query, no positive overlap, and one shared training-view fingerprint. |
| Training artifacts | Run directories and `logs/checkpoint-audit.json` | 24/24 terminal runs and 120/120 non-empty, step-consistent model/optimizer/scheduler/Trainer/RNG checkpoints pass deep validation. |
| Training dynamics | Canonical Trainer histories and W&B canonical runs | Every run has unique increasing steps through 3,907, finite loss/gradient/LR/epoch values, audited system metrics, and exactly one matching content-addressed W&B history tagged `canonical-current`; retained historical revisions are permitted but must not carry that tag. |
| Training-only paper tables and figures | `reports/training-dynamics/summary_manifest.json` and `plot_manifest.json` | Exactly 24 run rows, 120 five-stage trailing-window rows, six family/optimizer system summaries, two source-bound SVG figures, and content hashes for every completion/schedule/history source pass audit. |
| Query-disjoint recipe validation | `configs/validation_probe.json`, `data/validation-4096-seed20260826/manifest.json`, and `reports/recipe-validation/manifest.json` | All 4,096 queries are disjoint from the 500K training ledger, all positive/seven-negative groups pass audit, 24 final-checkpoint jobs are complete, and six recipes are selected without BEIR outcomes. |
| Confirmatory data and matrices | `configs/confirmatory_protocol.json`, `reports/confirmatory-data/receipt.json`, and `configs/generated/confirmatory/manifest.json` | Seeds 314159/271828/161803 reuse all 500,000 query/positive identities, independently resample seven distinct negatives from the reconstructed ten-document pools, exceed the frozen pairwise-change threshold, and generate exactly 18 validation-selected runs. |
| Confirmatory training/evaluation | 18 audited completion records and `reports/confirmatory/summary_manifest.json` | Two families × three optimizers × three new seeds complete with the query-disjoint-selected recipes; all `18 × 14 = 252` final-checkpoint retrieval units and seed/task uncertainty pass audit. |
| Retrieval evaluation | Pinned evaluation manifest and MTEB result files | All `24 × 5 × 14 = 1,680` run/checkpoint/task units identify the expected local checkpoint, dataset revision, split/subset, runtime, scorer, and finite nDCG@10. |
| Final statistics | `reports/coverage.json`, long-form tables, summaries, paired effects, and figures | `embed-optim-aggregate --strict` exits zero; no partial or best-effort report is accepted. |
| Retrieval time-to-reference | `configs/retrieval_dynamics_protocol.json` and `reports/retrieval-dynamics/summary_manifest.json` | The protocol hash and its 160/1,680-unit visibility disclosure pass audit; all 1,680 result files rehash into 120 checkpoint rows, and every run reports its first observed crossing of the within-family median AdamW final quality or remains explicitly right-censored, using step-proportional audited useful wall time and no interpolation. |
| Weight-space analysis | [`reports/weight-space/summary_manifest.json`](../reports/weight-space/summary_manifest.json) | 24 runs and 120 checkpoint rows pass record-hash, finite-value, partition, and source-input revalidation. |
| Common-state mechanism | `reports/common-state/summary_manifest.json` and `results/common-state-spectra/summary/summary_manifest.json` | All 20 frozen anchors, 1,760 gradient tensors, 5,280 optimizer-update tensors, and 360 prespecified exact spectra pass strict source-hash aggregation. |
| Scale-matched functional intervention | `configs/functional_intervention.json` and `reports/functional-intervention/manifest.json` | All 20 anchors contain the baseline plus 12 optimizer/direction/scale conditions on the frozen 224-query unseen probe; 58,240 paired sample records and all source hashes pass audit. |
| Final causal/confirmation blog section | `reports/outcome-summary.manifest.json` | Hybrid routing, common-scale local directions, final-stage three-seed shared-start contrasts, and all six confirmatory optimizer contrasts are rendered from strict hashed tables into the dedicated blog marker region; the mechanism marker is revalidated before the final whole-blog hash is signed. |
| Shared-checkpoint short branch | `configs/short_branch_protocol.json`, `reports/short-branch/subset-receipt.json`, and `reports/short-branch/summary_manifest.json` | Both families start from the same fixed AdamW 60% checkpoint; three operators use common-state-derived hidden LRs at the frozen `5e-4` global update/weight target and three order seeds on one exact 50K subset. All five branch checkpoints pass the two frozen functional probes and the 90-row cross-probe bridge. |
| Hybrid AdamW fairness control | `configs/hybrid_adamw_control.json`, eight audited training completions, and `reports/hybrid-adamw/summary_manifest.json` | Both families × four frozen hidden learning rates use the Muon parameter routing and fixed auxiliary recipe; all 112 final-checkpoint BEIR units pass provenance checks. |
| Representation bridge | Both representation-tier `summary_manifest.json` files and the two manifests under `reports/representation-space/` | Each tier contains exactly two pretrained plus 120 checkpoint reports with the frozen probe identity, sample groups, representation roles, and ranking-reference contract; the shared and LateOn-specific plotted dynamics rehash both strict summaries. |
| Cross-space join | `reports/mechanism-bridge/summary_manifest.json` | Exactly 120 checkpoint rows and 96 within-run transitions join the strict weight, representation, and 1,680-unit BEIR sources; correlations remain labeled descriptive. |
| Mechanism report | `reports/mechanism-summary.manifest.json` | The common-state, 360-spectrum, two-tier representation, figures, and 120-checkpoint bridge sources are rehashed before the fixed mechanism tables are rendered. |
| Blog | [`docs/blog.md`](blog.md) | Strict aggregation replaces the retrieval/system sections and in-progress sentinel, then strict mechanism rendering replaces its marked section; every reported number derives from audited aggregate artifacts. |
| NAACL manuscript | `paper/main.tex`, `paper/results.tex`, six files under `paper/generated/`, `paper/README.md`, `configs/paper_claim_protocol.json`, and `reports/paper-results.manifest.json` | `embed-optim-render-paper-results` generates the five headline macros and six evidence-table files (including the full per-task appendix) from every frozen tier; `make -C paper` builds with the pinned official ACL style; and `embed-optim-audit-paper --strict` proves that the claim protocol, source tables, generated TeX hashes, headlines, and result tables still match with no pending marker. |
| Unattended handoff | `logs/post-eval-pipeline/pipeline-ledger.json` | The strict evaluation, W&B, mechanism, representation, bridge, hybrid, confirmatory, short-branch, paper-build, test, formatting, distribution, and terminal `embed-optim-audit-paper --strict` steps all record successful terminal attempts. |
| Reproducible distribution | Wheel, sdist, repository tests, and CI | Package build succeeds; configs, workers, docs, report tables/figures/manifests, citation, license, and notices are present with working local links. |
| Publication hygiene | Git tracked tree/history and GitHub settings | No credential material is tracked; the repository remains private until the user explicitly requests publication. |

## Final independent audit

Run these after both the unattended evaluator and the post-evaluation pipeline ledger report
complete, even though the pipeline already runs these gates:

```bash
embed-optim-sync-wandb --matrix configs/experiment.yaml
embed-optim-prepare-validation --audit-only
embed-optim-validation-matrix --audit-only --verify-hashes
embed-optim-summarize-validation
embed-optim-prepare-confirmatory-data --audit-only --verify-source
embed-optim-generate-confirmatory-matrices --audit-only
embed-optim-evaluate-confirmatory --audit-only
embed-optim-summarize-confirmatory
embed-optim-short-branch --subset-only --audit-only
embed-optim-short-branch --audit-only
embed-optim-short-branch-evaluate --audit-only
embed-optim-summarize-short-branch
embed-optim-summarize-geometry \
  --geometry-root results/weight-space \
  --output-dir reports/weight-space \
  --verify-inputs
embed-optim-common-state-matrix --audit-only --verify-hashes
embed-optim-summarize-common-state \
  --matrix configs/experiment.yaml \
  --result-root results/common-state \
  --output-dir reports/common-state
embed-optim-common-state-spectra --audit-only --verify-hashes
embed-optim-common-state-spectra --summarize-only
embed-optim-plot-common-state-spectra
embed-optim-functional-intervention-matrix --audit-only --verify-hashes
embed-optim-summarize-functional-interventions
embed-optim-plot-representation-dynamics
embed-optim-plot-late-token-dynamics
embed-optim-aggregate --matrix configs/experiment.yaml --strict
embed-optim-summarize-hybrid-control
embed-optim-build-mechanism-bridge
embed-optim-render-mechanism-report
embed-optim-render-outcome-report
embed-optim-render-paper-results
make -C paper
embed-optim-audit-paper --strict

uv build
uv run pytest
uv run ruff check src tests scripts/eval
uv run ruff format --check src tests scripts/eval
```

Then inspect the built wheel rather than inferring its contents from `pyproject.toml`, verify the
GitHub CI result belongs to the final commit, and rerun a secret-pattern scan over both the tracked
tree and Git history. The final handoff should link the exact blog, coverage manifest, principal
tables/figures, W&B project, commit, pull request, and CI run.
