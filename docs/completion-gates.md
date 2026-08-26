# Study completion gates

This checklist defines what “complete” means for the formal AdamW/Muon/NorMuon study. A running
process, a plausible file count, or a green unit test is not sufficient by itself: each gate names
the authoritative artifact and the command that must validate its full scope.

| Requirement | Authoritative evidence | Passing condition |
|---|---|---|
| Frozen experiment | `configs/experiment.yaml` and the immutable contract in `embed_optim.aggregate` | Exactly two model families, three optimizers, four learning rates each, seed 42, and five checkpoint fractions: 24 runs total. |
| Shared training data | Materialized dataset manifest/rows plus every `completed.json` | Exactly 500,000 canonical rows, seven source quotas, seven distinct seeded negatives per query, no positive overlap, and one shared training-view fingerprint. |
| Training artifacts | Run directories and `logs/checkpoint-audit.json` | 24/24 terminal runs and 120/120 non-empty, step-consistent model/optimizer/scheduler/Trainer/RNG checkpoints pass deep validation. |
| Training dynamics | Canonical Trainer histories and W&B canonical runs | Every run has unique increasing steps through 3,907, finite loss/gradient/LR/epoch values, audited system metrics, and one matching content-addressed W&B history. |
| Retrieval evaluation | Pinned evaluation manifest and MTEB result files | All `24 × 5 × 14 = 1,680` run/checkpoint/task units identify the expected local checkpoint, dataset revision, split/subset, runtime, scorer, and finite nDCG@10. |
| Final statistics | `reports/coverage.json`, long-form tables, summaries, paired effects, and figures | `embed-optim-aggregate --strict` exits zero; no partial or best-effort report is accepted. |
| Weight-space analysis | [`reports/weight-space/summary_manifest.json`](../reports/weight-space/summary_manifest.json) | 24 runs and 120 checkpoint rows pass record-hash, finite-value, partition, and source-input revalidation. |
| Common-state mechanism | `reports/common-state/summary_manifest.json` and `results/common-state-spectra/summary/summary_manifest.json` | All 20 frozen anchors, 1,760 gradient tensors, 5,280 optimizer-update tensors, and 360 prespecified exact spectra pass strict source-hash aggregation. |
| Representation bridge | Both representation-tier `summary_manifest.json` files | Each tier contains exactly two pretrained plus 120 checkpoint reports with the frozen probe identity, sample groups, representation roles, and ranking-reference contract. |
| Blog | [`docs/blog.md`](blog.md) | Strict aggregation replaces both marked sections and the in-progress sentinel; every reported number derives from checked-in aggregate artifacts. |
| Reproducible distribution | Wheel, sdist, repository tests, and CI | Package build succeeds; configs, workers, docs, report tables/figures/manifests, citation, license, and notices are present with working local links. |
| Publication hygiene | Git tracked tree/history and GitHub settings | No credential material is tracked; the repository remains private until the user explicitly requests publication. |

## Final independent audit

Run these after the unattended evaluator reports complete, even though the supervisor already runs
the aggregation and W&B finalizers:

```bash
embed-optim-sync-wandb --matrix configs/experiment.yaml
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
embed-optim-aggregate --matrix configs/experiment.yaml --strict

uv build
uv run pytest
uv run ruff check src tests scripts/eval
uv run ruff format --check src tests scripts/eval
```

Then inspect the built wheel rather than inferring its contents from `pyproject.toml`, verify the
GitHub CI result belongs to the final commit, and rerun a secret-pattern scan over both the tracked
tree and Git history. The final handoff should link the exact blog, coverage manifest, principal
tables/figures, W&B project, commit, pull request, and CI run.
