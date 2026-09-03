# Project status and handoff

Last updated: 2026-09-03 UTC

This is the canonical handoff page for humans and coding agents. Read it before launching jobs or
changing result language. The active study is DenseOn-only; LateOn is retained solely as historical
provenance under the user-directed scope amendment.

## Snapshot

| Workstream | Status | Audited coverage |
| --- | --- | ---: |
| Dense discovery training | Complete | 12/12 runs, 60 checkpoints |
| Routing-matched AdamW controls | Complete | 4/4 runs, 20 checkpoints |
| Validation-frozen confirmation | Complete | 9/9 runs, 45 checkpoints |
| Shared-start short branches | Complete | 9/9 runs, 45 checkpoints |
| Dense BEIR dynamics | Complete | 1,750 task units |
| Supplemental five-stage dynamics | Complete | 65 rows, 910 task units |
| Candidate-breadth evaluations | Complete | 12/12 runs, 224 queries, 6 widths |
| Candidate-breadth frozen decision | Not supported | all three required gates failed |
| Candidate addendum release | Complete | 21/21 source-bound steps passed |
| Public checkpoint backup | Complete | 5,546 files, 416,844,858,513 bytes |
| Public result backup | Complete including candidate addendum | 49 addendum files, 24,378,651 bytes |
| GitHub visibility | Public | default branch plus auditable work branches |
| Clean-clone paper audit | Complete locally | 2,785 files, 107,442,256 bytes, SHA-256 verified |

The 34 Dense training runs are finished. No historical checkpoint should be overwritten. The next
scientific phase, if broader optimizer claims are desired, is a corrected no-packing rerun in a new
output namespace.

## Result that currently governs the paper

The validation-frozen three-seed full-corpus comparison is negative for the selected high-dose
Muon-family recipes:

- Muon minus AdamW mean nDCG@10: -0.030618, familywise 95% CI [-0.046395, -0.013768].
- NorMuon minus AdamW mean nDCG@10: -0.030416, familywise 95% CI [-0.044644, -0.013759].
- NorMuon minus Muon: +0.000202 with an interval crossing zero.

The discovery sweep remains exploratory: its best individual Muon and NorMuon settings slightly
exceed the best AdamW setting, while the four-rate median advantage does not survive the
validation-frozen recipe choice. Muon-family optimizer state, checkpoint size, and peak allocated
memory are smaller in this stack, but throughput is not higher.

## Critical execution finding

The frozen candidate-breadth check exposed a material execution-path defect. Historical Dense
training and training-style validation used SentenceTransformers flattened/packed inputs with
ModernBERT FlashAttention. Independent candidate and corpus scoring used padded inputs. Identical
examples are not batch invariant on the packed path.

Evidence:

- Across the 12 candidate runs, padded width-7 metrics fail to reproduce the legacy packed
  validation artifacts; the largest sample/metric error is 8.286419, versus the frozen 1e-5 limit.
- A pinned two-example implementation audit changes one cosine score by as much as 0.211914 when a
  second example is added in packed mode. With flattening disabled, the corresponding BF16 maximum
  is 0.001953.
- On corrected padded width 7, high-dose minus retrieval-optimal Muon is already worse: loss
  +0.177064 [0.089243, 0.279614] and margin -0.006670 [-0.012219, -0.001019]. NorMuon shows loss
  +0.190773 [0.060921, 0.332021] and margin -0.004567 [-0.010618, 0.001391].
- Increasing candidate width to 2,048 does not produce the prospectively required joint reversal.
  The frozen missing-candidate explanation is therefore `not_supported`.

The important new interpretation is not “Muon needs more negatives.” The narrow validation
advantage that selected 3e-3 disappears when the same eight texts are scored through the padded,
batch-stable path. This is a post-failure implementation diagnosis, not prospective mechanism
evidence. It limits the historical optimizer comparison to the exact pinned training stack and
means a corrected no-packing retrain is required before claiming a general property of clean
eight-way embedding training.

Primary receipts:

- `reports/candidate-breadth/summary.json`
- `reports/candidate-breadth/high_dose_contrasts.csv`
- `reports/candidate-breadth/packing_invariance.json`
- `logs/candidate-breadth-release/pipeline-ledger.json`

The candidate addendum release controller completed all 21 steps against the current source
contract at `2026-09-03T08:14:21Z`. Its checked-in ledger records return code 0 for every step,
including two strict paper audits, the full test suite, PDF release, and distribution audit. The
step-contract SHA-256 is
`63a7d0b932743adcbb172cbe7a03a863dba2e5d4d987792fbc1b0931e5cb63e8`. The compiled paper is 16
pages; the audited main-text endpoint is page 8, and all embedded fonts are non-Type-3.

## Repository and agent handoff health

`README.md` is the public entry point, this file is the canonical live state, and `AGENTS.md`
contains the non-negotiable operating rules. A clean index checkout now passes the strict Dense
paper audit using `configs/portable_paper_evidence.json`: 2,785 required evaluation artifacts,
107,442,256 bytes total, with every path, byte count, and SHA-256 checked. The closure is rebuilt
from the retrieval-dynamics, tail-stability, spectral-transplant, and supplemental five-stage
source manifests, so it cannot silently become stale.

Full model-state reconstruction remains a separate stronger mode. If the repository contains an
`outputs/` tree, the supplemental five-stage audit requires the original checkpoint-backed path and
will not fall back after a source failure. A clean clone without that 416GB tree validates the
published evaluation closure instead. Restore the Hugging Face checkpoint backup to perform the
full-source reconstruction.

The publication changes are maintained in GitHub PR #38. Local full-source release gates and the
clean-index portable audit are green; GitHub clean-clone CI must also be green before the PR is
marked ready or merged.

## Public artifacts

- Source: <https://github.com/qcznlp/embedding-optimizer-study>
- Checkpoints: <https://huggingface.co/qcz/embedding-optimizer-study-checkpoints>
- Analysis artifacts: <https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts>
- Training dashboard: <https://wandb.ai/stevezenguom/embedding-optimizer-study>

The two Hugging Face repositories are public. The checkpoint repository was verified by relative
path and byte size with zero missing, extra, or mismatched local experiment files. The candidate
addendum and refreshed project snapshot are also uploaded. A post-upload Hub API audit reports zero
missing, extra, or size-mismatched files for every refreshed prefix:

- `candidate-breadth`: 49 files, 24,378,651 bytes.
- `project/data/candidate-breadth-224-seed20260901`: 26 files, 302,374,905 bytes.
- `project/configs`: 39 files, 137,560 bytes.
- `project/logs`: 2,533 files, 82,515,170 bytes.
- `project/reports`: 232 files, 16,246,897 bytes.

## Safe continuation order

1. Read this file, `README.md`, and `AGENTS.md`; then run `git status` and preserve all evidence.
2. Audit the clean-clone evidence with `python scripts/portable_evidence.py --audit-only` and the
   strict Dense paper-audit command shown in `README.md`.
3. Read `configs/dense_scope_amendment.json` and `configs/candidate_breadth_probe.json`.
4. Audit candidate results with
   `python -m embed_optim.candidate_breadth_summary --audit-only`.
5. Reproduce the implementation control on a free CUDA device with
   `python -m embed_optim.packing_invariance --device cuda`, then verify its source hashes without
   model inference using `python -m embed_optim.packing_invariance --audit-only`.
6. Treat the completed 21/21 candidate release ledger as the source of truth. If source-bound files
   change, start a new audited release attempt rather than editing the completed receipt.
7. Treat the uploaded candidate addendum and project snapshot as immutable receipts. If any source
   changes, upload into a new namespace or refresh the exact affected prefix and repeat the remote
   relative-path/byte-size audit.
8. For corrected training, disable dense input flattening explicitly and use new run IDs/output
   roots. Do not relabel or overwrite any of the 34 historical runs.

## Operational constraints

- Do not inspect, edit, signal, stop, replace, or otherwise touch `gpu.py` or its processes. It is an
  external utilization keeper and yields automatically.
- Do not restart LateOn training or evaluation; it is outside the active scope.
- Do not print or commit API keys. Authentication is environment-managed.
- Frozen protocols and failure thresholds are evidence. Never relax a gate to make a result pass.
- Stages 1–4 of supplemental BEIR dynamics are descriptive; only their pre-existing stage-5 roots
  feed formal hybrid/confirmatory inference.
