# Project status and handoff

Last updated: 2026-09-03 13:35 UTC

This is the canonical handoff page for humans and coding agents. Read it before launching jobs or
changing result language. The active study is DenseOn-only; LateOn is retained solely as historical
provenance under the user-directed scope amendment.

For a fresh, artifact-only view on the experiment host, run
`python -m embed_optim.corrected_progress --output CURRENT_PROGRESS.json`. The tracked
`CURRENT_PROGRESS.json` is the latest durable machine-readable handoff snapshot; it is updated at
meaningful state transitions rather than every optimizer step. The active implementation was
merged through [PR #42](https://github.com/qcznlp/embedding-optimizer-study/pull/42), and the
execution checklist is [issue #41](https://github.com/qcznlp/embedding-optimizer-study/issues/41).

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
| Candidate addendum release | Complete | 21/21 current-source steps passed |
| Corrected Dense no-packing replication | Formal training active in two four-GPU pools | 0/12 complete; 2 active; 2/60 resumable checkpoints |
| Public checkpoint backup | Complete | 5,546 files, 416,844,858,513 bytes |
| Public result backup | Complete including candidate addendum | 49 addendum files, 24,378,651 bytes |
| GitHub visibility | Public | default branch plus auditable work branches |
| Clean-clone paper audit | Complete locally and in GitHub CI | 2,785 files, 107,442,256 bytes, SHA-256 verified |
| GitHub main CI | Green | merge `59556f2`, workflow run `33761595383` |

The 34 historical Dense training runs are finished. No historical checkpoint should be overwritten.
The corrected no-packing phase now has an explicit implementation. Its worst-case 256-row
engineering preflight passed at micro-batch 8 with 39,977,408,000 allocated bytes and
44,021,317,632 reserved bytes on the slowest/largest rank. The final corrective execution and
analysis protocol is locked at `configs/dense_no_packing_execution_protocol.json`; it uses a new
output namespace. Formal training started at 2026-09-03 11:50 UTC (19:50 in the host's UTC+8 local
time) with `padded-adamw-1e-6` and `padded-adamw-3e-6` in the two disjoint four-GPU pools. Both runs
have written their first resumable checkpoint at step 782, including the model, optimizer,
scheduler, trainer state, and all four RNG-state payloads. They continued training without a CUDA,
distributed, non-finite, or traceback marker; neither run is yet complete.

SentenceTransformers does not serialize the runtime `can_flatten_inputs` value into a saved Dense
checkpoint. Corrected validation and BEIR therefore use new isolated entrypoints that force and
verify padding after every reload. Their implementation was locked before any corrected checkpoint
or evaluation output existed in `configs/dense_no_packing_evaluation_protocol.json`; the historical
source-bound evaluators remain unchanged.

The corrected weight-space definitions and retrieval-bridge scientific plan are locked in
`configs/dense_no_packing_analysis_protocol.json`. That lock was written while the two active runs
were still below checkpoint 782 and no corrected checkpoint weights, validation outputs, BEIR
outputs, or corrected geometry outputs existed. It reports saved-checkpoint segment displacement
and cumulative displacement, stable/effective ranks, and all-rate rank-16 left/right subspace
overlaps. It explicitly does not relabel the displacement between retained checkpoints as a
per-step optimizer update. The retrieval bridge uses four leave-dose-index-out folds and publishes
all predeclared features rather than selecting one after seeing BEIR. Its executable source and
otherwise prediction-invariant implementation choices are bound separately in
`configs/dense_no_packing_bridge_implementation_protocol.json`; that implementation lock was made
after the first two step-782 checkpoint payloads existed but before any corrected validation,
BEIR, geometry, outcome, or bridge output existed.

The corrected outcome implementation is locked separately in
`configs/dense_no_packing_outcome_protocol.json`, also before corrected validation or BEIR output
exists. It fail-closes unless the full 840-unit grid and all 12 validation/system rows pass their
source and checkpoint audits. The primary task effect averages all four rates within optimizer;
the validation-selected recipe result is explicitly secondary. Both report the three common-
resample, 50,000-draw simultaneous max-T intervals defined in that protocol. Observed dynamics AUC
covers 20%–100% only and does not invent an initialization score. The geometry and outcome locks
were reviewed and merged into `main` through
[PR #44](https://github.com/qcznlp/embedding-optimizer-study/pull/44) and
[PR #45](https://github.com/qcznlp/embedding-optimizer-study/pull/45), respectively.

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
contract at `2026-09-03T10:57:12Z`. Its checked-in ledger records return code 0 for every step,
including two strict paper audits, the full test suite, PDF release, and distribution audit. The
step-contract SHA-256 is
`197cc2e24767220113b9d0be4c631c840f09d1e89a893d816069d4d0a3422149`. The compiled paper is 16
pages; the audited main-text endpoint is page 8, and all embedded fonts are non-Type-3.

## Repository and agent handoff health

`README.md` is the public entry point, this file is the canonical live state, and `AGENTS.md`
contains the non-negotiable operating rules. A clean index checkout now passes the strict Dense
paper audit using `configs/portable_paper_evidence.json`: 2,785 required evaluation artifacts,
107,442,256 bytes total, with every path, byte count, and SHA-256 checked. The closure is rebuilt
from the retrieval-dynamics, tail-stability, spectral-transplant, and supplemental five-stage
source manifests, so it cannot silently become stale.

GitHub CI exposed one additional portability defect that the producer host had masked: the deep
outcome-summary reconstruction still opened historical absolute CSV paths directly. Those paths
happened to exist on the producer host but were inaccessible in the GitHub runner. Historical
project paths are now rebased by the shared report reader, unrelated absolute paths retain literal
meaning, and a regression test covers a renamed clean checkout. The paper audit also reports the
specific failed outcome sub-contract instead of returning only an opaque false value.

Full model-state reconstruction remains a separate stronger mode. If the repository contains an
`outputs/` tree, the supplemental five-stage audit requires the original checkpoint-backed path and
will not fall back after a source failure. A clean clone without that 416GB tree validates the
published evaluation closure instead. Restore the Hugging Face checkpoint backup to perform the
full-source reconstruction.

The publication changes were merged to `main` through GitHub PR #38 at
`2026-09-03T11:02:11Z`. Both the final PR head and merge commit passed the clean-clone workflow;
the merge-commit run is <https://github.com/qcznlp/embedding-optimizer-study/actions/runs/33747507631>.

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
- `project/configs`: 40 files, 875,816 bytes.
- `project/logs`: 2,533 files, 82,798,459 bytes.
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
9. For the active corrective phase, read `configs/dense_no_packing_execution_protocol.json` and its
   preflight parent. Micro-batch 8 is selected for all 12 runs; do not tune execution scheduling by
   optimizer or change the locked analysis after formal outputs are visible.
10. Use `python -m embed_optim.corrected_progress` for an artifact-only status report and
    `docs/dense-no-packing-retrain.md` for exact resume/evaluation commands.
11. Materialize geometry only through `python -m embed_optim.corrected_geometry_matrix`; it verifies
    the analysis protocol and source bindings before reading complete corrected checkpoints.
12. Build corrected inference only through `python -m embed_optim.corrected_outcome_summary`; it
    requires the complete validation and 840-unit retrieval grid and verifies the outcome protocol.
13. Build the geometry-to-retrieval bridge only through
    `python -m embed_optim.corrected_retrieval_bridge`; it verifies both upstream summary manifests
    and the separately source-bound implementation protocol before fitting any model.

## Operational constraints

- Do not inspect, edit, signal, stop, replace, or otherwise touch `gpu.py` or its processes. It is an
  external utilization keeper and yields automatically.
- Do not restart LateOn training or evaluation; it is outside the active scope.
- Do not print or commit API keys. Authentication is environment-managed.
- Frozen protocols and failure thresholds are evidence. Never relax a gate to make a result pass.
- Stages 1–4 of supplemental BEIR dynamics are descriptive; only their pre-existing stage-5 roots
  feed formal hybrid/confirmatory inference.
