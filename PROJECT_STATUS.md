# Project status and handoff

Last updated: 2026-09-04 01:59 UTC

This is the canonical handoff page for humans and coding agents. Read it before launching jobs or
changing result language. The active study is DenseOn-only; LateOn is retained solely as historical
provenance under the user-directed scope amendment.

The owner retired the separate Markdown article on 2026-09-04. The manuscript under `paper/` is
now the sole publication deliverable; Markdown reports remain source-addressed evidence only.

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
| Corrected Dense no-packing replication | Formal training active in two four-GPU pools | 2/12 complete; second AdamW pair at steps 3214/3182; 18/60 resumable checkpoints |
| Corrected completion handoff | Detached controller restored after two audited migrations | Waiting for training under contract `25eefbe5...`; 2/2 full-run backups re-audited with upload commit IDs intact; no failed step |
| Corrected next-pair handoff | Artifact-only guard active with lease held | Waiting for both current runs; then yield on any Muon successor artifact or take over only after a five-minute absence grace |
| Sealed-checkpoint durability | Detached CPU/network supervisor active | 18/60 checkpoints covered; 42 not yet generated; 0 cycle failures |
| Corrected weight-space | Incremental frozen analysis active | 2/12 runs, 10/60 stages; 12 source-bound files remotely verified |
| Corrected W&B provenance | Read-only partial audit complete | 4/12 visible: 2 finished, 2 running, 0 identity/config/state problems |
| Public checkpoint backup | Historical archive complete; corrected archive incremental | Historical: 5,546 files, 416,844,858,513 bytes; corrected snapshot: 338 files, 33,457,786,318 bytes; stage receipts cover 18/60 checkpoints |
| Public result backup | Complete including candidate addendum | 49 addendum files, 24,378,651 bytes |
| GitHub visibility | Public | default branch plus auditable work branches |
| Clean-clone paper audit | Complete locally and in GitHub CI | 2,785 files, 107,442,256 bytes, SHA-256 verified |
| Corrected paper renderer | Implementation locked before results | four upstream manifests, all 9 bridge features, paper + standalone evidence report |
| Corrected state-by-operator factorial | Scientific and source-bound implementation locks complete; waiting for source checkpoints | 2 source states x 2 reset continuation operators x 3 order seeds; 0/12 runs started |
| Manuscript narrative | Rewritten around the positive Muon trajectory result | 8-page main text; clean outcome block remains source-gated |
| GitHub main CI | Green | see the repository Actions history for the current merge |

The 34 historical Dense training runs are finished. No historical checkpoint should be overwritten.
The corrected no-packing phase now has an explicit implementation. Its worst-case 256-row
engineering preflight passed at micro-batch 8 with 39,977,408,000 allocated bytes and
44,021,317,632 reserved bytes on the slowest/largest rank. The final corrective execution and
analysis protocol is locked at `configs/dense_no_packing_execution_protocol.json`; it uses a new
output namespace. Formal training started at 2026-09-03 11:50 UTC (19:50 in the host's UTC+8 local
time) with `padded-adamw-1e-6` and `padded-adamw-3e-6` in the two disjoint four-GPU pools. Both runs
deeply completed at step 3907 with all five scheduled checkpoints, including the model, optimizer,
scheduler, trainer state, and all four RNG-state payloads. The recovery supervisor then launched
`padded-adamw-1e-5` and `padded-adamw-3e-5`; at the 01:59 UTC artifact snapshot they were at steps
3214 and 3182, respectively, and both had deeply valid checkpoints through step 3126. Across
the corrected phase there is still no CUDA OOM, NCCL data-plane, non-finite, or traceback marker.

At 2026-09-03 15:28:38 UTC the interactive matrix controller and its torchrun TCPStore disappeared,
leaving the eight established training ranks adopted by PID 1. The NCCL data plane continued to
advance both runs and write checkpoints, but the heartbeat monitor repeatedly logged TCPStore
Broken-pipe warnings and no controller remained to launch later runs. This is one control-plane
incident affecting two runs, not one failure per repeated warning line. A source-bound detached
supervisor now waits for all eight adopted ranks to exit before invoking the unchanged matrix; it
will skip a deeply complete run or resume an incomplete one from its latest deeply valid
checkpoint. The recovery contract is
`configs/dense_no_packing_control_plane_recovery.json`, and its atomic live state is
`logs/dense-no-packing-v1/recovery-supervisor-state.json`. Fatal error markers remain separate from
these control-plane warnings.

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
`configs/dense_no_packing_bridge_implementation_protocol_v2.json`; that implementation lock was
made after the first two step-782 checkpoint payloads existed but before any corrected validation,
BEIR, geometry, outcome, or bridge output existed. Its v1 predecessor was never executed on
corrected outputs: a synthetic end-to-end test caught an outcome-manifest logical-key mismatch,
which v2 repairs without changing a feature, model, fold, threshold, or claim boundary.

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

The final corrected publication path is also fixed before results. The source-bound renderer in
`src/embed_optim/corrected_publication.py` consumes only complete, hashed outcome, geometry,
bridge, and execution-sensitivity manifests. It publishes all primary and secondary contrasts,
all 15 dynamics rows, all nine frozen geometry features, and the full sensitivity comparison into
one standalone evidence report and the manuscript include. Its timing,
requirements, source hashes, and claim boundary are frozen in
`configs/dense_no_packing_publication_protocol.json`; no corrected evaluation or analysis output
existed when that lock was written.

An operational completion controller merged through
[PR #52](https://github.com/qcznlp/embedding-optimizer-study/pull/52) and is now active. It closes
the handoff gap after formal training and watches only deep artifact completeness, uploads and
remotely size-audits each corrected run as soon as it
finishes, and after 12/12 runs invokes the already locked padded validation, 840-unit BEIR grid,
weight-space analysis, outcome inference, retrieval bridge, execution sensitivity, publication
renderer, and release audits in dependency order. Its own source, parent protocols, commands, and
arguments are content-bound in an atomic runtime ledger; it makes no new scientific selection and
does not alter the training supervisor.
Its pre-paper-only runtime contract SHA-256 was
`55fdd9d6eb159d7784bbb4f9c53e1f0baba99440ab44203e2bcdb3888d3a1e05`; the atomic ledger reported
2/12 runs complete and began uploading `padded-adamw-1e-6` at 19:39 UTC. Both completed runs were
uploaded and remotely size-audited by 19:43 UTC: 202 files and 19,120,907,393 bytes in total, with
zero missing, extra, or size-mismatched paths. The immutable per-run receipts are under
`reports/dense-no-packing/checkpoint-backup/`.

At 00:07 UTC on 2026-09-04 that controller detected the owner-directed paper-only source change and
failed closed before any new training completion or finalization step. Its lease is free; the
training matrix continued to advance and the independent sealed-checkpoint backup lease remained
held. A one-time transition from the exact old contract hash above to
`8687929d22adf4a6a07593b5aa67669f816aef0b902f2d035522802fd94b4c96` is frozen in
`configs/dense_no_packing_completion_contract_migration.json`. The migration accepts only the six
declared publication-bound protocol changes, requires the controller, matrix, execution protocol,
arguments, and command order to remain identical, and archives the original ledger byte-for-byte.
The migration executed at 01:16 UTC, archived the original 10,069-byte ledger under its
`7d4da69912e39199bc14a176d95d102a593e7f12b267b35dc13a20b3bd040c93` SHA-256, and resumed the
controller under the new contract. By 01:17 UTC both existing full-run backups had passed fresh
audit-only checks, the controller lease was held, the ledger was back in `waiting_for_training`,
and no failed step was present. The distributable operational receipt is
`reports/dense-no-packing/completion-contract-migration.json`; the ordinary `--resume` path is now
authoritative again.

That first resumed audit exposed an operational receipt bug in
`corrected_checkpoint_backup.py`: audit-only verification had no new upload response and therefore
overwrote the two stored Hugging Face upload commit IDs with null. The remote inventories remained
complete and unchanged. The original commit IDs (`71c58f98367eea5a15464163600a22c2005f7c76` and
`fd47604c588deae679e17411a6acfc0d455613f9`) were restored from the tracked pre-audit receipts. The
fix now preserves those fields only when the run, repository, prefix, local root, and inventory
digest all match, and fails closed otherwise. It also adds the invoked backup implementation to
the completion controller source contract. The exact second transition from `8687929d...` to
`25eefbe52b4cc275600dae631860d2aa69a0734c1c143813b4e7e4a9190f3c13` is frozen in
`configs/dense_no_packing_backup_provenance_migration.json`. It passed PR
[#66](https://github.com/qcznlp/embedding-optimizer-study/pull/66), executed at 01:30 UTC, and
archived the prior 12,031-byte ledger under SHA-256
`e75215aa6d8600769e35fcc044871eaf902cee6b4f105d5b9ab0ad377f7942e6`. The resumed controller
re-audited both complete runs with the original commit IDs intact, then returned to
`waiting_for_training` at 01:32 UTC with its lease held and no failed step. The distributable
receipt is `reports/dense-no-packing/backup-provenance-migration.json`. Training and the independent
sealed backup supervisor were unaffected throughout.

Before any corrected validation, BEIR, geometry, bridge, sensitivity, or publication output
existed, a renderer-shaped paper fixture showed that placing all corrected tables in the main text
would move the NAACL endpoint from page 8 to page 9 and make the primary table over-wide. PR
[#70](https://github.com/qcznlp/embedding-optimizer-study/pull/70) fixed the result-blind topology:
the three-contrast all-rate retrieval answer remains in the main narrative, while the complete
nine-feature geometry bridge and execution-path sensitivity tables are invoked in the appendix.
The final fixture places the primary table on page 6, ends the main paper on page 8, places both
detail tables on page 11, and has no overfull boxes. No scientific input, row, estimand, interval,
support rule, threshold, conclusion branch, or standalone evidence output changed.

That publication-only source transition was frozen in
`configs/dense_no_packing_publication_layout_migration.json` and executed at 02:36 UTC, moving the
completion contract from `25eefbe52b4cc275600dae631860d2aa69a0734c1c143813b4e7e4a9190f3c13` to
`abb5973ab7247ec427195ab9afa6f60add579e9e33829e3cc08a61f4947d5a67`. Both identities are recorded
in the migration protocol and receipt. The controller resumed under
`/usr/bin/python3`, re-audited both
complete remote runs with their original commit identities intact, and returned to
`waiting_for_training` at 02:38 UTC with its lease held and no failed step. The distributable
receipt is `reports/dense-no-packing/publication-layout-migration.json`; scientific completion
remains false at 2/12 runs.

A subsequent result-blind narrative audit found one remaining final-paper hazard: the source-bound
corrected renderer would replace the result section, but its all-rate verdict would not reach the
abstract or main Conclusion, where the historical packed-selector claim could remain visually
dominant. The pending amendment defines one identical corrected all-rate finding for the abstract
and Conclusion, retains the full corrected analysis in the main results section, and moves the
complete historical selector claim string to the appendix. No corrected validation, BEIR,
geometry, outcome, bridge, sensitivity, or publication output was visible; only 2/12 training runs
were complete. PR [#72](https://github.com/qcznlp/embedding-optimizer-study/pull/72) passed CI and
merged. The exact non-scientific transition in
`configs/dense_no_packing_publication_narrative_migration.json` was executed at 03:12 UTC,
archiving the prior 16,250-byte ledger under SHA-256
`7c55a3bf6dc0d0f47920f4f81d6d881859426f680ccbf6dfe90612ad2ebeb0c4` and moving the completion
contract from `abb5973ab7247ec427195ab9afa6f60add579e9e33829e3cc08a61f4947d5a67` to
`4152531e354bdf325411c7f4d6de044ea68b6d5f26a59fbf5914f8b55f3c9789`. The controller resumed
under `/usr/bin/python3`, re-audited the two completed checkpoint uploads without changing their
commit identities, and returned to `waiting_for_training` with its lease held and no failed step.
The distributable receipt is `reports/dense-no-packing/publication-narrative-migration.json`.

The legacy training recovery state remains `matrix_running` while the current pair advances, but
the supervisor implementation predates a code-level lease or heartbeat during its blocking matrix
child. That state alone therefore cannot prove that a parent will remain available to launch the
next pair. Before either current run reached checkpoint 3126, an artifact-only handoff guard was
frozen in `configs/dense_no_packing_matrix_handoff_guard.json`. It never inspects or signals a
process: after both current runs pass deep completion, it waits five minutes for the declared first
Muon pair's log/output artifacts. It yields immediately if the existing matrix advances, and only
invokes the unchanged recovery supervisor after the full absence grace and a final race check. Its
source, matrix, supervisor, GPU pools, ports, and timing are content-bound; live state is under
`logs/dense-no-packing-handoff-guard/`.

The guard passed PR [#68](https://github.com/qcznlp/embedding-optimizer-study/pull/68), started at
01:52 UTC, acquired its dedicated lease, and published `waiting_for_current_runs` with neither
current run yet marked complete and `takeover_launched=false`. Its distributable activation receipt
is `reports/dense-no-packing/matrix-handoff-guard.json`; later yield or takeover replaces this
operational status at the corresponding meaningful transition.

Because the experiment host may be shut down before an active run reaches all five stages, the two
sealed step-782 checkpoints from the second AdamW pair were independently uploaded at 21:17--21:18
UTC. A separate read-only pass verified 34 files and 3,584,170,699 bytes by exact path and byte size,
then matched every Hugging Face LFS object by SHA-256 and every ordinary Git object by blob SHA-1;
all missing, extra, size-mismatch, and digest-mismatch sets are empty. The receipts under
`reports/dense-no-packing/incremental-checkpoint-backup/` bind commits
`c3953328ee6e57ec20ce78f11b1a490b70fec385` and
`1aed10f29770d31b2eff1c158a4216e3223af2d8` and explicitly set
`scientific_completion=false`. The full-run completion controller remains authoritative and will
re-audit these paths together with all later checkpoints after each run finishes.

Both active runs subsequently produced deeply valid checkpoints through step 3126. The
sealed-checkpoint supervisor uploaded and remotely digest-audited every stage. The latest pair was
sealed at 01:49--01:52 UTC: 17 files and 1,792,134,291 bytes for `padded-adamw-1e-5`, and 17 files
and 1,792,134,449 bytes for `padded-adamw-3e-5`. Their Hugging Face commits are
`f378ead00b07516377ca185af9d0961f23043aa3` and
`620594542924e7e82d4f4253b5a03177627c7035`. Both receipts report empty missing, extra,
size-mismatch, and digest-mismatch sets and retain `scientific_completion=false`. Corrected remote
coverage is now 338 files and 33,457,786,318 bytes across the two complete runs and eight sealed
active checkpoints.

PR [#59](https://github.com/qcznlp/embedding-optimizer-study/pull/59) added an independent
lease-protected supervisor so every later sealed checkpoint receives the same digest-verified
backup without an interactive handoff. Its artifact lease remains held; it reads only training
artifacts, uses CPU/network resources, and neither imports CUDA nor controls training. Its runtime contract
SHA-256 is `a65074bc3e3898dedb0849cbc6f6a7e428105ce74280213c5f5b30554a7d89b7`;
the current atomic state records 18/60 checkpoints covered, 42 not yet generated, and zero cycle
failures. Whole-run receipts cover completed runs, intermediate receipts cover active stages, and
the final stage yields to the existing whole-run controller before the checkpoint-level fallback.
The live state and exclusive lease are under `logs/dense-no-packing-sealed-backup/`; none of these
durability states constitute scientific completion.

The frozen corrected geometry implementation has also completed all five stages for both finished
AdamW runs. Each stage contains exactly 88 hidden-matrix records; every recorded scalar is finite,
and all ten record files match their manifest byte counts and SHA-256 digests. The 12-file partial
tree (2,031,870 bytes) is remotely verified under
`qcz/embedding-optimizer-study-analysis-artifacts/corrected-dense-no-packing-v1/weight-space`.
This is an incremental computation and durability receipt only: no cross-optimizer geometry
summary or mechanism interpretation is permitted before the complete 12-run matrix.

The corrected W&B source-run path now has a dedicated read-only audit. Its first receipt at 20:18
UTC found the two locally complete runs remotely `finished` at step 3907, the two active runs
remotely `running`, and all eight future runs absent as expected. All four visible runs match the
deterministic ID, run name, group, tags, and resolved padded matrix configuration, with zero
problems. This post-output operational receipt does not modify W&B or add a scientific endpoint.

A separate corrected state-by-operator factorial is now scientifically frozen in
`configs/dense_no_packing_state_operator_factorial_protocol.json`. It was specified before any
corrected Muon checkpoint or corrected retrieval outcome existed. At the 60% checkpoints of the
historically retrieval-optimal AdamW and Muon rates, it will cross the two weight states with reset
AdamW and Muon continuation operators on the same fixed 50K branch view and three order seeds. Its
predeclared final-BEIR estimands separately measure the carried weight-state effect, the averaged
continuation-operator effect, and their interaction. This is the direct test of the paper's
state-feedback account; the motivating historical crossover remains explicitly post hoc. The
scientific lock is paired with the pre-output, source-bound implementation at
`configs/dense_no_packing_state_operator_factorial_implementation_protocol.json`. That executable
contract covers padded gradient calibration, a storage-efficient direction-norm pass, six
two-run reset-continuation matrices, all-five-checkpoint unseen-probe evaluation, 168 final-BEIR
task units, and the 100,000-replicate two-way seed/task bootstrap. Every execution entry point
fails closed if a bound source changes. No factorial output exists yet; each source checkpoint must
first pass both deep local validation and sealed remote-backup verification. Exact commands and
interpretation rules are in `docs/state-operator-factorial.md`.

## Historical findings that motivate clean adjudication

The paper's main observation is a coherent positive Muon region in the historical grid, not the
failure of one selected recipe. Best final nDCG@10 is 0.589867 for AdamW, 0.592309 for Muon, and
0.593429 for NorMuon; the Muon/NorMuon best runs beat AdamW on 10/11 of 14 tasks. Four-rate final
medians are 0.585762/0.590109/0.590970, and 2/3/3 rates reach the frozen AdamW reference. The
fastest observed first passage to that reference is 1.407 hours for AdamW versus 0.749/0.756 hours
for Muon/NorMuon, although Muon-family throughput per step is lower on this stack.

The historical packed validator separately selects a damaging high-dose Muon-family recipe. Its
three-seed full-corpus contrasts are:

- Muon minus AdamW mean nDCG@10: -0.030618, familywise 95% CI [-0.046395, -0.013768].
- NorMuon minus AdamW mean nDCG@10: -0.030416, familywise 95% CI [-0.044644, -0.013759].
- NorMuon minus Muon: +0.000202 with an interval crossing zero.

These negative contrasts describe recipes chosen by the batch-dependent packed validator; they are
not evidence that Muon lacks a useful retrieval regime. The manuscript's scientific spine is the
locally-weaker/globally-better puzzle: same-state norm-matched AdamW gives the larger immediate mean
margin gain, while repeated Muon updates reach better historical retrieval and a positive final
shared-start unseen-margin contrast in all three seeds. Frozen temporal, spectral-transplant, and
held-run tests reject spectral flattening as the tested explanation. The corrected padded matrix
decides whether the positive retrieval result survives clean execution and whether any of the nine
frozen geometry features predicts it out of dose.

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
14. Compare historical and corrected executions only through
    `python -m embed_optim.corrected_execution_sensitivity`; it matches all fixed rates and stages
    but never pools the executions or labels their difference a causal packing estimate.
15. Render final corrected prose and tables only through
    `python -m embed_optim.corrected_publication`; do not hand edit its generated paper include.
16. Run the source-bound state-by-operator factorial only after each checkpoint-2345 source passes
    its deep local and sealed remote-backup gates. Follow
    `configs/dense_no_packing_state_operator_factorial_implementation_protocol.json` and
    `docs/state-operator-factorial.md`; do not reinterpret the historical crossover as
    confirmatory evidence.
17. Keep `python -m embed_optim.corrected_completion_pipeline` running as the operational handoff.
    It backs up deeply complete runs during the matrix and starts final evaluation only after all
    12 runs pass the deep completion gate. Read its atomic ledger under
    `logs/dense-no-packing-finalization/` before starting any competing finalizer.

## Operational constraints

- Do not inspect, edit, signal, stop, replace, or otherwise touch `gpu.py` or its processes. It is an
  external utilization keeper and yields automatically.
- Do not restart LateOn training or evaluation; it is outside the active scope.
- Do not print or commit API keys. Authentication is environment-managed.
- Frozen protocols and failure thresholds are evidence. Never relax a gate to make a result pass.
- Stages 1–4 of supplemental BEIR dynamics are descriptive; only their pre-existing stage-5 roots
  feed formal hybrid/confirmatory inference.
