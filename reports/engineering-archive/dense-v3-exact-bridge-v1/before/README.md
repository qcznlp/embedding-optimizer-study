<!--
This README substantially modifies the lightonai/mdenseon-mlateon README at commit
b0db47a48f969d825446668b5b17bfc27a359fc1. See THIRD_PARTY_NOTICES.md.
-->

# Muon for dense-retriever adaptation

A reproducible study of AdamW, Muon, and NorMuon for supervised adaptation of
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised).

**Scientific objective.** Starting from the same DenseOn checkpoint, determine whether AdamW,
Muon, and NorMuon reach different retrieval-quality regions, how their trajectories reshape the
weights and 768-dimensional embedding space, and whether retrieval value is carried by the reached
state, the next optimizer, or their interaction. The paper's optimizer verdict is reserved for the
complete 12-configuration primary matrix; earlier exploratory outputs are retained for audit but do
not supply manuscript claims.

The project compares complete training dynamics, decontaminated BEIR retrieval quality, systems cost, and
weight/update geometry. Its central question is not whether Muon produces flatter update spectra—
that is largely built into the operator—but whether Muon and AdamW reach functionally different
weight states, and whether retrieval value is carried by the reached state, the continuation
operator, or their interaction. The manuscript and its
source-bound generated tables are the sole publication deliverable; standalone reports remain
auditable evidence rather than a parallel article.

The repository is public under Apache-2.0. For the current verified state, known limitations, and
exact continuation order, start with [PROJECT_STATUS.md](PROJECT_STATUS.md). Coding agents should
also read [AGENTS.md](AGENTS.md) before taking action.

## 60-second handoff

**Correctness-first hold (September 6):** all **12/12 primary runs finished** and **60/60
checkpoints are preserved**, but the matrix is not cleared for scientific inference. Actual
four-GPU testing established a duplicated loss-normalization defect in unchanged production.

The [real-data preflight](reports/engineering-archive/dense-natural-readiness-v1/README.md)
stopped **before GPU execution** on an empty positive in its fixed prefix. Exhaustive checks find
five affected training groups out of 500,000 and one out of 4,096 validation groups. Actual IDs
all match their original ledgers and source-qualified query IDs remain disjoint. The empty texts
trace to three authenticated upstream documents. A [subsequent full query-text audit](reports/engineering-archive/dense-data-partition-v1/README.md)
finds **84 training records matching 51 validation queries**, plus **one training–BEIR query match**.
Independent full-string replay confirms all pairs are raw-text equal. The expanded fixed-seed
proposal replaces **6 training / 52 validation groups**, retaining all other rows and source quotas.
The [new separate data candidate](reports/engineering-archive/dense-data-candidate-v1/README.md)
now implements this proposal and passes full content/ID/query-isolation checks. An independent
original-scanner and different-writer reconstruction matches all decisions, replacement groups,
complete dataset values and canonical row ledgers. Original data and earlier evidence remain
unchanged. This is exact-query hygiene, not semantic decontamination or an optimizer finding.

The [new revised-input/natural-data milestone](reports/engineering-archive/dense-revised-natural-v1/README.md)
observes both complete revised datasets and derives twelve new full-horizon identities. Its explicit
data amendment remains preparation-only; the actual loader accepts it and rejects 21 changed files.
Three actual four-GPU training calls now start from the immutable untrained base and complete three
steps per optimizer. A shared 288-group fixture covers all seven sources and all six training
replacements. All nine complete diagnostic checkpoints and final exports pass, followed by fresh
CPU readback. The maximum observed length is 7,679 under the unchanged 8,192 limit. This is not a
full epoch/all-rate campaign, natural-gradient oracle, natural-data resume or retrieval result.

The [prepared integrated correction](reports/engineering-archive/dense-correction-candidate-v1/README.md)
passes **96/96 exact NorMuon reference updates on CPU and separately on GPU**, leaves Muon
unchanged, and passes **9/9 short-input + 9/9 maximum-length gradient checks**, all 134 parameter
tensors under the original tolerance. Its actual checkpoint implementation passes **24/24
controlled same-group replay comparisons**; the [new independent-process check](reports/engineering-archive/dense-predeployment-checks-v1/README.md)
passes **24/24 complete comparisons** on the same candidate source, without another baseline.
Its unpatched `run_training` entrypoint completes **three baseline and three continuation calls**.
This is a new detached source checkout, not deployment or a retrieval finding. The communication
and backward-determinism controls do not establish default-DDP bitwise or cross-host guarantees.
All earlier failures remain intact.

The subsequent [full-identity integration](reports/engineering-archive/dense-full-identity-v1/README.md)
closes the missing complete-run admission gate in a distinct prepared source tree. It passes **six
actual four-GPU baseline/continuation calls**, relocating byte-identical data and initial model,
and seals **12 complete diagnostic checkpoints**. Its actual pre-setup admission accepts six
original/relocation controls and rejects **36 changed identities plus three corrupted real optimizer
files** before deserialization. No changed-recipe run was trained and no old checkpoint was
retrofitted. This new save/load/admission evidence is not another cross-process bitwise claim.

The next [primary contract and read-only rehearsal](reports/engineering-archive/dense-primary-contract-v1/README.md)
authenticates the original **500k data rows** and **11 untrained-base files** against immutable HF
digests, and prepares one shared checkpoint gate for training, backup and evaluation. Twelve real
diagnostic checkpoint payloads pass only under their diagnostic identities; all **60 old primary
checkpoints** and **three corrupted copies** are rejected by the new admission. All twelve real
CLI inspection plans and **51 focused checks** pass. The proposal is **not execution-authorized**;
backup networking is tested with a mock, and no new primary run/upload/evaluation has occurred.

The [new whole-run/task/grid reader](reports/engineering-archive/dense-primary-completion-v1/README.md)
now accepts three complete diagnostic runs, deeply checks nine full-model checkpoints, and rejects
three continuation-only directories as whole runs. One real SciFact diagnostic worker exited zero;
its unchanged output passes the CPU reader without imputing undefined auxiliary nAUC values or
changing nDCG. The new adapter requires all twelve complete runs before all **840 distinct task
cells**. Its 63 focused checks pass. No accepted primary result was created.

Keep the test results separate: the current identity candidate's full suite has **963 cases /
11 unchanged, unwaived old source-contract failures**; its 79 focused checks pass. The isolated
paper/audit checkout passes **all 2,105 cases**, including 54 bridge numerical/integration checks.
Tests belonging to one source tree do not certify a different tree or a full scientific experiment.
The candidate is not whole-repository green. Its six actual metadata regressions were fixed;
no old protocol hash, tolerance or acceptance rule was loosened.

The [v3 core-chain implementation](reports/engineering-archive/dense-primary-v3-chain-v1/README.md)
now admits both complete revised data directories and passes twelve actual CLI inspection plans.
It rejects 68 actual old-input/checkpoint, diagnostic-identity and draft-execution cases. Its
whole-grid CLI refuses absent v3 primary runs. This is preparation-only train/backup/BEIR/whole-grid
integration, not a complete validation/outcome/mechanism/publication pipeline or a formal run.

The [v3 validation implementation](reports/engineering-archive/dense-v3-validation-v1/README.md)
now admits all 4,096 revised rows and matches unchanged GPU scoring on three diagnostic models
and 59 declared rows covering all 52 replacements and seven sources. All parameter values are
unchanged. Independent scalar replay and fresh CPU readback pass; 18 changed cases reject even
when corrupted bundle hashes are refreshed. Actual primary selection refuses missing full runs.
This is bounded correctness evidence, not full validation or an optimizer finding; the maximum
actual scoring input is 1,379 tokens. No formal recipe has been selected.

The [v3 outcome adapter](reports/engineering-archive/dense-v3-outcomes-v1/README.md) now joins
complete validation and the 840-cell primary grid to unchanged max-T inference, stage summaries
and admitted system metadata. Its first independent-process CSV replay failed on field order;
that failure is preserved. An explicit serializer-only revision passes independent scalar checks
and fresh CPU readback, with all values/criteria unchanged. Eighteen changed cases reject, and
the actual primary CLI refuses absent v3 runs. All positive outcome tables remain synthetic.

The [new changed-entry component](reports/engineering-archive/dense-v3-geometry-v1/README.md)
corrects a measurement mismatch without rewriting old geometry: parameter mass in changed matrices
is not the fraction of individual entries changed. Exact counts on three real diagnostic trajectories
and nine checkpoints agree with 1,584 independent NumPy comparisons, then fresh CPU readback.
Six altered census files and six altered amendments reject. The actual primary CLI refuses missing
complete v3 runs. This verifies only the entry census, not complete geometry or useful embedding
dimensions. The original failed reference preflight and test invocation are preserved; final tests
pass with explicit absolute source paths. No optimizer finding or formal primary result is produced.

The [complete v3 geometry consumer](reports/engineering-archive/dense-v3-geometry-chain-v1/README.md)
now passes all-rate/stage synthetic orchestration and real diagnostic numeric/basis readback.
Its first independent global-norm check failed despite old-kernel agreement; the preserved failure
led to an explicit FP64 global-reduction revision, with original ranks/seeds/spectra/tolerances
unchanged. The actual retry verifies 792 diagnostic matrix records, 3,168 numeric scalar checks,
264 undefined cases and twelve full-SVD controls, then fresh CPU replay. Ten changed cases reject.
No full primary geometry or optimizer conclusion exists. Exact controls expose approximately 9–12%
sketch stable-rank overestimation for selected Muon/NorMuon matrices, so exact-spectrum/projector
robustness is required before interpreting absolute ranks or mechanisms.

That [bounded robustness audit](reports/engineering-archive/dense-geometry-robustness-v1/README.md)
is now complete: twelve inherited diagnostic matrices have independent full spectra/projectors,
156 sketch-accuracy cases and fresh numerical replay. Original rank-16 sketch/exact overlaps are
about 0.13–0.27 for Muon/NorMuon versus 0.90–0.94 for AdamW. A separate 108-pair crossed-seed
diagnostic and fresh replay also expose substantial shared-probe coupling; changing seeds alone
does not recover exact subspaces. No primary feature is silently rewritten and no optimizer-quality
claim follows. The original approximate features and all failed-attempt provenance remain intact.

The [separate complete exact-geometry consumer](reports/engineering-archive/dense-v3-exact-geometry-v1/README.md)
is now prepared and verified across **all 88 hidden matrices / nine real diagnostic checkpoints**:
1,584 matrix/anchor records, 1,056 nonzero full spectra/projectors, 528 exact zeros, independent
NumPy/PyTorch checks and complete fresh CPU replay. The full primary positive fixture is synthetic;
no real primary output or optimizer conclusion is produced. Zero/unresolved spaces and aggregation
denominators are explicit, and the old approximate feature family is not silently replaced.

A separate fixed synthetic check exposed a **bridge numerical boundary**: a feature exactly
contained in the baseline is labeled predictively useful from a roughly 1.44e-16 rounding-only
RMSE difference; its mathematically zero residual also receives finite correlations. The original
bridge is unchanged and the full counterexample is retained. The
[new v3 bridge consumer](reports/engineering-archive/dense-v3-bridge-v1/README.md) now corrects this
under a separate numerical amendment: exact rational OLS/error comparisons, explicit redundancy/
resolution/extrapolation handling and no arbitrary zero-residual correlation. Both old null cases
give exact zero gain. All nine original feature values and mathematical prediction rules remain.
Independent full systems match all 540 held-out predictions, 36 fold comparisons, nine pooled
decisions and eighteen residual association values, followed by complete fresh CPU replay.
All positive panels are synthetic. The first independent reference-helper failure and its narrow
type-compatibility correction are preserved; no analysis tolerance changed. The v3 adapter is
prepared, not deployed, and exact-feature/dimension sensitivity remains separate work.

BEIR's exact three dispatch handles remain paused with their original ledger, lease and completed
prefix. All diagnostic launchers have exited; production numerical source and checkpoints are
unchanged. Next review the [replication transition proposal](reports/engineering-archive/dense-correction-candidate-v1/replication-transition-proposal.md),
review the [completed revised inputs and bounded natural preflight](reports/engineering-archive/dense-revised-natural-v1/README.md),
then integrate the separately declared exact-sensitivity/dimension/publication consumers, the separately routed
factorial control and the evidence-preserving runtime handoff. The old v2 consumers still bind old
data; do not merely refresh old hashes. Resolve the existing deployment/publication gates. Do not resume
a competing controller, use the obsolete zero-step migration or promote
existing scores into paper conclusions. No implementation incident belongs in the manuscript.

The independent ten-run HF audit covers 1,010 files at original immutable commits; automatic
60-checkpoint backup coverage does not extend that separate audit. Last verified raw geometry
covers ten runs/50 stages. Accepted retrieval, dimension-utility and factorial outputs remain
pending. New source/evidence is still isolated and unpushed because WIP-publication permission is
unresolved. [Issue #41](https://github.com/qcznlp/embedding-optimizer-study/issues/41) is the public
progress channel, but the integration returned 403 on its latest write attempts. This milestone is
local only; no alternate credentials or identity were used to bypass the rejection.

- Human-readable source of truth: [PROJECT_STATUS.md](PROJECT_STATUS.md)
- Latest committed machine-readable snapshot: [CURRENT_PROGRESS.json](CURRENT_PROGRESS.json)
- Exact run/evaluation/backup commands: [docs/dense-no-packing-retrain.md](docs/dense-no-packing-retrain.md)
- Frozen scientific contract: [configs/dense_no_packing_execution_protocol.json](configs/dense_no_packing_execution_protocol.json)
- Frozen reload/evaluation contract: [configs/dense_no_packing_evaluation_protocol.json](configs/dense_no_packing_evaluation_protocol.json)
- Frozen weight-space/bridge contract: [configs/dense_no_packing_analysis_protocol.json](configs/dense_no_packing_analysis_protocol.json)
- Frozen dimension-utilization contract: [configs/dense_dimension_utilization_protocol.json](configs/dense_dimension_utilization_protocol.json)
- Dimension analysis and interpretation: [docs/dimension-utilization.md](docs/dimension-utilization.md)
- Frozen outcome/statistical contract: [configs/dense_no_packing_outcome_protocol.json](configs/dense_no_packing_outcome_protocol.json)
- Source-bound bridge implementation: [configs/dense_no_packing_bridge_implementation_protocol_v2.json](configs/dense_no_packing_bridge_implementation_protocol_v2.json)
- Historical/corrected sensitivity implementation: [configs/dense_no_packing_sensitivity_implementation_protocol.json](configs/dense_no_packing_sensitivity_implementation_protocol.json)
- Source-bound paper renderer: [configs/dense_no_packing_publication_protocol.json](configs/dense_no_packing_publication_protocol.json)
- Prospective state×operator mechanism follow-up: [docs/state-operator-factorial.md](docs/state-operator-factorial.md)
- Its frozen contracts: [scientific](configs/dense_no_packing_state_operator_factorial_protocol.json), [implementation](configs/dense_no_packing_state_operator_factorial_implementation_protocol.json), [paper rendering](configs/dense_no_packing_state_operator_factorial_publication_protocol.json), [200-word abstract migration](configs/dense_no_packing_state_operator_abstract_compliance_migration.json), and [completion handoff](configs/dense_no_packing_state_operator_factorial_completion_protocol.json)
- Resume-safe mechanism handoff: `python -m embed_optim.state_operator_factorial_completion --resume`
- Active control-plane recovery receipt: [configs/dense_no_packing_control_plane_recovery.json](configs/dense_no_packing_control_plane_recovery.json)
- Read-only corrected W&B audit: `python -m embed_optim.corrected_wandb_audit --allow-partial`

On the experiment host, run the artifact-only snapshot from the **experiment checkout**
(`/root/embedding-optimizer-study` here), not the isolated paper checkout. The matrix's relative
output paths are resolved against the working directory. No system processes are inspected:

```bash
python -m embed_optim.corrected_progress --output CURRENT_PROGRESS.json
```

The committed JSON is a handoff snapshot, not a heartbeat. Run the command above for the freshest
local state. Meaningful transitions—completed/failed runs, evaluation milestones, verified
backups, or changed scientific conclusions—must also be recorded in `PROJECT_STATUS.md` and issue
#41. The external `gpu.py` utility and its processes are strictly out of scope.

- Dense-only NAACL plan: [docs/naacl-dense-paper-plan.md](docs/naacl-dense-paper-plan.md)
- Authoritative manuscript: [paper/](paper/README.md)
- Live dashboard: [Weights & Biases](https://wandb.ai/stevezenguom/embedding-optimizer-study)
- Public checkpoint backup: [qcz/embedding-optimizer-study-checkpoints](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints)
- Public analysis-artifact backup: [qcz/embedding-optimizer-study-analysis-artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts)

Engineering incident records and superseded exploratory outputs remain available in the repository,
but the manuscript imports only the primary matrix and its source-bound weight-space, dimension,
retrieval, and factorial analyses.

## Verify a clean clone

The Git repository includes a 107.4MB minimal, content-addressed paper-evidence closure; it does not
duplicate the 416GB public checkpoint archive. A new contributor or agent can therefore audit the
historical evidence immediately after installing the development dependencies. The strict final-paper
command is expected to fail on this draft until the complete primary results and their evidence
closure exist:

~~~bash
uv sync --extra dev --extra eval --extra analysis
uv run python scripts/portable_evidence.py --audit-only
uv run embed-optim-audit-paper \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
uv run pytest
~~~

The first command rehashes all 2,785 included evaluation files and reconstructs the exact expected
closure from four source manifests. On the experiment host, where `outputs/` is present, the Dense
five-stage audit keeps using the full checkpoint-backed reconstruction. In a clean clone, it uses
the explicit portable closure and reports that narrower provenance boundary.

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

## Historical exploratory evidence — not primary findings

The results and generated conclusion block in this section belong to the superseded historical
experiments. They are preserved for audit, not imported into the rewritten manuscript. The primary
matrix's scientific validation is still incomplete; neither a final optimizer ranking nor a weight-space explanation is
established. Consult the current [integrity audit](reports/experiment-integrity/README.md) for checked
coverage and remaining work.

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

Audit the complete study status without conflating those two evaluation contracts:

```bash
embed-optim-evaluation-progress \
  --study-root "$PWD" \
  --output logs/evaluation/study-live-audit.json
```

This mode provenance-validates the exact 1,750-unit DenseOn BEIR grid, reports discovery, hybrid,
and each confirmatory seed separately, shows progress through the 31-step completion and 18-step
finalization controllers, and verifies the current source-addressed candidate-breadth paper
artifact. It also reports the 45 query-disjoint and 46 unseen shared-start probe jobs as a separate
mechanism contract; controller counts are informational and never replace their own
content-addressed audits.

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

The completed 14-task routing control makes the main optimizer comparison essentially unchanged.
Across all four AdamW learning rates, routing hidden matrices separately and fixing the auxiliary
AdamW rate changes mean nDCG@10 by only +0.000077. Three learning-rate contrasts are positive and one
is negative; the 56 task-by-learning-rate units contain 28 wins, 6 ties, and 22 losses. Parameter
routing is therefore too small and inconsistent to explain the Muon-family discovery or selection
effects, although this control does not isolate orthogonalization from update scale.

The earlier exploratory weight-space observations were:

- a Frobenius-matched one-step margin proxy is smaller for Muon than AdamW, but this does not rank
  either optimizer's step in general;
- across the four frozen learning-rate points, Muon's one-seed final median unseen margin and BEIR
  exceed AdamW by 0.0110 and 0.00435;
- in the post-hoc fixed-state intervention, the adverse query tail is redistributed rather than
  uniformly dominated;
- Muon-family endpoints move about twice as far in hidden-weight space for similar fixed-probe score
  drift; and
- spectral flattening alone has little or no anchor-level association with tail protection.

Shared-start branches show that accumulated paths differ, while frozen spectrum-versus-basis
transplants reject the obvious signature-as-mechanism account. The prospective crossed reset
experiment separates the value of the reached weight state, the next optimizer, and their
interaction. These historical observations do not supply the manuscript's primary evidence.

### Archived historical conclusion — superseded

This retained generated block is not the current paper conclusion and must not be cited as the
primary optimizer verdict.

<!-- FINAL-CONCLUSION:BEGIN -->

Validation-frozen three-seed DenseOn: Muon versus AdamW: negative, -0.0306 nDCG@10 (familywise 95% CI [-0.0464, -0.0138]); NorMuon versus AdamW: negative, -0.0304 nDCG@10 (familywise 95% CI [-0.0446, -0.0138]). The routing-matched hybrid AdamW minus native AdamW averaged +0.0001 across DenseOn's four rates (3 positive, 1 negative, and 0 zero learning-rate points), descriptive evidence about parameter routing as an alternative explanation. Frozen shared-start tail endpoints for DenseOn: Muon: mixed; NorMuon: mixed. The temporal bridge was a claimable negative, the fixed-state chain was a claimable negative, and their joint spectral-component account was a claimable negative; this rejects only the tested mechanism, not formal mediation or a universal optimizer ranking.

<!-- FINAL-CONCLUSION:END -->

## Historical experimental contract

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
| Compute | Eight NVIDIA L20Z GPUs, split into two disjoint four-GPU pools |

Muon and NorMuon operate on the 88 two-dimensional Transformer hidden matrices
(110,297,088 parameters). Embeddings, pooling projection, norms, and biases use auxiliary AdamW at
3e-6. Hybrid AdamW reproduces that routing and measures its effect within AdamW. Its comparison with
Muon-family runs is a separately tuned, matched-routing recipe comparison—not a scale-matched
identification of orthogonalization alone.

## Repository layout

| Path | Purpose |
| --- | --- |
| configs/dense_no_packing_retrain.yaml | Current 12-run DenseOn primary matrix |
| configs/experiment.yaml | Frozen historical discovery matrix; use an explicit Dense family filter |
| configs/dense_scope_amendment.json | Active scope and strict expected counts |
| configs/dense_training_queue.json | Frozen 18-run confirmation/short-branch queue |
| configs/dense_retrieval_dynamics_extension.json | Source-bound five-stage hybrid/confirmation extension |
| src/embed_optim/ | Training, evaluation, optimizers, audits, reports, and interventions |
| scripts/eval/dense_parallel.py | Eight-GPU dense retrieval evaluator |
| paper/ | Authoritative ACL/NAACL manuscript and generated result tables |
| reports/ | Content-addressed summaries and publication figures |
| configs/portable_paper_evidence.json | Exact minimal evidence closure for clean-clone auditing |
| tests/ | Unit, integration, provenance, distribution, and numerical regression tests |

## Restore large artifacts on another machine

The Git repository deliberately excludes large model and analysis state. Restore only the current
primary namespace at each run's **verified immutable revision**, using the content audit as its
index. Do not download the entire HF repository at a moving revision: that also retrieves historical
experiments excluded from the paper. Exact current coverage is in [PROJECT_STATUS.md](PROJECT_STATUS.md).

Follow [the checkpoint restoration guide](docs/checkpoint-restoration.md) for complete-run downloads,
a smaller 1.35 GB example, verification and analysis dependencies. The example was actually fetched
from HF into a fresh directory and passed offline CPU encoding and optimizer/scheduler restoration.
It is not a physical second-host or distributed-training resume test. The new diagnostic source is
still isolated pending source-publication approval; the guide makes that availability boundary explicit.

On a live experiment host, an already sealed intermediate corrected checkpoint can be preserved
before its run finishes with `python -m embed_optim.incremental_checkpoint_backup --run-ids ...
--steps ...`. This is a digest-verified durability operation, not evidence of run completion; exact
commands and receipt semantics are documented in
[the corrected retraining handoff](docs/dense-no-packing-retrain.md).

For an unattended campaign, `embed-optim-supervise-sealed-checkpoint-backup` watches the frozen
five-stage schedules and performs the same digest-verified upload as soon as each checkpoint is
sealed. It is CPU/network-only, holds an exclusive lease, is source-bound, gives the existing
whole-run completion controller priority at the final stage, and records
`scientific_completion=false` throughout. It neither launches nor controls training.

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

## Historical reproduction commands — do not use for the active matrix

The commands from this section through the historical finalizer reproduce completed exploratory
work. They are not the current training/evaluation queue. For the active 12-run primary matrix,
follow [the primary runbook](docs/dense-no-packing-retrain.md) and its existing controller; do not
launch a competing sweep or overwrite historical outputs.

### Run the DenseOn discovery sweep

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

## Run the post-hoc candidate-breadth diagnostic

The source-bound three-regime diagnostic reconstructs the final discovery training loss,
query-disjoint validation metrics, and full-corpus BEIR score for all 12 DenseOn discovery runs:

~~~bash
embed-optim-three-regime-diagnostic \
  --protocol configs/three_regime_diagnostic.json \
  --output-dir reports/three-regime-diagnostic
embed-optim-three-regime-diagnostic \
  --protocol configs/three_regime_diagnostic.json \
  --output-dir reports/three-regime-diagnostic \
  --audit
~~~

Its checked-in [report](reports/three-regime-diagnostic/README.md) is explicitly post hoc and cannot
alter the three-seed optimizer comparison or substitute for the nested candidate-breadth test.

This diagnostic is deliberately separate from the frozen three-seed comparison. It asks whether the
Muon-family validation ordering changes when the same 224 query-positive pairs are scored against
nested sets of 7, 10, 32, 128, 512, and 2,048 mined negatives. The width-7 slice must first reproduce
the original query-disjoint validation outputs within the frozen sample-level tolerance. A structural
source or checkpoint mismatch still fails hard. A numerical reproduction failure is retained as a
falsification result, forces the frozen decision to `not_supported`, and prevents the wider slices
from being interpreted as a causal bridge.

The completed audit did fail that prerequisite: the maximum width-7 sample/metric discrepancy is
8.286419 against a tolerance of 1e-5. A controlled two-row check localizes the problem to the
historical flattened/packed SentenceTransformers path: changing packed batch composition changes a
cosine score by as much as 0.211914, while the padded control changes it by at most 0.001953 in BF16.
On padded width 7, the high-dose Muon-family advantage is already absent, and widening to 2,048
candidates does not yield the required reversal. The missing-candidate explanation is therefore not
supported; the historical selection result is implementation-confounded.

Prepare and independently audit the source-bound nested candidate data:

~~~bash
embed-optim-prepare-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json \
  --output data/candidate-breadth-224-seed20260901 \
  --resume

embed-optim-prepare-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json \
  --output data/candidate-breadth-224-seed20260901 \
  --audit-only \
  --receipt reports/candidate-breadth/data-audit.json
~~~

The release audit does not trust hashes reported by the generated manifest. It reselects the 224
rows from the frozen validation ledger, reconstructs all 2,048 negative IDs from the pinned mined
score revision, verifies the source document text, and compares every materialized query and
candidate row. Checkpoint evaluators repeat the complete local file and row audit but reuse that
one release-gated upstream reconstruction, avoiding twelve redundant scans of the source parquet
files.

Evaluate all 12 discovery final checkpoints, then build and re-audit the frozen decision summary:

~~~bash
embed-optim-candidate-breadth-matrix \
  --protocol configs/candidate_breadth_probe.json \
  --source-audit-receipt reports/candidate-breadth/data-audit.json \
  --gpus 0,1,2,3,4,5,6,7

embed-optim-summarize-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json

embed-optim-summarize-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json \
  --audit-only

embed-optim-render-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json

embed-optim-render-candidate-breadth \
  --protocol configs/candidate_breadth_probe.json \
  --audit-only

python -m embed_optim.packing_invariance \
  --device cuda
python -m embed_optim.packing_invariance \
  --audit-only
~~~

Matrix resume is evidence-preserving rather than a manifest-only shortcut: every existing run reloads
its `scores.npz`, checks the exact query and width axes, recomputes all sample and source aggregates,
byte-compares both JSONL outputs, and reruns the width-7 frozen-baseline check before it may skip model
inference.

For the publication handoff, use the single post-hoc release controller after the canonical Dense
finalizer has completed and its story changes have been integrated:

~~~bash
embed-optim-candidate-breadth-release \
  --upstream-finalization-ledger logs/dense-finalization-pipeline/pipeline-ledger.json \
  --protocol configs/candidate_breadth_probe.json \
  --gpus 0,1,2,3,4,5,6,7 \
  --workdir "$PWD" \
  --resume
~~~

The controller requires the exact 18-step upstream ledger, rehashes its completion source and every
attempt log, and binds that immutable historical release into a new ledger. It intentionally does
not compare the historical implementation hashes with the post-hoc checkout: adding the frozen
candidate diagnostic changes those source bytes without invalidating the already completed formal
experiment. Before and after every new step it nevertheless rehashes both ledgers, the candidate
protocol, and the complete current source/command contract. It prepares or audits the data,
content-resumes the 12-checkpoint matrix, rebuilds the current result blocks, renders the candidate
figure and publication text, and reruns the paper, test, style, build, and distribution gates.

The primary rule was frozen before any candidate-breadth data or scores were visible. Its decision
is labelled supported only if all 12 width-7 reproductions pass and both Muon and NorMuon reverse
their high-dose-versus-retrieval-optimal loss and margin ordering by width 2,048. Attenuation without
reversal is reported separately, and an unchanged anti-calibrated ordering falsifies this proposed
account. The observed prerequisite failure also forces `not_supported`; the later padded widths are
reported as an explicitly post-failure diagnostic rather than as a successful nested bridge. A
passing rule would have been consistent with missing-candidate coverage contributing to the gap; the
post-hoc diagnostic does not establish that contribution causally and cannot replace the formal
full-corpus optimizer comparison. Paired loss and margin contrasts additionally receive descriptive
95% source-stratified paired percentile bootstrap intervals: 50,000 resamples independently preserve
each of the seven fixed 32-query source strata. This uncertainty plan was recorded before candidate
data or scores were visible and does not enter the frozen decision rule. The summary writes audited
calibration and paired-contrast tables plus publication-ready SVG/PDF panels under
`reports/candidate-breadth/`. The final archival step updates
`reports/engineering-archive/candidate-breadth-paper-fragment.tex` and verifies it against a
content-addressed manifest. This engineering diagnostic is not an input to the scientific paper.

## Render the final Dense-only deliverables

After the Dense completion ledger passes, the canonical, resume-safe finalizer regenerates every
scoped evidence report, audits the paper, runs the local quality gates, builds the paper and Python
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
The candidate-breadth release command above is the only additional controller needed after this
canonical finalizer; it treats this finalization ledger as immutable upstream evidence and does not
repeat the already audited W&B mutations.

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
