<!--
This README substantially modifies the lightonai/mdenseon-mlateon README at commit
b0db47a48f969d825446668b5b17bfc27a359fc1. See THIRD_PARTY_NOTICES.md.
-->

# Optimizers, weight trajectories, and dense retrieval

Research code for comparing AdamW, Muon and NorMuon when adapting
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised).
The intended deliverable is a NAACL paper with reproducible model and analysis artifacts.

<!-- FINAL-CONCLUSION:BEGIN -->
**Results pending (FINAL_CONCLUSION_PENDING).** The paper's optimizer conclusions await complete
retrieval, functional and crossed-continuation evidence. Historical results, numerical diagnostics
and passing tests do not substitute for those findings.
<!-- FINAL-CONCLUSION:END -->

**Current execution:** all **12 corrected v3 training runs are complete**, and all **60 checkpoints**
are HF-verified. All **840 checkpoint/task retrieval evaluations are complete**;
the [complete five-stage curves and tables](reports/dense-v3-complete-trajectories-v1/README.md)
have been generated and numerically reproduced from copied source. Crossed
continuation outcomes and manuscript/source release remain incomplete.
The [new real-execution milestone](reports/engineering-archive/dense-v3-real-experiment-advance-v1/README.md)
now completes **all 61 vector states and both genuine GPU calibrations**.
At **2026-09-12 20:07 UTC**, **all 12/12 crossed continuations are complete**;
all sixty checkpoints passed native reading and **60/60** are HF-verified.
Both training supervisors and the backup coordinator exited zero.
Both BEIR pilots passed: **2/168** tasks complete, with six full-corpus GPU
workers active at the dated observation.
Two [five-stage probe waiters](reports/engineering-archive/dense-v3-factorial-probe-v1/README.md)
are now running: **25/60** new checkpoint probes and one verified reused reference
at that snapshot. Each uses one released GPU after complete-pool training.
The [final statistical handoff](reports/engineering-archive/dense-v3-factorial-summary-v1/README.md)
is also queued: it waits for complete actual outcomes before computing the
three unchanged contrasts and independently checking their intervals.
The [actual primary/functional manuscript-result fragments](reports/dense-v3-manuscript-results-v1/README.md)
now reconstruct exactly from copied source/data, retaining all optimizer
contrasts and all 108 exploratory comparator cells. They are not installed as
final paper results; complete continuation evidence and publication gates remain.
The [complete current primary scientific consumer](reports/engineering-archive/dense-v3-current-publication-consumer-v1/README.md)
now joins genuine native primary, weight and functional evidence for all
12 primary runs / 60 states / 840 BEIR results / 12 validations and regenerates
the original scientific outputs. Its [full development paper](reports/engineering-archive/dense-v3-current-publication-consumer-v1/actual/paper-preview-v2/build/main.pdf)
includes all-state empirical weight-to-retrieval maps and every learning-rate
trajectory. All twelve PDF pages were inspected; main text ends on page eight
with embedded non-Type-3 fonts. Factorial findings and final release remain pending;
the old abstract/layout reader failures are preserved, not waived.
Two [native GPU-resume checks](reports/engineering-archive/dense-v3-gpu-resume-v1/README.md)
are queued after their pools' scientific GPU work. Both real checkpoint downloads
are complete: **36 files / 3.14 GB** pass original hashes and native CPU reading.
Actual GPU endpoint equivalence remains pending; the checks do not delay evaluation.
The [functional inference](reports/dense-v3-functional-inference-v1/README.md)
now includes all **61 feature states, nine primary contrasts and 240 independently
verified predictions**. Native helpful participation is slightly higher for Muon,
but not rotation-stable or predictive under the declared criterion. Degrading
attribution mass improves held-dose prediction, with a positive association;
its optimizer difference is inconclusive. This is not a mechanism claim.
The [subsequent functional controls](reports/dense-v3-functional-sensitivity-v1/README.md)
retain all 24 comparisons: no functional predictor passes all four recipe
baselines. All 1,440 predictions are independently verified and numerically replayed.
The original functional inference also now numerically replays from copied source:
all nine table families, decisions and generated figure/text source match exactly.
These are dated observations, not a perpetual heartbeat. See the current handoff
for the active training/backup entries; do not restart completed primary work.
All [functional and calibration payloads](docs/functional-analysis-restoration.md)
are now backed up: **624 files / 7.5 GB**, anonymously downloaded from an immutable
revision and independently checksum-verified. The [latest operational archive](reports/engineering-archive/dense-v3-current-publication-consumer-v1/README.md)
preserves complete training, current native scientific joins, actual paper previews,
backup receipts and the active downstream handoff.
The [complete weight-to-retrieval prediction readout](reports/dense-v3-weight-retrieval-v1/README.md)
now includes all nine original and five separate exact features, independently
verified predictions and all-feature figures. Displacement magnitude predicts
held-dose variation under the declared baseline; stable-rank features do not
meet the criterion. This is not a functional or causal explanation.
The [subsequent post-result comparator sensitivity](reports/dense-v3-predictor-sensitivity-v1/README.md)
shows that these predictive conclusions depend on baseline specification:
raw learning-rate controls remove the large displacement gain, and no feature
passes all four tested recipe baselines. All 84 comparisons are retained; these
are exploratory qualifications, not revised endpoint results or a causal mechanism.
Both numerical analyses now have a [verified immutable data backup](docs/predictor-analysis-restoration.md):
all 47 files recovered anonymously, with independent reconstruction of errors
from 840 original and 5,040 exploratory predictions. Source remains local WIP.
The [trajectory recovery guide](docs/retrieval-trajectories-restoration.md) restores
the public immutable tables/figures and indexes all five raw-score snapshots.
The current results are under `verified-v3-*`,
not the older scientifically held campaign. Training completion is not paper completion.

Start with [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md) for a concise, dated status, exact source
locations and the remaining work. [PROJECT_STATUS.md](PROJECT_STATUS.md) retains the detailed history;
agents must follow [AGENTS.md](AGENTS.md). Use the [current checkpoint recovery guide](docs/checkpoint-restoration.md)
to download exact immutable revisions and verify their files on another machine.
The [weight-analysis recovery guide](docs/weight-analysis-restoration.md) addresses
the separate complete numerical snapshot: 298 files / 4.98 GB, not WIP source code.
The [paper directory](paper/README.md) contains the scientific narrative and release requirements.
This checkout still contains unpublished WIP; a clean remote clone does not yet contain the complete
current source and evidence. No committed-source release is claimed.

## Research question

Starting from the same pretrained retriever, do the optimizers reach different retrieval-quality
regions, and which weight-state changes have functional value?

The study connects three measurements: saved weight trajectories, coordinate-level representation
utility, and full-corpus retrieval. A separate crossed state-by-operator continuation tests how the
reached state changes subsequent optimization. Muon's characteristic spectrum is not itself the
contribution; an explanation must survive held-out prediction and appropriately bounded controls.

## Declared primary experiment

| Component | Specification |
| --- | --- |
| Initialization | DenseOn-unsupervised, revision 0edbd55684eb782bce55ee74c95b25c97cbe7f43 |
| Data | Same revised 500,000 queries for every run; deterministic selection and negatives |
| Objective | One positive + seven hard negatives; no in-batch negatives; temperature 0.02 |
| Schedule | One epoch, global batch 128, maximum context 8192 |
| AdamW rates | 1e-6, 3e-6, 1e-5, 3e-5 |
| Muon / NorMuon rates | 1e-4, 3e-4, 1e-3, 3e-3 |
| Retained steps | 782, 1563, 2345, 3126, 3907 |
| Retrieval evaluation | Fourteen pinned decontaminated BEIR tasks; 840 checkpoint/task units |
| Primary comparison | Average all four rates per optimizer; paired-task simultaneous intervals |
| Secondary selection | Frozen 4,096-query validation loss, never BEIR test scores |

The active version is [the v3 primary contract](configs/dense_primary_v3_protocol.json), with explicit
data and implementation parents. Its original draft/released-source guards are preserved. The owner
separately authorized the content-bound twelve-run execution, deferring code publication; that
training is now complete. This exception does not release a new controller, analysis or paper.
Exact model/data revisions, runtime and checkpoint identities remain required, not optional metadata.

Training used eight NVIDIA L20Z GPUs in two disjoint four-GPU pools; this is not a claim of literal
H100 hardware. Completed evaluation workers and their source-bound handoffs must not be restarted
or replaced using commands from an older archived campaign.

DenseOn is the only active architecture. The [scope amendment](configs/dense_scope_amendment.json)
retains historical LateOn artifacts for provenance, without new LateOn experiments or pooled
architecture claims. All rates are reported; learning-rate cells are not independent training seeds.

## Weight space to functional utility

Weight analysis retains every scheduled state, hidden matrix and declared rate. The original
approximate measurements and separately named exact-spectrum/projector counterparts remain
distinct, with explicit denominators and undefined subspaces.

The functional probe has 224 fixed queries, balanced across fourteen tasks, each with eight ordered
candidates. Analysis deletes each of 768 coordinates, applies 80 shared random-removal masks, and
repeats endpoint measurements under three shared rotations. Helpful/degrading contributions are
computed after cosine re-normalization. This shortlist analysis is not full-corpus BEIR.

Each proposed geometry or functional predictor is evaluated outside its learning-rate dose,
controlling for optimizer, checkpoint stage and within-optimizer rate. Every predictor is reported,
including null or unidentified ones. Predictive support is not causal mediation; sampled-rotation
robustness is not a proof of arbitrary-basis invariance.

Both complete primary weight-geometry branches, the update map, all retained retrieval
outcomes and the locked weight-to-retrieval predictive calculations have been computed.
Functional inference and richer recipe controls are complete and independently
checked. Crossed outcomes and a supported mechanistic explanation remain
incomplete; see the dated
[current evidence table](CURRENT_EXPERIMENT.md#goal-and-current-outcome).

<details>
<summary>Archived component checks — not current experiment completion or scientific findings</summary>

The [functional inference preparation](reports/engineering-archive/dense-v3-dimension-inference-v1/README.md)
connects the task-family inference and four genuinely named predictors, with independent statistical
and exact-regression checks. The [latest source/transport milestone](reports/engineering-archive/dense-v3-portable-reconstruction-v1/README.md)
loads actual contracts after relocation and verifies local authoring transport with explicitly
simulated upstream admission. The [raw-vector reader](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
also passes full-768D reconstruction of all 61 synthetic states in two relocated processes, with
exact tables/arrays and numerical mutation refusals. The [original-outcome reader](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
also reconstructs all raw validation records, 840 synthetic task units and ten original tables in
two cold processes, with 21 semantic refusals each. No primary model was encoded. Original
bridge/inference reconstruction now passes both complete joint cold audits; its bounded
acceptance is separate from still-incomplete primary publication. The original geometry
reader now also passes both complete native-shape cold audits. These are not optimizer findings. Choices are retained
in [the dimension protocol](configs/dense_dimension_utilization_protocol.json).

</details>

## Develop and verify

From the intended checkout root:

```bash
uv sync --extra dev --extra eval --extra analysis
export PYTHONPATH="$PWD/src:$PWD"
CUDA_VISIBLE_DEVICES='' uv run python -B -m pytest -q
```

The active primary assembly has 56 source-bound files and passed its 104 relevant training tests,
in addition to real four-GPU/full-model checks. The newer factorial components have separate,
explicitly bounded tests; their CPU passes do not authorize GPU continuation. Do not run a
repository-wide installation/update inside a live worker's frozen environment.

<details>
<summary>Retained development and reconstruction test history</summary>

The latest completed isolated full regression exits zero: JUnit reports 3,065 tests,
zero failures/errors/skips and 3,056 serialized testcase elements. The preserved
count discrepancy is described in the factorial-routing execution record. The new
[publication preparation](reports/engineering-archive/dense-v3-primary-publication-v1/README.md)
passes 63 focused tests, complete synthetic generation/layout checks, the full regression
and bounded preservation acceptance. It refuses missing primary runs and simulated admissions,
preserves raw functional inference, and does not install results in the manuscript.
The [joint reader](reports/engineering-archive/dense-v3-joint-reconstruction-v1/README.md)
has passed its first complete synthetic cold audit, including all 21 semantic controls;
independent complete replay also passes with all 477 outputs exact and all 21 controls.
The joint archive's generated validation.json verifies its bounded synthetic acceptance,
not primary admission or complete publication. The [geometry reader](reports/engineering-archive/dense-v3-geometry-reconstruction-v1/README.md)
passes 67 focused cases and both complete native-shape cold audits, including all 28 rehashed
refusals on each attempt. These are synthetic reconstruction checks, not primary results.
The [original-outcome reader](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
has 62 passing focused cases and two complete passing cold audits; its first test/launcher
failures are preserved, with no production-kernel changes. The [raw-vector reader](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
has 38 focused checks and two passing full-width cold audits with no numerical tolerance. The first
full-suite test-isolation failure and all prior numerical/source evidence remain preserved. On this host the
pinned interpreter is `/usr/bin/python`; the old live `.venv`
has different versions. The separate training identity candidate retains eleven unwaived source-
contract failures among 963 cases. These source trees and runtime scopes are not interchangeable.

The newer [exact-sensitivity integration](reports/engineering-archive/dense-v3-exact-reconstruction-v1/README.md)
has 50 passing focused tests, exact readback of 1,584 retained real diagnostic
spectrum records, and complete independent synthetic geometry/SymPy checks.
Both complete cold reconstructions, the 2,713-case regression and bounded acceptance
now pass. All 503 outputs match across the two attempts. The newer
[exact publication entry](reports/engineering-archive/dense-v3-exact-publication-v1/plan.md)
has 95 passing focused checks, two complete synthetic generations with identical
outputs, and a passing complete v3 layout (main end page 7). Its new whole regression
has finished as 7103, exit 0; no handle in that archive remains live.
None of these engineering checks supplies a primary optimizer finding.

The newer [active document component](reports/engineering-archive/dense-v3-manuscript-consumer-v1/README.md)
adds 154 passing focused checks and a fresh complete synthetic PDF: exact generated
inputs, actual 166-word abstract, all nine captions, main endpoint page 7, and embedded
non-Type-3 fonts. The old nominal abstract reservation is not treated as an actual
per-macro bound. Whole regression 5462 is terminal/exit 0. This is not complete primary/factorial
admission or a release entry; the real manuscript remains pending and unchanged.

</details>

The formal GPU environment is specified separately by [formal_runtime.json](configs/formal_runtime.json),
[requirements-formal.lock](requirements-formal.lock), and
[requirements-formal-flash.txt](requirements-formal-flash.txt). A development installation is not
presented as a verified formal runtime. Do not launch old controller or training commands from
archived instructions; current release, source and owner-approval gates must be satisfied first.

For exact CPU-only diagnostic commands and retained output hashes, use the current
[outcome reproduction record](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/commands.md)
and [vector reproduction record](reports/engineering-archive/dense-v3-vector-reconstruction-v1/commands.md).
Use new output directories. These commands neither train models nor supply accepted primary results.

## Manuscript and release

The [manuscript](paper/main.tex) is a draft with visible pending results. To compile the existing
draft without generating experimental findings:

```bash
make -C paper
```

A final release requires real generated evidence and the strict gates:

```bash
make -C paper release
python -m embed_optim.paper_audit --strict --families dense \
  --scope-amendment configs/dense_scope_amendment.json
```

These gates are expected to reject an incomplete draft. Do not remove placeholders or update
output hashes to bypass them. The final repository must also pass distribution checks and
clean-clone reconstruction of its accepted publication evidence. A portable summary audit does
not re-encode models or revalidate checkpoint payloads.

For the historical artifact workflow, the 34 frozen Dense source runs retain their tracking
provenance gate: `embed-optim-audit-wandb-dense-sources` precedes `embed-optim-sync-wandb`, which
precedes `uv build`; distribution auditing uses `--include-wandb`. See the full
[completion contract](docs/completion-gates.md) for exact receipts. These are ordering requirements,
not permission to publish WIP or a substitute for a separately reviewed v3 release.

## Artifacts and navigation

- [Concise current experiment and next actions](CURRENT_EXPERIMENT.md)
- [Detailed chronological evidence](PROJECT_STATUS.md)
- [Checkpoint restoration and integrity](docs/checkpoint-restoration.md)
- [All 60 immutable checkpoint locations and file digests](docs/primary-v3-checkpoints.json)
- [Current complete weight-analysis snapshot and recovery](docs/weight-analysis-restoration.md)
- [Crossed state-by-operator design](docs/state-operator-factorial.md)
- [Engineering evidence archive](reports/engineering-archive/)
- [Checkpoint repository](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints)
- [Analysis artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts)
- [Tracking issue](https://github.com/qcznlp/embedding-optimizer-study/issues/41)
- [Weights & Biases](https://wandb.ai/stevezenguom/embedding-optimizer-study)

Restore exact immutable revisions and verify payload digests; do not download mixed historical
namespaces and infer primary eligibility from filenames. HF withdrawal and GitHub publication are
subject to the explicit outstanding safety/access gates in [AGENTS.md](AGENTS.md). No credential
belongs in source or released logs.

The former 957-line README, including historical numbers and reproduction commands, is preserved
verbatim in [the chronological archive](reports/engineering-archive/dense-v3-dimension-interventions-v1/before/README.md).
It is not the default execution guide and does not provide accepted primary findings.

## License and attribution

Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
[CITATION.cff](CITATION.cff), [CONTRIBUTING.md](CONTRIBUTING.md),
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [SECURITY.md](SECURITY.md).
