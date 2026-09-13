# Project status and continuation

Updated: **2026-09-07 03:25 UTC**. This is local preparation, not a released experiment.

## Bottom line

**The project is still before the new valid primary experiment.** The old 12 training runs and
60 scheduled checkpoints finished and are preserved, but remain on scientific hold. No complete
v3 run, accepted primary optimizer contrast or finished NAACL paper exists.

Prepared data, training-identity consumers, validation, outcome aggregation, geometry and retrieval
bridges pass their bounded checks. The portable raw-vector reader passes both complete full-width
cold audits, including independent replay and numerical refusals. All 2,419 isolated regression
tests pass after preserving and resolving a test-isolation failure. Next are the original outcome
local-role reader (its raw fixture is prepared), geometry/inference reconstruction and primary publication,
followed by reviewed release and the real experiment. Tests are not scientific findings.

## Study and paper target

Compare AdamW, Muon and NorMuon from the same pinned DenseOn-unsupervised initialization, on one
common deterministic revised 500k-query view, with one positive and seven random-seeded hard
negatives, no in-batch negatives, maximum length 8192, one epoch and global batch 128.
Each optimizer has four declared rates:

| Optimizer | Learning rates |
| --- | --- |
| AdamW | 1e-6, 3e-6, 1e-5, 3e-5 |
| Muon / NorMuon | 1e-4, 3e-4, 1e-3, 3e-3 |

The five retained steps are 782, 1563, 2345, 3126 and 3907. Full evaluation is 12 × 5 × 14 =
840 decontaminated BEIR checkpoint/task units, plus frozen validation selection. The primary
comparison averages all four rates; validation-selected recipes are secondary. Task resampling
does not estimate training-seed variability. Observed hardware is L20Z, not literal H100.

The paper's spine is weight trajectories -> representation utility -> out-of-dose retrieval
prediction, followed by a bounded state-by-operator reset continuation. Intrinsic Muon geometry
is not a novel retrieval finding. Coordinate utility needs deletion and rotation controls;
predictive associations are not causal mediation. A negative mechanism result must also be reported.
DenseOn is the only active architecture; the paper is the sole article deliverable.
Implementation incidents are excluded from every manuscript section.

## Latest completed work

[Raw-vector reconstruction work](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
has reached the following **bounded numerical verification** boundary:

- Implements exact reconstruction of all 61 states from externally authenticated FP32 vectors,
  retaining every 768-coordinate deletion, random mask, endpoint rotation and FP64 attribution.
- 38 focused tests, a 49-case ordered reproducer and the full **2,419-case retry** pass. The first
  full attempt's 17 failures are preserved: old test snapshots polluted the import registry and
  were correctly refused. New test isolation was fixed; production/source gates were not weakened.
- Both full-width cold CLI audit **72073** and independent replay **7506** are terminal/pass.
  All 61 synthetic states, four complete tables and 432 numerical outputs match exactly. Both
  one-unit numerical mutations are refused on each attempt, with network and old producer paths
  blocked. Each child imports only its 70 archived package modules. The exact audit/replay and
  bounded acceptance records are retained in that archive. Follow its
  [execution record](reports/engineering-archive/dense-v3-vector-reconstruction-v1/RUNNING.md).
- This verifies synthetic raw-feature reconstruction under the pinned runtime on this host,
  not a different physical machine, absent model payloads, primary outcomes or the paper.
- The separately [prepared original-outcome fixture](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
  has completed same-location raw readback: 12 × 4,096 synthetic validation records, 840 synthetic
  BEIR cells and all ten original table schemas. Actual validation row identities are revalidated.
  All scores/timing/checkpoint admissions remain synthetic, with minimal upstream placeholders.
  It is not the missing complete local-role outcome reader or its cold integration acceptance.

The accepted [source/transport parent](reports/engineering-archive/dense-v3-portable-reconstruction-v1/README.md)
remains unchanged:

- A real 238-file source-only payload loads the same frozen contracts and twelve expected run
  identities after relocation. A fresh process uses only its 69 archived package imports.
- The authoring entry point requires existing complete checkpoint-backed inference before selecting
  source/result files. It preserves original provenance bytes and uses explicit local roles plus an
  external manifest hash. It does not copy entire experiment trees, model payloads or raw data text.
- A separate tiny 251-file positive fixture explicitly simulates upstream admission. Actual missing
  primary runs are rejected before output. Independent-process transport replay and corruption,
  rehashed-manifest, omission and undeclared-directory refusal checks pass.
- **47 focused tests** and **2,381 full isolated tests** passed at that accepted parent milestone.
  This verifies source closure and local transport, not numerical reconstruction from all raw
  vectors, real primary admission, remote publication or a completed paper.

The accepted [functional inference parent](reports/engineering-archive/dense-v3-dimension-inference-v1/README.md)
already connects the following; it is preserved unchanged:

- The complete v3 consumer connects source-admitted original outcomes/geometry and reconstructed
  dimension features. Both actual CLI modes reject missing primary runs before later inputs/output.
- The original nine-contrast simultaneous task family, all four rates and all three required native
  directions are retained. A zero standard error refuses the family; an unmet joint rule is not equivalence.
- Four genuinely named functional predictors are added without replacing any original weight-space
  feature. Exact fits retain every dose/stage, undefined folds and unsupported predictors.
- Independent Torch FP64 resampling checks nine intervals, 27 rotation contrasts and 68 figure
  points. SymPy full rational normal equations check 240 predictions, 16 fold errors, four pooled
  decisions and eight residual correlations. These inputs are explicitly synthetic, not model results.
- A fresh process repeats the numerical reconstruction after relocating inputs and rejects six
  rehashed table/text mutations. Full downstream admission is simulated in integration tests;
  neither rehearsal proves complete primary checkpoint/vector reconstruction or publication readiness.
- 64 focused tests pass. The first independent oracle failed on symbolic Boolean rank counting;
  its exact source and input are preserved, five oracle controls were added, and the complete retry
  and fresh replay pass. Production inference, scientific thresholds and paper gates did not change.
- The generated standalone synthetic PDF compiles without Type 3 fonts and was visually checked.
  No artificial values were installed in manuscript includes. The full isolated suite passes
  **2,334 cases** at that parent milestone, without failures, errors or skips.

The [separate real numerical/input parent](reports/engineering-archive/dense-v3-dimension-interventions-v1/README.md)
already verifies 688,128 pretrained rank checks, all masks/rotations, and 2,016 original texts against
42 pinned BEIR files. The [export/feature parent](reports/engineering-archive/dense-v3-dimension-chain-v1/README.md)
also verifies all 134 actual loaded base tensors and complete synthetic 61-state raw/feature wiring.
Those checks were not repeated as missing work. No primary vectors or optimizer findings were added.
Final source-authenticated publication and portable numerical reconstruction remain incomplete.
Current full gather still records host-bound source paths. Transport preserves those bytes without
opening the old paths. The raw-vector reader now independently reconstructs features locally, but complete original
outcome/geometry/inference reconstruction and primary publication are not implemented/accepted yet.

## Accepted preparation and what remains

| Surface | Current evidence | Remaining requirement |
| --- | --- | --- |
| Shared data | Revised 500k / 4,096 views; only 6 / 52 groups changed; complete independent reconstruction | Formal v3 deployment; no semantic-decontamination claim |
| Training core and identity | Actual short full-model GPU/maximum-context/continuation checks; sealed payload readers | One assembled released checkout and all full-horizon primary runs |
| Validation / outcomes | Real bounded scorer checks, complete-grid admission, independent statistical replay | Actual full validation and 840 primary BEIR units |
| Geometry | Approximate and exact consumers; all 88 hidden matrices on nine diagnostic checkpoints independently checked | All primary rate/stage states |
| Retrieval bridges | Original nine-feature and separate five-feature exact sensitivity; exact independent fits and replay | Real complete primary inputs, no cherry-picked support |
| Functional dimensions | Real kernel/input/loading checks; complete feature wiring; independently checked task-family inference and exact four-feature bridge; portable source/transport foundation | Actual primary vectors, final source-authenticated publication and portable numerical reconstruction |
| Crossed continuation | Frozen scientific design and old prepared implementation | Routed-control v3 integration and actual continuation experiment |
| Manuscript / repository | Weight-space-centered draft; all engineering evidence preserved | Real generated findings, clean-clone evidence, strict release and authorized publication |

The distinct identity candidate is **not whole-repository green**: 963 tests include eleven
unchanged, unwaived source-contract failures. The isolated suite does not certify that source tree.

## Scientific hold and runtime authority

The old campaign used duplicated accumulation normalization, producing approximately quarter-scale
raw gradients. Its completed checkpoints do not become valid primary evidence by renaming or
re-evaluating them. The correction and revised identities live in separate preparation trees.

Live source: /root/embedding-optimizer-study, main f231a6430712388778f32ad1736a4cb6de3bec3e.
Development: /root/embedding-optimizer-story-refactor, narrative/weight-space-spine, same base.
Numerical parent: /tmp/dense-correction-source.0YHywN.
Identity candidate: /tmp/dense-identity-source.8rUgLF. No source deployment occurred.

The exact old BEIR controller chain remains stopped in place, with its original lease, four-step
completed prefix and main contract 4152531e.... The factorial remains waiting at zero steps under
6605090d.... Old zero-step main migrations are inapplicable. [AGENTS.md](AGENTS.md) retains exact
handles, ledger identity and all operational prohibitions. Never touch gpu.py or its processes.

WIP-commit exception and post-acceptance twelve-run rerun questions remain unanswered; automatic
goal continuation is not approval. Draft-parent bindings also prevent a status-only release.
Do not deploy, launch formal training, transition controllers, commit/merge/push or waive release gates.
GitHub writes returned 403; no retry or alternate credentials. This local WIP is not in a clean
remote clone. HF withdrawal is safety-rejected; no deletion or history erasure occurred.

The [artifact snapshot](CURRENT_PROGRESS.json) counts legacy files, not live jobs or accepted
scientific results. It must be refreshed from the live experiment directory. Automatic 60-checkpoint
coverage is distinct from the separate independent ten-run HF audit (1,010 files / 82,373,100,558 bytes).

## Exact next sequence

1. Implement the original-outcome reader on local roles, using the prepared raw fixture without
   weakening checkpoint identity requirements to fit its minimal simulated placeholders. Both
   full-width vector audits are complete; do not repeat them as missing. Preserve all source
   snapshots, failed attempts and prior source/transport, numerical and inference checks. The
   [dimension inference draft](configs/dense_primary_v3_dimension_inference_protocol.json) is
   preparation only; it does not release its bound primary/export/feature parents.
2. Complete numerical reconstruction on the externally anchored local roles: original outcome/
   geometry primitives, original and functional bridges, fixed inference families
   and rendered evidence. Do not use producer-location fallback. Preserve every already verified feature,
   task-family rule and unsupported result. Do not repeat completed kernel/input/numerical checks
   or treat relocated synthetic summaries as full primary reconstruction.
3. Integrate the routed factorial AdamW control separately; the primary-only policy rejects it.
   Complete the reviewed assembled-source/runtime/release-parent handoff without refreshing old
   hashes into false compatibility. Existing completed diagnostics need no repeated baseline runs.
4. Obtain outstanding authority/access, execute all twelve primary runs in new namespaces and
   verify durable checkpoints. Then complete actual validation/BEIR, analyses and crossed continuations.
5. Write the paper from those accepted findings, render figures/tables and verify all manuscript,
   clean-clone, reconstruction and distribution gates before authorized release.

## Evidence navigation

- [Complete raw-vector reconstruction checks](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
- [Next original-outcome reader: complete raw fixture preparation](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
- [Current source closure and authoring transport](reports/engineering-archive/dense-v3-portable-reconstruction-v1/README.md)
- [Current functional inference preparation](reports/engineering-archive/dense-v3-dimension-inference-v1/README.md) ·
  [Complete export/feature preparation](reports/engineering-archive/dense-v3-dimension-chain-v1/README.md) ·
  [Real dimension interventions and probe inputs](reports/engineering-archive/dense-v3-dimension-interventions-v1/README.md)
- [Exact-feature bridge sensitivity](reports/engineering-archive/dense-v3-exact-bridge-v1/README.md) ·
  [Original-feature bridge](reports/engineering-archive/dense-v3-bridge-v1/README.md)
- [Complete exact geometry](reports/engineering-archive/dense-v3-exact-geometry-v1/README.md) ·
  [Approximation/seed controls](reports/engineering-archive/dense-geometry-robustness-v1/README.md) ·
  [Approximate geometry integration](reports/engineering-archive/dense-v3-geometry-chain-v1/README.md)
- [Outcomes](reports/engineering-archive/dense-v3-outcomes-v1/README.md) ·
  [Validation](reports/engineering-archive/dense-v3-validation-v1/README.md) ·
  [V3 core](reports/engineering-archive/dense-primary-v3-chain-v1/README.md)
- [Revised natural-data entrypoint](reports/engineering-archive/dense-revised-natural-v1/README.md) ·
  [Reconstructed revised data](reports/engineering-archive/dense-data-candidate-v1/README.md) ·
  [Original query-partition audit](reports/engineering-archive/dense-data-partition-v1/README.md)
- [Training identity and candidate failures](reports/engineering-archive/dense-full-identity-v1/README.md) ·
  [Whole-run/task/grid readers](reports/engineering-archive/dense-primary-completion-v1/README.md)
- [Dimension scientific protocol](configs/dense_dimension_utilization_protocol.json) ·
  [Primary v3 draft](configs/dense_primary_v3_protocol.json) · [Paper gates](paper/README.md)
- [Artifact restoration](docs/checkpoint-restoration.md) · [Factorial design](docs/state-operator-factorial.md)

The exact previous [status](reports/engineering-archive/dense-v3-dimension-interventions-v1/before/PROJECT_STATUS.md),
[README](reports/engineering-archive/dense-v3-dimension-interventions-v1/before/README.md) and
[agent instructions](reports/engineering-archive/dense-v3-dimension-interventions-v1/before/AGENTS.md)
retain the full chronology, failure details and older command plans. They are history, not the
default launch instructions. Current claims and authority are defined above.
