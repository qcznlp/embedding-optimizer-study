# Project status and handoff

Last updated: **2026-09-06 16:23 UTC**; the machine-readable snapshot is refreshed at this handoff.

## Where the project stands

**The revised-data/untrained-base preflight passes: three actual four-GPU entrypoint calls and nine complete diagnostic checkpoints. The explicit v3 data amendment is prepared; formal v3 consumer/deployment integration remains incomplete. The old matrix stays on scientific hold.**
All 12 primary runs finished and all 60 scheduled checkpoints are preserved. They were executed
with the unchanged training implementation, whose duplicated accumulation normalization produces
approximately quarter-scaled raw gradients. The tested correction is in a new detached preparation
checkout, not deployed into either existing source tree.

The [preceding real-data audit](reports/engineering-archive/dense-natural-readiness-v1/README.md)
checks all 4,500,000 training text fields and every actual materialized ID. All 500,000 row
identities match the original ledger; five rows (0.001%) contain empty text: one positive and
four negatives. The frozen 4,096-query validation set retains zero source-qualified query-ID overlap and
correct row linkage, but has one empty negative. Neither dataset has an exact nonempty
positive–negative or repeated-negative text collision within its declared groups. This does
not rule out semantic false negatives or annotation noise.

All six affected cells trace to three exact-empty source documents; the relevant upstream
parquet files match immutable HF digests. No tokenizer/kernel/optimizer cause is inferred.
The small frequency does not establish an effect on retrieval or explain an optimizer advantage.
The fixed 288-row diagnostic prefix failed its nonempty-text guard before any GPU lease,
torchrun or model load. Its failure is preserved; no sample was skipped to make it pass.
No GPU worker or training run occurred in that failed-prefix attempt.

The [original-data query-partition audit](reports/engineering-archive/dense-data-partition-v1/README.md)
authenticates all 19,385 evaluation query records across fourteen pinned BEIR tasks and compares
actual strings. An independent non-hash replay reproduces all 84 training–validation matches and
the one training–BEIR match; **all 85 pairs are already identical raw strings**. Validation has
no internal normalized-query duplicates and no BEIR query overlap. ID disjointness alone did not
establish text disjointness. These exact-string checks do not rule out semantic near-duplicates.

The [expanded proposal](reports/engineering-archive/dense-data-partition-v1/revision-proposal.json)
replaces six whole training groups and 52 validation groups, preserving 499,994 and 4,044 rows
unchanged and retaining all source quotas and sample counts. It supersedes the insufficient
six-empty-group-only proposal without rewriting its evidence. All actual values in the 1,024-row
training probe, 50k branch subset and 224-query historical validation subset still match their
parents; none contains a newly targeted position. That is not a new parent-identity admission.

The [new data preparation and independent reconstruction](reports/engineering-archive/dense-data-candidate-v1/README.md)
implements that expanded proposal. All 54 used upstream files match immutable HF digests.
The original population/seeded priority, first eligible score record and seven-of-ten negative
draw remain fixed. Every traversed candidate has a recorded decision. All original query IDs/texts,
held-out BEIR query texts and accepted replacements are protected. Exactly six training groups and
52 validation groups change; every other field/row remains unchanged and quotas/totals are retained.
Full text, actual ID linkage and query-text isolation pass for both complete prepared datasets.

An independent original-scanner/full-document-loader path reconstructs all decisions and replacements,
then uses a different row-generator writer to reproduce every complete dataset value and canonical
row-ledger digest. Its complete normalized-string joins also pass, without relying on hash matching.
This is logical value equality, not identical Arrow container bytes or a semantic decontamination claim.
Prepared inputs remain in `/tmp/dense-partition-candidate.kHXoGW`, with independent reconstruction
in `/tmp/dense-partition-reconstruction.HsE86I`; preserve both. Neither replaces the original data.

The first two-row-group-cache CPU attempt was interrupted through its own session for repeated IO;
its source and 50,628 partial decisions remain archived. The bounded 64-group-cache retry reproduces
that prefix byte-for-byte, completes both datasets and passes independent reconstruction. No rule
was relaxed. No GPU worker or formal training ran in that data-materialization milestone.

The [new revised-input/natural-entrypoint milestone](reports/engineering-archive/dense-revised-natural-v1/README.md)
observes the actual revised 500k/4,096 inputs, derives twelve new full-horizon identities and checks
all three subsets against revised parents. The explicit v3 data amendment preserves the old
protocols and is not execution-authorized. Its actual loader accepts the baseline and rejects all
21 changed-file cases, including old roots, missing data, altered diagnostic rules and false release
authority. No changed configuration is trained by that admission audit.

Three actual unpatched prepared `run_training` calls now start from the immutable untrained DenseOn
base. The shared 288-group coverage fixture includes all seven sources and all six training
replacements, with fixed seed/quotas and all original fields retained. AdamW, Muon and NorMuon each
complete three steps on four GPUs, producing nine complete content-sealed checkpoints and matching
final exports. First-step loss/gradient-norm logs are finite and identical. Actual maximum input
length is 7,679, not 8,192; the earlier separate maximum-context checks remain distinct evidence.
A fresh CPU verifier invocation rechecks the saved artifacts and complete-run state successfully.

Preserve `/tmp/dense-natural-readiness.a4xZzR`: 248 bound files / 15.28 GB plus the final receipt.
These diagnostic checkpoints are not new primary results or remotely published backups. No full
epoch, all-rate natural-data campaign, natural-batch gradient oracle or natural-data continuation
was tested. **Revised-input preparation, data amendment and bounded natural preflight are complete.**
Next integrate complete v3 primary/validation and downstream consumers plus the separately routed
factorial control. Old v2 consumers still bind old data; do not merely refresh expected hashes.
All owned calls exited. No formal run, old-controller transition or source publication occurred.

The new [integrated candidate](reports/engineering-archive/dense-correction-candidate-v1/README.md)
passes official optimizer comparisons, full-model short/8192-token gradient oracles, controlled
checkpoint replay, and the actual unpatched `run_training` entrypoint including continuation.
The [preceding predeployment checks](reports/engineering-archive/dense-predeployment-checks-v1/README.md)
passed controlled independent-process replay and identified the missing complete-run identity gate.
That gap is now closed in a **distinct [identity-integration candidate](reports/engineering-archive/dense-full-identity-v1/README.md)**:
the actual entrypoint derives and enforces complete input/configuration/runtime/source identity,
and its actual Trainer saves and verifies content-sealed checkpoints. Six real four-GPU baseline/
continuation calls and 45 actual admission decisions pass. All are engineering evidence, not
retrieval findings or acceptance of the old matrix. The new source does not inherit a bitwise
replay claim merely from the predecessor's different source identity.

The [new primary-contract milestone](reports/engineering-archive/dense-primary-contract-v1/README.md)
authenticates the original 500k data and untrained base, derives all twelve exact recipe identities,
and prepares a shared checkpoint boundary for training, backup and evaluation. Real read-only
rehearsal accepts twelve authenticated diagnostic checkpoints only as diagnostics, rejects all
sixty old primary checkpoints and three corrupted copies, and passes all twelve CLI inspection
plans. The draft is deliberately not execution-authorized; no new formal run/upload/evaluation
or accepted scientific result has been produced.

The [current completion-reader milestone](reports/engineering-archive/dense-primary-completion-v1/README.md)
adds terminal run/timing/final-export checks, deep model/optimizer/scheduler validation, complete
fourteen-task admission and twelve-run-before-840-cell collection. Three actual complete diagnostic
baselines pass; three continuation-only directories are correctly refused as whole runs. One
real SciFact diagnostic evaluator finished on a separately leased GPU, and its unchanged raw output
passes the final CPU reader. This is not a primary evaluation or an optimizer-quality result.
The actual full-grid CLI still rejects the missing new primary runs. No formal run was started.

| Surface | Verified now | What this does not establish |
| --- | --- | --- |
| Primary execution/durability | 12/12 runs; 60/60 resumable and remotely covered checkpoints | Scientific validity |
| Numerical parent optimizer conformance | NorMuon 96/96 exact updates on CPU and separately on GPU; Muon 96/96 unchanged per device | Model-quality conclusions or universal numerical equivalence |
| Numerical parent GPU gradients | 9/9 short-input and 9/9 maximum-length raw/clipped checks, all 134 parameter tensors | Every natural-text batch or full training horizon |
| Numerical parent controlled replay | 24/24 exact same-group and 24/24 exact new-process-group comparisons | A replay claim for changed source, default DDP bitwise equivalence or another physical host |
| New identity candidate: real entrypoint | 3 baseline + 3 relocated-data/model continuation calls; 12 complete content-sealed checkpoints; final exports match scheduled weights | A full primary run, natural-data readiness or frozen CLI/protocol acceptance |
| New actual admission | 6 original/relocation positives accepted; 36 changed identities and 3 corrupted real optimizer payloads rejected before setup/deserialization | Digital-signature authentication or proof of historical configuration drift |
| New candidate regression suite | 963 cases: 952 pass, **11 unchanged source-contract failures**, no exclusions; 79 focused cases pass | Whole-repository acceptance; those gates remain unwaived |
| Primary input audit | Actual 500k row linkage; 16 data files; 11 untrained-base files match immutable HF digests; 12 derived recipe identities | Twelve independent preflights or natural-data GPU execution |
| Prepared shared primary consumers | 12 real diagnostic payloads accepted only as diagnostics; 60 old checkpoints and 3 corrupt copies rejected; 12 actual CLI inspection plans; 51 focused tests pass | Released training, real new upload/scoring, complete-run or downstream acceptance |
| Whole-run reader: actual files | 3 complete diagnostic runs / 9 full-model checkpoints accepted; 3 continuation-only directories rejected | A complete new 500k primary run or restored missing prefix |
| Task/grid reader | 1 actual SciFact full-corpus diagnostic and unchanged-file CPU replay; 63 focused checks; exact 12×5×14 grid adapter | All 14 actual tasks, the 840-cell primary grid, validation selection or scientific findings |
| Natural-data entry preflight | Actual fixed prefix rejected before GPU launch at index 230 / empty positive | A successful natural-data training run |
| Full text and actual ID audit | 500k training / 4,096 validation rows linked exactly; five / one ineligible text groups; three upstream empty documents authenticated | Clean revised data, semantic label quality or optimizer effects |
| Full query-partition audit | All 14 pinned tasks; 84 train–validation pairs / 51 validation rows and 1 train–BEIR pair independently reproduced as raw equality | Semantic near-duplicate absence, revised data readiness or optimizer effects |
| Prepared revised data | 500k / 4,096 rows; exactly 6 / 52 changes; full content, actual ID linkage, quotas and defined query isolation pass | Formal protocol/consumer admission, natural-data GPU execution or semantic label correctness |
| Independent reconstruction | Original scanner/loader and independent priority/string joins; all decisions, 58 replacement groups, complete values and canonical ledgers agree with a second writer | Identical Arrow containers, independent laboratory replication or optimizer effects |
| Actual v3 input preparation/admission | 500k/4,096 observed rows; 12 new identities; 3 subsets match revised parents; baseline accepted and 21 changed files rejected | Formal v3 consumer release or deployment |
| Revised natural-data GPU entrypoint | 3 actual calls from untrained base; 7 sources / all 6 replacements; 9 complete checkpoints; fresh CPU readback passes | Full epoch/all rates, natural-gradient oracle, natural-data resume, cross-host replay or optimizer findings |
| Isolated paper/audit suite | 1,662 cases, zero failures/errors/skips; 30 new amendment/coverage/owned-launch tests pass | Correctness of the different numerical candidate source tree or formal scientific acceptance |
| BEIR | Exact original dispatch chain paused; existing results preserved | Complete 840-unit primary grid or accepted optimizer conclusion |
| Weight geometry | Last independent archive: ten runs, 50 stages, 4,400 hidden-matrix records | Functional dimension utility or retrieval explanation |
| Dimensions / crossed continuation | Implementation and interpretation checks prepared locally | Real accepted primary mechanism results |
| Manuscript | Weight-space-centered narrative prepared; actual result includes pending | A complete, releasable NAACL paper |

The independent ten-run HF audit still verifies 1,010 files / 82,373,100,558 bytes at original
immutable upload commits. Automatic 60-checkpoint coverage does not silently extend that separate
audit. The [current snapshot](CURRENT_PROGRESS.json) is artifact-only, not a heartbeat or a
scientific-validity decision.

Refresh that snapshot with the experiment checkout as the working directory and an explicit
output path. Running the same relative-path command in the isolated paper checkout sees no
training artifacts; that is not evidence that completed runs disappeared.

## What the latest verification actually resolved

The preceding [origin probe](reports/engineering-archive/dense-gradient-origin-v1/README.md)
established that identical local gradients were reduced through different DDP bucket layouts
after restart. The [single-update probe](reports/engineering-archive/dense-gpu-replay-localization-v1/README.md)
then exactly reproduced the resulting optimizer update differences. These are floating-point
reproducibility observations, not newly discovered optimizer/serialization defects.

The new diagnostic deliberately fixes the collective vector layout and owned attention backward
order while retaining the already tested single-normalization candidate. It verifies the same
global-average objective independently. Three optimizers and two checkpoint restart points now
reproduce all entry/intermediate/final state, gradients, data order and per-rank RNG exactly.
A new torchrun group repeats the continuations from sealed files without retraining a baseline.
Maximum-length execution passes with the original tolerance and without OOM.
The numerical parent's own independent-group replay finished at 09:55:29 UTC. Its launcher
exited zero and all 24 full
fingerprint comparisons pass, including subsequent gradients/states, consumed rows and rank RNG.

This control is not automatically a formal-training default or a performance optimization:
it holds communication until all gradients are ready and allocates a full gradient vector.
The earlier failed default-path receipts and the separate production normalization defect remain
unchanged. Do not repeatedly reopen the now-resolved short-fixture replay origin question.

The new prepared policy assigns normalization once, explicitly configures all 22 owned attention
backward flags, selects official additive-epsilon NorMuon without changing Muon, and records a
numerical checkpoint contract that rejects legacy continuations. Its full entrypoint also passes
the new rank-before-shared-write synchronization guard. All 12 proposed recipes retain their
scientific hyperparameters, changing only the declared numerical policy and new namespaces.

That numerical-only parent contract does not bind the whole run. Its CPU probe authenticates nine actual
saved metadata files, then calls the real entrypoint with changed configurations and stops before
setup, seeding, output writes or model/data loading. For each of the three algorithms, ten changes
are not rejected by the preload gate; changing optimizer LR or accumulation correctly rejects.
Both diagnostic attempts retain exit 1 for this missing readiness condition. No changed run was
trained and no historical checkpoint/configuration was modified.

The tested comparator has now become an actual entrypoint/Trainer contract in a separate source
tree. It derives observed data and initial-model file identities, resolved recipe/runtime settings,
local source hashes and pinned package/numerical-stack identity before setup. Actual arguments and
all-rank identity must agree. Every checkpoint is sealed only after all ranks write RNG state;
both model and optimizer restore verify the complete identity and all payload digests first.
Old or incompatible checkpoints are rejected, not retrofitted. Digests do not replace the trusted
remote backup audit; they are not signatures. Remote base snapshots must be pre-populated at the
declared immutable revision before this read-only admission step.

The real four-GPU integration passed at 10:47:01 UTC. Original versus byte-identical relocated
data/model/output paths retain the same identity for all three optimizers. The subsequent CPU
probe authenticates 258 actual producer files and exercises 45 admission decisions: six accepts,
36 recipe/data/runtime rejections and three one-byte optimizer corruption rejections, all before
setup or deserialization. No changed-recipe run is trained. This closes the integration task,
not the separate primary protocol/runtime transition. All diagnostic output remains preserved.

The independent [old NorMuon counterexample](reports/engineering-archive/normuon-full-reference-v1/README.md)
remains: its clamped normalization differs from official additive epsilon on 44/96 small-matrix
cases. The candidate passes 96/96 exact weight/state checks per device. Neither repairs old
checkpoints nor establishes a retrieval advantage. The initial candidate's six metadata
compatibility failures were fixed; its remaining eleven frozen-source refusals were not relaxed.

The new primary proposal binds those prepared training bytes and the consumer/runtime closure.
Its actual-input receipt observes the shared 500k dataset once and compares the initial model at
revision `0edbd556...` to HF's immutable file digests. Training inspection, backup and evaluation
use the same expected run identity and complete checkpoint seal. Backup's mocked network tests
retain the original upload commit; evaluation caches are protocol/content-addressed. Prepared
training/evaluation share a cooperative GPU lease. No real upload or evaluator was run here.
Execution requires a reviewed release in one assembled, committed checkout; a draft or a mixture
of these two source trees is rejected. Whole-run/task/grid readers are now implemented, while
their actual primary inputs, natural-data execution readiness and downstream scientific integration
remain pending. The original scientific recipes, selection rules and failed gates remain intact.

The new completion reader explicitly separates a successfully ended continuation from a complete
run with all earlier checkpoint/timing evidence. Its real CPU checks authenticate payloads before
weights-only deserialization and verify finite model/moment tensors, exact scheduler/group semantics
and matching final inference bytes. A preserved first-reader manifest-field compatibility failure
was corrected without changing the primary data requirement or numerical training implementation.

The single diagnostic SciFact worker exited zero; the first reader rejected six undefined *unused*
auxiliary nAUC fields. The revised reader records those fields explicitly, keeps raw JSON unchanged,
and still rejects invalid primary nDCG/main scores or metadata. An independent check confirms that
the unchanged original scientific nDCG reader accepts the same file. No metric is imputed, no task
is dropped, and no primary criterion changes. The original failure and source-level MTEB control
are archived. Only CPU parsing was repeated; no second GPU evaluator ran.

## Scientific goal and paper logic

Determine how AdamW, Muon and NorMuon reshape reached weight states; test whether that changes
the functional use of embedding dimensions; then test whether it explains retrieval outcomes.
The intended evidence chain is **weight trajectory → dimension utility → retrieval**, followed
by a crossed continuation for a fixed source-state/operator pair. It is a hypothesis to test,
not an established conclusion and not a decomposition of the primary gain.

Only a scientifically accepted primary matrix and its audited follow-ups can supply paper
findings. Do not infer retrieval quality from a flatter update spectrum, assert that AdamW has
an inherently better next step, or recycle invalidated exploratory scores. One primary seed and
one base model limit generalization; four learning rates are not four independent seeds.

The owner excludes implementation-error, packing and padding narratives from the entire paper,
including its appendix. They remain repository engineering provenance only. No blog and no new
LateOn training/evaluation are in scope. The narrower
[factorial claim interpretation](reports/paper-review/factorial-claims-v2/README.md) is retained:
positive interaction is not sufficient for beneficial co-adaptation, and an inconclusive effect
does not establish another effect's dominance.

## Runtime and authority boundary

Live numerical source: `/root/embedding-optimizer-study`, main
`f231a6430712388778f32ad1736a4cb6de3bec3e`.
Isolated development: `/root/embedding-optimizer-story-refactor`,
branch `narrative/weight-space-spine`. Use this checkout's absolute `src` and repository paths
in `PYTHONPATH`; the installed package otherwise resolves to live main.
Preserved numerical parent: `/tmp/dense-correction-source.0YHywN`, detached at the same base.
Current full-identity candidate: `/tmp/dense-identity-source.8rUgLF`, also detached at that base.
Its 14-file source manifest and complete copies are in the new archive. Neither candidate is
deployed. Keep suites separate: current candidate 963 cases / 11 failures; isolated checkout
1,662 cases / zero failures. The older candidate's 923-case receipt remains valid historical evidence.

All diagnostic launchers have exited. BEIR's exact retained controller/wrapper/scheduler remain
stopped in place, with their original creation times, lease and four-step completed prefix.
The main ledger remains at active step BEIR with contract `4152531e...`; the factorial ledger
remains waiting for main at zero steps, contract `6605090d...`. The
[handoff receipt](reports/engineering-archive/dense-gpu-correctness-v1/handoff/result.json)
defines the exact handles. Do not kill/restart them casually or resume evaluation into an
owned diagnostic pool. No production numerical source, checkpoint or ledger was modified.

The old [seven-file main repair](reports/engineering-archive/main-resume-v1/README.md) targets
`4380a363...`, while the prepared four-layer successor projects to `41fdf48c...` locally and
`7aafd7af...` on the host. These are **undeployed**. The old main migration requires an inactive
zero-step ledger; those preconditions no longer hold. A reviewed transition must preserve the
current completed prefix, logs, backups and leases. These post-processing repairs do not fix
gradient normalization.

Formal correction/retraining, controller transition and WIP source publication remain separate
reviewed actions. The repository's pre-commit rule still requires no pending manuscript results;
do not self-relax it or merge/push the dirty branch unapproved. Issue #41 is the current public
progress channel, but its integration returned 403 on the last comment/update attempts. This
milestone is **local only**, not posted there. No alternate credentials/identity were used.
New source/evidence files are not yet available from a clean public clone.

HF withdrawal remains safety-rejected: 9,629 proposed files / 366,252,912,201 logical bytes.
No deletion occurred. Never retry, split or bypass the rejected operation. Any approved historical
erasure additionally needs refreshed shared-object/current-backup dependency checks and exact scope.

Never inspect, read, edit, signal, stop, replace or otherwise touch `gpu.py` or its processes.
Do not use broad process inspection, launch duplicate controllers, overwrite checkpoints,
relax frozen acceptance gates, print credentials or promote diagnostic numbers into the paper.

## Next bounded steps

1. Review the [whole-run/task/grid reader and actual evidence](reports/engineering-archive/dense-primary-completion-v1/README.md),
   the [prepared primary input contract](reports/engineering-archive/dense-primary-contract-v1/README.md)
   and preceding [identity persistence evidence](reports/engineering-archive/dense-full-identity-v1/README.md).
   These bounded implementation tasks are complete in preparation, not deployment. Do not repeat
   them as missing work or infer primary execution from diagnostic/rehearsal artifacts.
2. The expanded six-training/52-validation-group revision is now materialized and independently
   reproduced, its explicit data amendment is prepared, and the seven-source actual natural-data
   entrypoint passes. Do not repeat those as missing work. Integrate the new v3 input/identity/grid
   into complete primary training/backup/evaluation, validation selection, outcome, weight-space/
   dimension and publication consumers; the old v2 consumers still bind old data. Preserve every
   failed prefix/incomplete attempt and the nine new diagnostic checkpoints. Prepare an
   evidence-preserving runtime handoff. Follow the preserved
   [transition proposal](reports/engineering-archive/dense-correction-candidate-v1/replication-transition-proposal.md),
   with the identity-implementation task now complete.
   Do not merely replace eleven hashes: the primary-only policy currently rejects the routed
   factorial AdamW control, which needs a separately reviewed integration. Preserve all old locks,
   ledger prefixes and results; keep scientific estimands and selection rules fixed except for
   the explicitly declared, result-independent data amendment.
3. Resolve deployment/WIP-publication authority and GitHub integration write access, then run the accepted primary revision.
   Re-audit checkpoint/optimizer/RNG persistence, source identity and remote durability.
4. After primary acceptance, complete the 840 full-corpus BEIR units and validation selection,
   weight-space/dimension analyses and the frozen crossed continuation. Historical exports cannot
   replace accepted primary inputs.
5. Render actual findings only, verify the portable evidence closure and strict manuscript gates,
   and release the paper/repository. Synthetic paper builds and passing unit tests are not results.

## Evidence and continuation map

- [Latest actual v3 input admission and untrained-base natural-data preflight](reports/engineering-archive/dense-revised-natural-v1/README.md);
  [preparation-only data amendment](configs/dense_primary_v3_data_amendment.json).
- [Prepared 6/52-group revision and independent complete-data reconstruction](reports/engineering-archive/dense-data-candidate-v1/README.md).
- [Original full query-partition audit and independent raw-string replay](reports/engineering-archive/dense-data-partition-v1/README.md);
  its preserved prospective proposal is now implemented in the new, separate data candidate.
- [Preceding real-data failure, full-text/ID checks and source origins](reports/engineering-archive/dense-natural-readiness-v1/README.md);
  its original six-empty-group-only proposal is preserved but insufficient.
- [Current whole-run/task/grid readers and real-file evidence](reports/engineering-archive/dense-primary-completion-v1/README.md);
  [new non-execution-authorized completion lock](configs/dense_primary_v2_completion_protocol.json).
- [Current primary inputs, draft consumers and real-file rehearsal](reports/engineering-archive/dense-primary-contract-v1/README.md);
  [not-yet-released proposal](configs/dense_primary_v2_protocol.json).
- [Current full-identity integration and evidence](reports/engineering-archive/dense-full-identity-v1/README.md);
  [preceding predeployment checks and counterexamples](reports/engineering-archive/dense-predeployment-checks-v1/README.md).
- [GPU control and exact commands](reports/engineering-archive/dense-canonical-replay-v1/README.md);
  [earlier GPU failures/controls](reports/engineering-archive/dense-gpu-correctness-v1/README.md).
- [Integrity audits](reports/experiment-integrity/README.md);
  [checkpoint restoration](docs/checkpoint-restoration.md).
- Frozen [training](configs/dense_no_packing_execution_protocol.json),
  [evaluation](configs/dense_no_packing_evaluation_protocol.json),
  [analysis](configs/dense_no_packing_analysis_protocol.json) and
  [outcomes](configs/dense_no_packing_outcome_protocol.json).
- [Dimension utilization](docs/dimension-utilization.md);
  [state/operator continuation](docs/state-operator-factorial.md);
  [paper plan](docs/naacl-dense-paper-plan.md) and [release gates](paper/README.md).
- [Public progress issue #41](https://github.com/qcznlp/embedding-optimizer-study/issues/41);
  [W&B](https://wandb.ai/stevezenguom/embedding-optimizer-study);
  [checkpoints](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints);
  [analysis archive](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts).

The exact previous status page is preserved in
[the pre-update archive](reports/engineering-archive/dense-full-identity-v1/before/PROJECT_STATUS.md).
Its superseded milestone counts and launch instructions are history, not current authority.
