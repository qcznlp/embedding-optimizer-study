# Project status and handoff

Last updated: **2026-09-06 10:58 UTC**; the machine-readable snapshot is refreshed at this handoff.

## Where the project stands

**The full run/checkpoint identity is now integrated and tested; the formal matrix is still on scientific hold.**
All 12 primary runs finished and all 60 scheduled checkpoints are preserved. They were executed
with the unchanged training implementation, whose duplicated accumulation normalization produces
approximately quarter-scaled raw gradients. The tested correction is in a new detached preparation
checkout, not deployed into either existing source tree.

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

| Surface | Verified now | What this does not establish |
| --- | --- | --- |
| Primary execution/durability | 12/12 runs; 60/60 resumable and remotely covered checkpoints | Scientific validity |
| Numerical parent optimizer conformance | NorMuon 96/96 exact updates on CPU and separately on GPU; Muon 96/96 unchanged per device | Model-quality conclusions or universal numerical equivalence |
| Numerical parent GPU gradients | 9/9 short-input and 9/9 maximum-length raw/clipped checks, all 134 parameter tensors | Every natural-text batch or full training horizon |
| Numerical parent controlled replay | 24/24 exact same-group and 24/24 exact new-process-group comparisons | A replay claim for changed source, default DDP bitwise equivalence or another physical host |
| New identity candidate: real entrypoint | 3 baseline + 3 relocated-data/model continuation calls; 12 complete content-sealed checkpoints; final exports match scheduled weights | A full primary run, natural-data readiness or frozen CLI/protocol acceptance |
| New actual admission | 6 original/relocation positives accepted; 36 changed identities and 3 corrupted real optimizer payloads rejected before setup/deserialization | Digital-signature authentication or proof of historical configuration drift |
| New candidate regression suite | 963 cases: 952 pass, **11 unchanged source-contract failures**, no exclusions; 79 focused cases pass | Whole-repository acceptance; those gates remain unwaived |
| Isolated paper/audit suite | 1,424 cases, zero failures/errors/skips | Correctness of the different candidate source tree |
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
1,424 cases / zero failures. The older candidate's 923-case receipt remains valid historical evidence.

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

1. Review the [full-identity integration and real acceptance evidence](reports/engineering-archive/dense-full-identity-v1/README.md).
   This task is complete in the prepared source. Do not redo the missing-gate diagnosis or claim it
   remains a standalone proposal. Retain exact scope: real save/load/admission is tested, whereas
   a new independent-process bitwise replay of these changed bytes was not performed.
2. Prepare/review new source-bound primary, reload, durability and downstream protocols, natural-data
   execution readiness, and an evidence-preserving runtime handoff. Follow the preserved
   [transition proposal](reports/engineering-archive/dense-correction-candidate-v1/replication-transition-proposal.md),
   with the identity-implementation task now complete.
   Do not merely replace eleven hashes: the primary-only policy currently rejects the routed
   factorial AdamW control, which needs a separately reviewed integration. Preserve all old locks,
   ledger prefixes and results; keep scientific estimands, data and selection rules fixed.
3. Resolve deployment/WIP-publication authority and GitHub integration write access, then run the accepted primary revision.
   Re-audit checkpoint/optimizer/RNG persistence, source identity and remote durability.
4. After primary acceptance, complete the 840 full-corpus BEIR units and validation selection,
   weight-space/dimension analyses and the frozen crossed continuation. Historical exports cannot
   replace accepted primary inputs.
5. Render actual findings only, verify the portable evidence closure and strict manuscript gates,
   and release the paper/repository. Synthetic paper builds and passing unit tests are not results.

## Evidence and continuation map

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
