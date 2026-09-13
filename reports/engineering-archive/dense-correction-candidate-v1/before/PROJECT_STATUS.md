# Project status and handoff

Last updated: **2026-09-06 08:32 UTC**; the machine-readable snapshot is refreshed at this handoff.

## Where the project stands

**Correctness verification has advanced; the formal matrix is still on scientific hold.**
All 12 primary runs finished and all 60 scheduled checkpoints are preserved. They were executed
with the unchanged training implementation, whose duplicated accumulation normalization produces
approximately quarter-scaled raw gradients. The tested correction is isolated, not deployed.

The new [GPU acceptance control](reports/engineering-archive/dense-canonical-replay-v1/README.md)
passes the independent global-gradient oracle on both short and 8192-token inputs. It also passes
complete checkpoint replay, including a separately launched process group. This is engineering
evidence for the explicitly controlled implementation, not a new retrieval result or validation
of the old matrix.

| Surface | Verified now | What this does not establish |
| --- | --- | --- |
| Primary execution/durability | 12/12 runs; 60/60 resumable and remotely covered checkpoints | Scientific validity |
| Controlled GPU gradients | 9/9 short-input and 9/9 maximum-length raw/clipped checks, all 134 parameter tensors | Every natural-text batch or full training horizon |
| Controlled GPU replay | 24/24 exact same-group checks and 24/24 exact new-process-group checks | Another host or every historical optimizer checkpoint |
| Regression suite | 1,367 JUnit cases, zero failures/errors/skips | Correctness of an untested runtime path |
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

This control is not automatically a formal-training default or a performance optimization:
it holds communication until all gradients are ready and allocates a full gradient vector.
The earlier failed default-path receipts and the separate production normalization defect remain
unchanged. Do not repeatedly reopen the now-resolved short-fixture replay origin question.

The independent [NorMuon reference audit](reports/engineering-archive/normuon-full-reference-v1/README.md)
still has an unresolved implementation-conformance boundary: local clamped normalization differs
from official additive epsilon on 44/96 small-matrix cases. The prior 352/352 actual-model update
matches narrow its observed impact on that fixture, but are not a universal conformance proof.

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
progress channel; new local source/evidence files are not yet available from a clean public clone.

HF withdrawal remains safety-rejected: 9,629 proposed files / 366,252,912,201 logical bytes.
No deletion occurred. Never retry, split or bypass the rejected operation. Any approved historical
erasure additionally needs refreshed shared-object/current-backup dependency checks and exact scope.

Never inspect, read, edit, signal, stop, replace or otherwise touch `gpu.py` or its processes.
Do not use broad process inspection, launch duplicate controllers, overwrite checkpoints,
relax frozen acceptance gates, print credentials or promote diagnostic numbers into the paper.

## Next bounded steps

1. Prepare and audit the minimal production correction: one loss-normalization owner and an
   explicit optimizer-reference/numerical policy. Preserve the passing diagnostic as a control;
   do not silently deploy its communication strategy as a speed improvement.
2. Prepare the state-preserving runtime handoff and the revised replication plan in new output
   namespaces. Keep data, recipe grid, selection rules and scientific estimands fixed unless
   an explicit prospective amendment is reviewed. Do not resume old-matrix scientific publication.
3. Resolve the deployment and WIP-publication boundaries, then run the accepted primary revision.
   Re-audit checkpoint/optimizer/RNG persistence, source identity and remote durability.
4. After primary acceptance, complete the 840 full-corpus BEIR units and validation selection,
   weight-space/dimension analyses and the frozen crossed continuation. Historical exports cannot
   replace accepted primary inputs.
5. Render actual findings only, verify the portable evidence closure and strict manuscript gates,
   and release the paper/repository. Synthetic paper builds and passing unit tests are not results.

## Evidence and continuation map

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
[the pre-update archive](reports/engineering-archive/dense-canonical-replay-v1/before/PROJECT_STATUS.md).
Its superseded milestone counts and launch instructions are history, not current authority.
