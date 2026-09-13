# Project status and evidence

Snapshot: **2026-09-13**. Scientific computation is complete; source delivery and exact
training-recovery validation are not. Read [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md)
for the next actions and [AGENTS.md](AGENTS.md) for binding safety/integrity limits.

The full 2,380-line predecessor is preserved verbatim in
[the historical status archive](reports/engineering-archive/dense-v3-portable-handoff-v1/before/PROJECT_STATUS.md).
Its old active-job descriptions are historical, not current dispatch instructions.
No original run, result, failed check, unknown exit, authority restriction or source receipt
was removed or reclassified by this documentation consolidation.

## Completed work

| Component | Verified scope | Evidence |
| --- | --- | --- |
| Primary training and durability | 12 complete full-horizon runs; 60 checkpoints at all five stages; immutable HF checks | [Checkpoint restoration](docs/checkpoint-restoration.md), [complete native assembly](reports/engineering-archive/dense-v3-current-publication-consumer-v1/README.md) |
| Primary retrieval and selection | 840 checkpoint-task results, 14 baseline tasks, all 12 validations; validation-only selection | [Final inference](reports/dense-v3-final-inference-v1/README.md), [all trajectories](reports/dense-v3-complete-trajectories-v1/README.md) |
| Weight states and prediction | 60 states, original and exact measurements; 14 predictors; 84 exploratory comparisons / 5,040 predictions | [Weight controls](reports/dense-v3-predictor-sensitivity-v1/README.md) |
| Functional representation analysis | 61 states; 9 primary and 27 rotation contrasts; 240 predictions; 24 exploratory controls / 1,440 predictions | [Functional inference](reports/dense-v3-functional-inference-v1/README.md), [controls](reports/dense-v3-functional-sensitivity-v1/README.md) |
| Crossed continuation | 12 genuine runs, 60 checkpoints, all five-stage probes, 168 full-corpus task results | [Native closeout](reports/engineering-archive/dense-v3-final-evaluation-closeout-v1/README.md) |
| Tracking | All 24 W&B runs finished; 5,172 ordered rows / 15,516 scalar values match native histories | [Tracking audit](reports/engineering-archive/dense-v3-final-tracking-audit-v1/README.md) |
| Numerical reproduction | Complete primary and factorial numerical graph, generated findings, maps and PDF reconstruction | [Primary reproduction](docs/paper-results-reproduction.md), [complete closeout](reports/engineering-archive/dense-v3-final-evaluation-closeout-v1/README.md) |
| Manuscript | Revised complete-result text and closest references; 158-word abstract; main ends page 8; all 13 pages inspected | [Reviewed paper](reports/paper-review/dense-v3-complete-manuscript-revision-v1/README.md) |
| Stable paper entry | Twelve authenticated inputs in paper/current; actual Make and extracted-wheel rebuild; 86 focused tests pass | [Build instructions](paper/current/README.md), [actual package execution](reports/engineering-archive/dense-v3-current-paper-entry-v1/README.md) |
| Distribution audit | Original full audit now passes; genuine relocated input result unchanged; original negative controls retained and exercised | [Portable input roles](reports/engineering-archive/dense-v3-portable-input-roles-v1/README.md) |
| Primary source integration | 33 modules / 56 bindings byte-identical to actual training; all 12 sources load in one checkout; 271 focused tests + 3 current-source/CLI controls pass | [Current source](docs/current-training-source.md) |
| Saved continuation inspection | Two genuine step-313 AdamW/Muon checkpoints; original native readbacks exact using 22 wheel-local modules and explicit local runtime assets | [Portable reader and limits](docs/saved-factorial-checkpoints.md) |
| Complete numerical entry | Original full replay bundle preserved unchanged locally; all 189 inputs verified and eight generated inputs equal the reviewed paper; no scientific rerun | [Complete replay command](docs/paper-results-reproduction.md#complete-paper-replay) |

Artifact acceptance and observed OS exits remain distinct. The final two primary NorMuon
runs have complete native artifact/view acceptance, but their original OS exit codes are
unobserved. Do not infer zero from valid artifacts or W&B completion. The final tracking
API tool exit and other specifically unknown historical exits remain unasserted.

## Findings supported by the completed evidence

All values below are **nDCG@10 points**, i.e. fractional scores multiplied by 100.
The primary experiment uses one training seed and the specified four-rate grids.

- Averaging each optimizer's four declared rates gives inconclusive pairwise comparisons.
  This estimates the chosen grids, not each optimizer's global best performance.
- Validation-selected final scores are AdamW **58.9263**, Muon **59.2431** and
  NorMuon **59.3637**, compared with the pretrained **51.2961**.
  Selected NorMuon-AdamW is **+0.4374**, simultaneous 95% interval **[+0.1463, +0.7285]**.
  Selected Muon-AdamW is **+0.3168**, with its interval crossing zero.
- Selected Muon-family point estimates exceed selected AdamW at all five retained stages.
  Earlier crossing of AdamW's final score does not establish deployable wall-time savings:
  selection used full-horizon validation. The selected AdamW rate is the upper tested boundary.
- Weight displacement and degrading-attribution mass help prediction under the original
  comparator, but no tested weight or functional descriptor passes all four exploratory
  recipe comparators. Their explanatory value is therefore comparator-dependent.
- Native-coordinate Muon helpful participation differs positively from AdamW, but changes
  sign under two of three shared rotations. The joint useful-dimension criterion is not met.
  These data do not establish a basis-robust useful-capacity mechanism.

The fixed crossed continuation separates reached states from subsequent reset update rules:

| Averaged endpoint contrast | Points | Marginal 95% interval |
| --- | ---: | ---: |
| Muon-source minus AdamW-source | +0.3222 | [+0.0471, +0.6784] |
| Reset Muon minus reset AdamW | -0.5264 | [-0.8966, -0.1855] |
| State-by-operator interaction | +0.2080 | [-0.1229, +0.5551] |

The source-state contrast favors the Muon-reached state, whereas the reset-operator
contrast favors AdamW under this particular calibration and continuation horizon.
This does not establish universal optimizer superiority, within-cell significance,
a causal geometric mediator or a decomposition of the primary-training difference.
The two sources are fixed; three continuation-order seeds do not replicate their training.
Primary cells compare complete recipes, including different auxiliary rates; using common
auxiliary AdamW during continuation does not erase those source-history differences.

The [post-result cosine analysis](reports/paper-review/dense-v3-cosine-sensitivity-v1/README.md)
clarifies the finite deletion measurement and its renormalization/switching terms.
It is measurement interpretation, not a new intrinsic Muon property or proven gain mechanism.

## Durability and reuse

| Public artifact group | Actual verified scope | Restoration instructions |
| --- | --- | --- |
| Model checkpoints | All 120 current scientific checkpoints backed up and immutable metadata/hash-verified; not all re-downloaded | [Checkpoints](docs/checkpoint-restoration.md) |
| Functional vectors and calibration | 624 files / 7.5 GB, actual anonymous full download and checksums | [Functional analysis](docs/functional-analysis-restoration.md) |
| Continuation probes | 638 files / 586,119,383 bytes, anonymous full download; exact copied-source metric replay | [Continuation probes](docs/continuation-probe-restoration.md) |
| Complete continuation outcomes | 593 files / 5,199,649 bytes, anonymous full download and checksum verification | [Continuation outcomes](docs/continuation-outcomes-restoration.md) |

Immutable revisions and manifests, rather than mutable repository HEADs, identify these
artifacts. Do not repeat their completed uploads/downloads merely to restate progress.
A backup or numerical replay is not proof of bitwise GPU resume or fresh-host execution.

## Remaining delivery work

1. **Complete scientific-consumer/factorial source integration.** Primary numerics and all
   original primary bindings are now integrated in this checkout. The factorial's original
   66-file source parent differs only in the portable input module and intentionally refuses
   admission; retain its failed source-bound tests until a genuine compatibility transition.
   This is not permission to replace old source hashes or claim a unified full release.
2. **Preserve passing distribution admission.** The original full audit now passes after
   authenticated runtime roles/current documentation were separated from historical provenance.
   Keep the original scanner and all negative controls through source consolidation; see
   the [actual portable-input checks](reports/engineering-archive/dense-v3-portable-input-roles-v1/README.md).
3. **Exact training recovery.** Both repaired resume attempts reach step 391, but both exact
   endpoint readers fail. All 134 model tensors and optimizer moments differ; scheduler,
   counters and all-rank RNG match. Cause remains unresolved. See
   [the complete diagnosis](reports/engineering-archive/dense-v3-resume-endpoint-diagnosis-v1/README.md).
   Subsequent [CPU boundary checks](reports/engineering-archive/dense-v3-resume-cpu-boundaries-v1/README.md)
   show exact genuine model/optimizer loading and matching resumed index order. This
   is now followed by [actual four-rank GPU evidence](reports/engineering-archive/dense-v3-first-gradient-boundary-v1/README.md):
   both device-loaded states and first four token batches match exactly; all post-DDP
   gradients agree across ranks. Independent FP64 rank means differ by relative L2
   2.32e-8 / 2.47e-8 with fitted scale approximately one. No optimizer update was made,
   no endpoint tolerance was changed, and the prior divergence's cause remains unresolved.
   The subsequent [paired GPU boundary](reports/engineering-archive/dense-v3-paired-backward-boundary-v1/README.md)
   holds actual weights/inputs/RNG fixed and finds all 536 local contributions per rank
   identical, but 123/134 post-DDP tensors different in both cases. Relative L2 differences
   are 2.0473483980114333e-8 / 2.1862897063898537e-8; DDP bucket rebuilding is observed.
   Both coordinators and all eight ranks exit zero without any optimizer update. This
   localizes the paired difference after local differentiation, not a proven full repair.
   Independent CPU comparison artifacts are complete; its original tool exit is unknown.
4. **Final reviewed publication.** Finish the assembled-source/runtime/release-parent
   transition and strict current scientific/manuscript gates before source publication.
   Local files/builds are not evidence that GitHub has been updated.

The original failed resume attempts, failed outer-CWD numerical replay and successful
source-identical component-CWD replay are all preserved. The latter uses unchanged
I/O guards and has no refused reads/network attempts; it does not relabel the first failure.
No engineering incident belongs in the manuscript.

## Runtime and access status

No currently registered study job remains live. Completed/failing sessions must not be
polled or restarted as missing work. Protected historical stopped controllers remain protected;
their exact identities and ledger rules are in AGENTS.md and the preserved predecessor.

The GitHub 403 affected issue comment/update writes, not a tested Git push. Read access and
account-level admin/push metadata succeeded, but application write recovery remains unverified.
The existing access question must not be repeated; no alternate credentials or denial bypass.
The rejected historical HF erasure remains rejected; nothing was deleted.

Keep the goal active: the completed scientific matrix and reviewed paper are substantial
results, but the remaining release and recovery requirements have not been verified complete.
