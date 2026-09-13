# Project status and evidence

Snapshot: **2026-09-13**. Scientific computation and both declared exact recovery
checks are complete; the scoped recovery add-on and complete source payload are publicly
delivered and remotely verified. Read [CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md)
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
| Exact GPU recovery | Both genuine step-313 resumes reach 391 with all 134 model tensors, complete optimizer/scheduler, all four rank RNGs and selected counters bitwise equal; eight ranks and two fresh readers exit zero | [Warm-reducer recovery](reports/engineering-archive/dense-v3-warm-reducer-recovery-v1/README.md) |
| Scoped restoration add-on | 83 tests pass; 25 package members byte-verified in wheel/sdist; five actual CPU integration tests pass from the wheel with seven original roots refused | [Usage and bounded coverage](docs/training-restoration.md), [execution evidence](reports/engineering-archive/dense-v3-portable-restore-v1/README.md) |
| Version-isolated complete paper | Actual wheel entry recomputes the entire original primary/factorial graph, checks eight shared inputs, and builds the reviewed manuscript; 194 integration tests pass | [One-command reproduction](docs/versioned-paper-reproduction.md), [actual execution](reports/engineering-archive/dense-v3-versioned-paper-reproduction-v1/README.md) |
| Reviewed default/release transition | Default/all select paper/current; actual Make default and complete release both pass; original Make preserved as legacy.Makefile; README conclusion generated from authenticated outputs | [Build instructions](paper/README.md), [all-source test entry](docs/source-version-testing.md) |

All 3,800 source-version tests now pass (current 2,873; original analysis 733;
original factorial 194), with zero failures/errors/skips. Full distribution,
style, portable evidence and CFF checks also pass. The complete source payload is
public on GitHub main at `803c70dcc3a2c195c5f56ee5fc0e3c633d645524`, with successful
normal Git push and ten anonymous byte-identical readbacks. See
[publication evidence](reports/engineering-archive/dense-v3-source-publication-v1/README.md).
`/tmp/dense-v3-release-transition.zFMpOKti/RUNNING.md` records terminal handles
and preserved failures. No scientific experiment or statistical result changed.

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

## Delivery status and remaining limitation

The complete source payload is published and anonymously readable. It includes
the reviewed manuscript, original complete numerical closure, explicit source-role
test runner, restoration tools, immutable protocols and retained execution evidence.
All 3,800 cases pass across the three declared source roles; current default/release
and original distribution checks pass.

The two declared GPU recovery cases and portable CPU/wheel adapter checks are
complete. Their coverage remains bounded: not every checkpoint, NorMuon or a
physical second host. No additional experiment or recovery run is required for
the delivered paper. Original failed attempts, unknown exits and source-role
refusals remain preserved, not rewritten as successes.

Only the previously requested historical HF erasure remains unexecuted because
the deletion was safety-rejected. It requires separate exact-scope direction and
dependency safeguards for current/shared artifacts. Do not retry or subdivide the
denied deletion. Current scientific backups are not affected.

## Runtime and access status

No scientific or recovery job remains live. The bounded recovery diagnostics
**58312 / 65312** are terminal, both coordinator exits zero, all eight ranks and
both fresh readers zero. Their 29 controls, first-gradient gates and full endpoint
comparisons pass. Read
`/tmp/dense-v3-warm-reducer-recovery.rP4NVV4i/RUNNING.md` for source/authority bindings.
Completed/failing earlier sessions must not be polled or restarted as missing work.
Protected historical stopped controllers remain protected;
their exact identities and ledger rules are in AGENTS.md and the preserved predecessor.

The GitHub 403 affected issue comment/update writes. The normal Git source push now
succeeded and was independently read back; this does not reopen the denied API surface.
The existing access question must not be repeated; no alternate credentials or denial bypass.
The rejected historical HF erasure remains rejected; nothing was deleted.

The scientific, paper and reproducible-source deliverables are complete. The separately
requested historical HF erasure remains unresolved; it must not trigger additional
training, repeated release checks or a retry of the denied deletion.
