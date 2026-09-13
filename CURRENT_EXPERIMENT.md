# Current DenseOn experiment and delivery

**Final closeout:** [FINAL_STATUS.md](FINAL_STATUS.md) supersedes the dated
snapshot below. Complete hosted CI passed, and the owner-approved 9,629-file
HF withdrawal is executed and independently verified. No scientific/recovery
job remains live; do not repeat the old pending actions.

## Historical pre-closeout snapshot

Snapshot: **2026-09-13**. The scientific, paper and reproducible-source deliverables
are locally verified and publicly available; hosted CI repair remains pending.
Read [AGENTS.md](AGENTS.md) before acting and
[PROJECT_STATUS.md](PROJECT_STATUS.md) for findings and interpretation limits.

The source payload is published on GitHub main at
**803c70dcc3a2c195c5f56ee5fc0e3c633d645524**. Normal Git push succeeded; ten
source/handoff/manuscript/inventory files were anonymously downloaded and match
their local bytes. [Publication evidence](reports/engineering-archive/dense-v3-source-publication-v1/README.md).
This status update follows that already-verified source payload.

## Goal and scope

A defensible **NAACL paper and reproducible source/model-analysis artifacts**
about AdamW, Muon and NorMuon in DenseOn adaptation. Dense only; paper only.
No new LateOn or blog work. Never touch gpu.py or its processes.

## Verified completion

| Work | Completed scope |
| --- | --- |
| Primary training | 12/12 full runs, same 500K queries, four rates per optimizer, 60 checkpoints |
| Primary retrieval | 840/840 checkpoint-task units; 14 baseline tasks; all 12 validation selections |
| Weight analysis | All 60 states, original/exact branches, all declared predictors and exploratory recipe controls |
| Functional analysis | 61 states including pretrained; 9 primary and 27 rotation contrasts; predictions and controls complete |
| Crossed continuation | 12/12 runs, 60 checkpoints/probes, 168 full-corpus task results and six inference tables |
| Tracking | 24 finished runs; 5,172 ordered history rows / 15,516 scalars agree with native records |
| Backups | All 120 scientific checkpoints backed up; required analysis groups anonymously downloaded and verified |
| Manuscript | Reviewed complete-result paper; 158-word abstract, eight-page main, all 13 pages inspected |
| Complete reproduction | Original full numerical graph and reviewed paper pass actual version-isolated execution; eight shared inputs match exactly |
| Local source-version tests | 3,800 passes: current 2,873, original analysis 733, original factorial 194; no failures/errors/skips |
| Hosted CI | Run 34766616352 passed all 3800 cases; the latest run 34769598250 passed the exact emulated functional probe but failed before its full numerical child started. Full hosted acceptance remains pending |
| Latest local check | Both manually configured and repository-built interpreters pass the complete forced-emulation numerical-to-reviewed-paper Make release, with unchanged comparisons and eight exact shared inputs; all3800 final regression cases also pass |
| Distribution | Actual wheel/sdist build and unchanged full audit pass; style, portable evidence, credential checks and isolated CFF validation pass |
| Recovery | Both declared genuine AdamW/Muon 313-to-391 GPU restores pass complete bitwise endpoints; scoped CPU/wheel add-on checks also pass |
| GitHub | Complete source payload publicly published and anonymously read back |

There are **no live scientific or recovery jobs**. Hosted regression verification
is still being repaired; local passing results are not remote CI acceptance. All earlier failed
attempts and unknown exits are preserved; do not restart completed work as missing.
The detailed preceding handoffs are retained at the published commit above and
in the engineering archives. CURRENT_PROGRESS.json is historical accounting,
not a live heartbeat or the current scientific verdict.

## Use the delivered work

- [Reviewed paper and build instructions](paper/current/README.md)
- [Complete numerical-to-paper reproduction](docs/versioned-paper-reproduction.md)
- [Complete source-version test command](docs/source-version-testing.md)
- [Checkpoint restoration](docs/checkpoint-restoration.md)
- [Functional/calibration restoration](docs/functional-analysis-restoration.md)
- [Continuation probes](docs/continuation-probe-restoration.md) and [outcomes](docs/continuation-outcomes-restoration.md)
- [Scoped training recovery](docs/training-restoration.md)

Default `make` / `make all` selects paper/current. `make release` requires an
explicit accepted NUMERICAL_BUNDLE and new RELEASE_OUTPUT. Historical Make rules
remain byte-identical in paper/legacy.Makefile, with explicit legacy targets.
A passing source-version matrix is not admission of a changed fresh training
worker or proof of another GPU/physical-host recovery case.

## Remaining delivery verification and prior cleanup request

Finish all local verification before one final push; no PR or intermediate push.
The authentic precompiled FlashAttention artifact and original primary dependency
versions are already independently verified; do not repeat that build or upload.
See [runtime binary provenance](docs/ci-runtime-binary.md). The remaining repair
addresses exact CPU arithmetic and subprocess execution, not training or evaluation.
Both local configured-interpreter full replays and final source-role regressions
pass; hosted acceptance is tracked separately in [CPU replay](docs/cpu-numerical-replay.md).
No scientific source, result, frozen hash or numerical tolerance changes. The reviewed PDF and
83 other manifest-bound archived files are now public at commit
`65c584ce11435dc84aec675a6ed733a9a5fc97d1`; its PDF was anonymously byte-verified.

The historical Hugging Face withdrawal was safety-rejected: **9,629 paths /
366,252,912,201 logical bytes** were not deleted. Do not retry, split or bypass
that denial. Any further cleanup requires separate exact-scope direction and
protection of current/shared artifacts. Current scientific backups remain intact.

The old GitHub issue-write 403 concerns a separate API surface. Normal Git
source publication succeeded; do not bypass or repeatedly retry that issue API.
Neither access history justifies new experiments or repeated release checks.

## Interpretation limits

All-rate primary optimizer contrasts are inconclusive. Validation-selected
NorMuon exceeds AdamW by +0.4374 nDCG@10 points (simultaneous 95% interval
[+0.1463, +0.7285]); selected Muon is +0.3168 with its interval crossing zero.
These are one-seed/grid-specific estimates, not independent seed replications.

No tested weight or functional predictor passes all four exploratory recipe
comparators. Native helpful participation is not rotation-stable. The fixed
crossed continuation separates reached state from subsequent update rule:
state +0.3222, reset operator -0.5264, interaction +0.2080 points. These marginal
intervals and two fixed source histories do not establish universal superiority,
basis-robust useful capacity or mediation. Keep implementation-error narrative
out of every part of the manuscript.
