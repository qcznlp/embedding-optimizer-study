# Current DenseOn experiment and handoff

Primary retrieval completed **2026-09-12 10:47 UTC**; complete copied-source
numerical replay completed **11:08 UTC**. This is a dated completion snapshot,
with the last outstanding fourth-stage score backups recovered/reconstructed at **11:25 UTC**,
not a live heartbeat or full scientific completion. Original pool/observer receipts
are in the [complete-trajectory archive](reports/dense-v3-complete-trajectories-v1/README.md);
the [paper-framing snapshot](reports/paper-review/dense-v3-retrieval-usefulness-v1/observations.json) is unchanged.
Read [AGENTS.md](AGENTS.md) before acting; its safety and authority limits still apply.

## Goal

Deliver a defensible **NAACL paper and reproducible repository/model-analysis
artifacts** about how AdamW, Muon and NorMuon change DenseOn weight space and
retrieval. DenseOn is the only active architecture. No new LateOn work or blog.

The evidence chain is: complete retrieval comparison → reached weight states and
trajectories → functional dimension utility → held-out-dose prediction, with a
separately bounded state-by-reset-operator continuation. Spectral fingerprints
alone are not an embedding-specific explanation; prediction is not mediation.

## Current evidence

| Component | Verified state | What remains |
| --- | --- | --- |
| Corrected primary training | **12 / 12 complete**, no training run queued | Do not retrain these completed recipes |
| Training-completion portability preparation | **[172 original evidence files / all 12 runs / 60 checkpoints](reports/engineering-archive/dense-v3-campaign-evidence-candidate-v1/README.md)** read by an isolated local-role adapter; 23 reader + five composition tests pass, actual recovered-HF composition and isolated read agree | Not installed into the formal contract or scientific consumer. Preserve ten observed exit-zero and two unobserved/null exits; reader/source remains local, with no GitHub publication |
| Original completion-record durability | **[41 files / 342,406 bytes](docs/training-completion-evidence-restoration.md)**, including all 39 original additional records, anonymously recovered and independently read; recovered-only inputs reconstruct the same 172-file evidence | Immutable data-only addition; all ten prior corrected subtrees and root entries unchanged. No new training, tensor verification, functional recovery or source release |
| Native training dynamics | **12 / 12**, all 4,692 logged observations and 60 stage summaries independently replayed | Descriptive curves/timing, not retrieval or mechanism inference |
| Training-artifact durability | **149 / 149**, 26.99 MB, anonymous recovery and recovered-native numerical replay verified | Original root-byte exception retained; see recovery caveat below |
| Primary checkpoint durability | **60 / 60**, 1,200 files, 89,889,820,336 logical bytes | Final source/analysis release remains separate |
| Zero-update baseline | **14 / 14 full-corpus tasks complete** | Reference only, not an optimizer comparison |
| Primary BEIR | **840 / 840 task cells complete**; both original pools completed 420 cells each, final observer completion present | No primary training or BEIR task remains; do not restart or poll retired live-only handles |
| Complete primary checkpoint BEIR | **[60 / 60](reports/dense-v3-complete-trajectories-v1/README.md)**, every configuration at all five retained steps; original native readback and complete copied-source reconstruction pass | Previous 54 records exactly unchanged. All 60 outcomes now have raw-score backups; model checkpoints are also all backed up |
| Complete retrieval trajectories | **[Eight actual tables and all twelve curves](reports/dense-v3-complete-trajectories-v1/descriptive/summary.md)** generated; copied-source replay reproduces every CSV, all six endpoint contrasts unchanged | Latest tables/figures need data-only backup. Curves are descriptive, not seed robustness, a functional mechanism or formal publication-consumer admission |
| Complete fourth-stage evaluation durability | **[557 / 557 files](docs/fourth-stage-evaluation-restoration.md)**, 1,959,719 bytes; anonymous recovery and isolated reconstruction of 168 scores / twelve exact means passed | Six states overlap the previous pool-B backup; six are new. All sixty primary outcomes are now backed up. No source publication or new model/statistical execution |
| Fourth-stage pool-B score table | **[6 / 6 original configurations](reports/engineering-archive/dense-v3-fourth-stage-pool-b-evaluations-v1/tables/fourth-stage-pool-b-checkpoint-scores.md)**, 84 new task values; all previous 48 complete records exactly unchanged | Immutable data-only backup and anonymous reconstruction below are complete; no partial-cohort optimizer-wide or mechanism verdict |
| Fourth-stage pool-B evaluation durability | **[281 / 281 files](docs/fourth-stage-pool-b-evaluation-restoration.md)**, 983,157 bytes; anonymous recovery and independent reconstruction of all 84 scores / six means passed | New immutable addition only; prior eleven corrected subtrees and root entries unchanged. No source publication or new model/statistical execution |
| Complete third-stage score table | **[12 / 12 configurations](reports/engineering-archive/dense-v3-complete-third-stage-evaluations-v1/tables/third-stage-checkpoint-scores.md)**, all 168 task scores; only six pool-A checkpoints / 84 values are new, and the previous 42 complete records are unchanged | All twelve outcomes now have the complete third-stage HF snapshot below; the earlier six-state pool-B snapshot remains unchanged |
| Complete third-stage evaluation durability | **[557 / 557 files](docs/third-stage-evaluation-restoration.md)**, 1,959,113 bytes; anonymous recovery and independent reconstruction of all 168 scores / twelve means passed | All 48 complete checkpoint outcomes now have off-host score backups. Six states overlap the prior pool-B snapshot; no source publication or new inference |
| Third-stage pool-B score table | **[6 / 6 original configurations](reports/engineering-archive/dense-v3-third-stage-pool-b-evaluations-v1/tables/third-stage-pool-b-checkpoint-scores.md)**, 84 new task values; previous 36 complete records unchanged | Immutable data-only HF snapshot below; this is not the full twelve-configuration third-stage comparison |
| Third-stage pool-B evaluation durability | **[281 / 281 files](docs/third-stage-pool-b-evaluation-restoration.md)**, 982,548 bytes; anonymous recovery and independent reconstruction of all 84 scores / six means passed | New immutable addition only; prior seven corrected subtrees unchanged. No source publication or new scientific inference |
| Complete second-stage score table | **[12 / 12 configurations](reports/engineering-archive/dense-v3-complete-second-stage-evaluations-v1/tables/second-stage-checkpoint-scores.md)**, all 168 task scores; only six pool-A checkpoints / 84 values are new, and the previous 30 records are unchanged | All twelve outcomes now have the complete second-stage HF snapshot below; the earlier six-state pool-B snapshot remains unchanged |
| Complete second-stage evaluation durability | **[557 / 557 files](docs/second-stage-evaluation-restoration.md)**, 1,961,293 bytes; anonymous recovery and independent reconstruction of all 168 scores / twelve means passed | Six pool-B states overlap the earlier snapshot; only six pool-A states / 84 values are newly backed up. No source publication or new scientific inference |
| Step-1563 pool-B evaluation preservation | **[281 / 281 files](docs/second-stage-pool-b-evaluation-restoration.md)**, 983,874 bytes; six checkpoints / 84 task values anonymously recovered and independently reconstructed | New immutable data-only snapshot; not the complete twelve-configuration second stage or source publication; previous snapshots unchanged |
| First-stage evaluation durability | **[557 / 557 files](docs/first-stage-evaluation-restoration.md)**, 1,957,694 bytes; anonymous recovery and independent reconstruction of all 168 first-stage scores / twelve means passed | Separate immutable data-only addition; no middle-stage outcomes or source publication |
| Validation-only LR selection | **12 / 12 complete**, native all-row readback and fixed selector verified | Coordinator terminal; do not poll its stale live-only handle or rerun validation |
| Frozen final-checkpoint inference | **All six contrasts computed and unchanged in the complete trajectory replay**: primary four-rate averages inconclusive; selected NorMuon−AdamW +0.437 points, simultaneous interval [+0.146, +0.729] | Secondary, task-level support only; seed robustness and mechanism remain unestablished |
| Benchmark support / endpoint decomposition | **14 tasks, 28 query/qrel files, 84 signed task contributions** checked; independent counts/arithmetic agree | Post-hoc description; no new inference, task exclusion or query-level uncertainty estimate |
| Endpoint-statistics durability | **16 / 16 files**, 218,466 bytes, including every contrast and PDF/PNG/SVG figure; complete anonymous recovery and independent numeric mapping verified | Existing analysis transported, not new inference or source publication |
| Endpoint/validation artifact durability | **660 / 660 files**, 29.53 MB, complete anonymous recovery and offline score reconstruction verified | Final checkpoints only; intermediate evaluations and source release remain separate |
| Weight geometry and update map | Both geometry branches **60 / 60**; all-state map/readback complete | Link to complete functional/retrieval outcomes |
| Weight-artifact durability | **298 / 298 files**, 4.98 GB, immutable public backup and complete anonymous recovery verified | Not a source release or second-host experiment |
| Functional dimension analysis | **[1 / 61 vector states accepted, 0 / 61 feature states](reports/engineering-archive/dense-v3-first-stage-evaluations-v1/functional-first-state/README.md)**; original coordinator terminal, exit 1 | Pretrained state passed native readback; next-state receipt-path creation failed before its worker started. Preserve outputs; no automatic retry or approved recovery yet |
| Accepted pretrained-vector durability | **[10 / 10 files](docs/pretrained-vector-restoration.md)**, 6,439,310 bytes; complete anonymous HF recovery and isolated NumPy array/provenance readback passed | Preserves the existing one vector state only. No new encoding/features or failed-coordinator recovery; copy the local guide/reader off-host separately |
| Functional record-layout preparation | **[19 bounded CPU filesystem tests passed](reports/engineering-archive/dense-v3-functional-record-layout-candidate-v1/README.md)** on an isolated candidate; all 61 cell paths and six JSON families exercised with model/process stubs | Candidate is not installed; full recovery coordinator, approved execution, baseline reuse and real outcomes remain pending |
| Crossed continuation | **0 / 12 formal branches** | GPU calibration/admission, worker handoff, branches and complete outcomes |
| Paper / source release | Narrative/method definitions revised; draft builds, **not complete** | Verified generated findings, portable reconstruction and release gates |

The [actual complete-trajectory readout](reports/dense-v3-complete-trajectories-v1/README.md)
now contains all 60 checkpoint outcomes / 840 task scores, all eight statistical
and dynamics tables, PDF/PNG/SVG curves, and generated descriptive tables. The
unchanged native reader returned zero at **10:51 UTC**; the original statistical
functions ran at **10:52 UTC** and copied-source replay at **11:08 UTC** reproduces
all eight CSVs byte-for-byte. All 54 previously accepted records and all six
endpoint effects/intervals/decisions remain exactly unchanged. Both original
evaluation pools and the observer have native completion records. Do not poll
their old live-only handles or restart any completed evaluation.

Validation-selected Muon and NorMuon have higher macro point estimates than
selected AdamW at each of five retained stages; their 40% points exceed selected
AdamW's final point. These are descriptive facts, not new significance tests,
cross-seed robustness or a 60% wall-time-saving claim. Full-horizon validation
was used for selection. Functional analysis remains one accepted pretrained
vector state and zero feature states, with no approved recovery; formal crossed
continuation remains zero of twelve. Full-trajectory tables/figures still need
immutable data-only backup; the last six raw outcomes are now preserved below. Manuscript/source release
and historical safety boundaries remain unchanged.

The [complete fourth-stage data backup](reports/engineering-archive/dense-v3-complete-fourth-stage-artifact-backup-v1/README.md)
now preserves all twelve step-3126 outcomes, with six prior pool-B overlaps and
six newly backed pool-A states. All **557 files / 1,959,719 bytes** were anonymously
recovered at **11:23:59 UTC**, and the copied isolated stdlib reader reconstructed
all 168 raw scores, 168 successful worker records, twelve means and 180 CSV rows
at **11:25:49 UTC**. Revision `5eb4cceed5c3dbee05d46850ec0d76aedff7d4cc`
preserves twenty other root entries and all twelve prior corrected subtrees.
All sixty checkpoint outcomes now have raw-score backups. This is not a source
release, new inference, functional recovery or second-host model experiment.

### Historical evaluation milestones

The following observations preserve their original timestamps; incomplete-grid
counts and then-live handles are historical, not current work to repeat.

The [complete-trajectory candidate](reports/engineering-archive/dense-v3-complete-trajectory-candidate-v1/README.md)
supplied a local data-only entry and complete-population renderer. The actual
available-input check at **09:49:47 UTC** and copied-source replay at **09:55:11 UTC**
agree on all 54 checkpoint records / 756 task values and refuse the incomplete
grid before creating trajectory outputs. The final **97 adapter + 27 renderer-input
checks** pass. Original summary/statistical functions remain unchanged, and an
AST comparison confirms the original unused-auxiliary nAUC policy. Renderer
fixtures and the area fixture are explicitly synthetic, not measured trajectories.
No actual complete-grid statistics or figure, new model execution, functional
recovery, crossed branch, manuscript finding or source release occurred.
At the separately timestamped **09:54 UTC** native observation, all four original
pool-A workers remain live and **780 / 840** tasks are complete. The eight new
native task exits since the 772-task backup snapshot are successful; the latest
four exits and exact successor source/authority records are retained with this
candidate. The six incomplete checkpoint states remain required.

The subsequent [six complete fourth-stage pool-B readback](reports/engineering-archive/dense-v3-fourth-stage-pool-b-evaluations-v1/README.md)
completed at **2026-09-12 09:09:50 UTC**. Independent raw-file reconstruction at
**09:10:48 UTC** verifies all 54 checkpoint outcomes, 756 raw task scores,
756 original exit-zero workers, 864 score/metadata snapshots and exact rational
means. Only six original pool-B step-3126 states / 84 values are new; all 48
previous records are exactly unchanged. An archived-source replay at **09:11:27 UTC**
reproduces three generated tables byte-for-byte. These six new outcomes are locally
archived only and not yet added to the earlier HF backups. Original pool B completed
all 420 tasks / 30 checkpoints at **08:59:47 UTC**; four original pool-A workers
remain live at the **09:11 UTC** snapshot, with 772 / 840 tasks complete. No queue,
source, statistical rule, functional recovery, crossed branch or manuscript finding
changed. The six other step-3126 states and complete downstream analysis remain.

Those six step-3126 outcomes now have a separate
[immutable data-only recovery entry](docs/fourth-stage-pool-b-evaluation-restoration.md).
All **281 files / 983,157 bytes** were anonymously recovered at **09:19:12 UTC**;
the isolated stdlib reader reconstructed 84 raw scores, 84 original exit-zero
workers, six exact means and 90 CSV rows at **09:19:29 UTC**. Revision
`dd16fe4cfe3e21ba8c8b4583038ed93fe4ce6e92` preserves twenty other root entries
and eleven prior corrected subtrees. All 54 complete checkpoint outcomes now
have off-host backups. The [local evidence archive](reports/engineering-archive/dense-v3-fourth-stage-pool-b-artifact-backup-v1/README.md)
preserves final 46 adapter / 18 reader tests and actual one-shot transport/replay
records. This is same-host recovery, not a source release, new inference or
functional/coordinator recovery. At the **09:19 UTC** native snapshot the
remaining four pool-A workers are live and 772 / 840 task cells are complete.

The training/durability evidence is the content-bound
`launch/observations/all-twelve-training-sixty-checkpoints-evaluation-running-20260910.json`
in the experiment tree, SHA-256
`1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3`.
Ten original worker exits are observed zero; the final two have verified complete
artifacts and observed termination, with OS exit codes **unobserved/null**.
Do not manufacture exit records or weaken the preserved original guards.

The complete final-checkpoint score grid and its fixed validation-selected comparison
are now available. A full-trajectory optimizer verdict and useful-dimension mechanism
are not established.
Current weight displacements differ in spectral concentration,
but that is not proof of more useful embedding dimensions. Learning-rate cells
are not independent training seeds. Historical scores cannot fill missing v3 cells.

The [complete final-checkpoint readback](reports/engineering-archive/dense-v3-all-final-evaluations-v1/README.md)
now contains every declared optimizer/rate cell at step 3907. The unchanged native
reader verifies all twelve original complete receipts; independent raw-score
reconstruction verifies all 168 task values and all twelve means. An archived
table replay reproduces the CSV/Markdown bytes. The first six-record archive is
preserved and overlaps this cohort; it is not six additional runs. No manuscript
result, validation-based selection, significance test or mechanism conclusion
was installed by this descriptive endpoint milestone.

The subsequent [first intermediate-checkpoint readback](reports/engineering-archive/dense-v3-first-intermediate-evaluations-v1/README.md)
completed at **2026-09-11 01:39:29 UTC**. Six step-782 checkpoints, exactly the
original pool-B cohort, then obtained complete fourteen-task results. The unchanged
native reader verified the eighteen complete checkpoint records available then;
independent raw-score reconstruction checked all 252 task values and exact means.
Only six states / 84 task values were new; the twelve previous final records remain
unchanged overlapping evidence. That historical observation was not the full
first-stage rate grid and did not support an optimizer-wide partial-cohort comparison.

The subsequent [complete first-stage readback](reports/engineering-archive/dense-v3-first-stage-evaluations-v1/README.md)
finished at **2026-09-11 03:20:45 UTC**. All twelve configurations at steps 782 and
3907 now have complete results: **24 / 60 checkpoint evaluations**. The original
native reader passes, and independent reconstruction checks all **336 raw task
values**, 336 original exit-zero workers, 384 score/metadata snapshots and every
exact mean. The preceding eighteen records remain exactly unchanged; only the
six pool-A step-782 states / 84 values are new. Thirty-six intermediate checkpoints
remained at that observation. Both original pools had naturally entered step 1563. No new statistical
test, rate selection, trajectory interpolation or manuscript finding was installed.

The subsequent [first six complete second-stage readback](reports/engineering-archive/dense-v3-second-stage-pool-b-evaluations-v1/README.md)
finished at **2026-09-11 12:00:20 UTC**. The original pool-B six step-1563 states
bring full checkpoint coverage to **30 / 60**. Independent raw-file reconstruction
at **12:04:49 UTC** verifies 420 raw task values, 420 original successful evaluation
workers, 480 raw-score/metadata snapshots and every exact rational mean. All 24
previous complete records are unchanged; only six checkpoints / 84 values are new.
The six other step-1563 states and all 24 states at steps 2345 / 3126 remain required.
This partial learning-rate cohort is descriptive only, not a new optimizer-wide
comparison, trajectory verdict or mechanism finding. At the **12:06 UTC** snapshot,
pool B is evaluating step 2345 and pool A continues step 1563. The new outcomes are
locally archived, not claimed as part of the earlier off-host evaluation backups.

Those six original pool-B step-1563 outcomes now have a separate
[immutable data-only recovery entry](docs/second-stage-pool-b-evaluation-restoration.md).
All **281 files / 983,874 bytes** were anonymously recovered at **12:43:54 UTC**;
a byte-identical standalone reader reproduced all 84 raw scores, 84 original
exit-zero worker records, six exact means and 90 CSV rows at **12:44:33 UTC**.
The new revision is `35ce505d3cf3389f4f1a2f35b46259749bc1d7ed`; all twenty other
root entries and five earlier corrected subtrees, including root attributes,
remain unchanged. This is same-host data recovery and numerical replay, not
another model run, a full twelve-configuration second stage, source publication
or new statistical evidence. The [local evidence archive](reports/engineering-archive/dense-v3-second-stage-pool-b-artifact-backup-v1/README.md)
preserves the actual 34 adapter / 18 reader tests, commands and remote/recovery
receipts. Original primary and failed functional sources are unchanged.

The subsequent [complete second-stage readback](reports/engineering-archive/dense-v3-complete-second-stage-evaluations-v1/README.md)
finished at **2026-09-11 13:52:01 UTC**. All twelve configurations at steps
782, 1563 and 3907 now have complete results: **36 / 60 checkpoints**.
Independent raw-file reconstruction at **13:52:43 UTC** verifies all 504 raw
task values, 504 original exit-zero workers, 576 score/metadata snapshots and
every exact rational mean. All thirty previous records are exactly unchanged;
only six original pool-A step-1563 checkpoints / 84 task values are new.
The full second-stage table retains all twelve configurations / 168 values,
including the previously accepted pool-B six; those overlaps are not new runs.
A source-relocated replay at **13:54:41 UTC** reproduces all three table bytes
on this host. The new six-state outcomes are locally archived, not yet added
to the earlier HF backups. At the **13:55 UTC** snapshot both original pools
are evaluating step 2345, with eight live workers and **504 / 840** completed
task cells. All 24 checkpoints at steps 2345 / 3126 remain required.
No new rate selection, statistical test, full-trajectory or mechanism conclusion,
functional recovery, manuscript finding or source publication was installed.

Those complete twelve step-1563 outcomes now have a separate
[immutable data-only recovery entry](docs/second-stage-evaluation-restoration.md).
All **557 files / 1,961,293 bytes** were anonymously recovered at **14:27:01 UTC**;
the copied standalone reader reproduced all 168 raw scores, 168 original exit-zero
workers, twelve exact means and 180 CSV rows at **14:29:06 UTC**.
Revision `aa1515cbf98cb1b2af0be58cf05a4985bf1254cb` preserves all twenty other
root entries and six earlier corrected subtrees, including root attributes.
Only six pool-A states / 84 values are newly backed up; the six pool-B overlaps
are not new runs. The [local evidence archive](reports/engineering-archive/dense-v3-complete-second-stage-artifact-backup-v1/README.md)
retains all 34 adapter / 18 reader tests and actual upload/recovery commands.
This is same-host data recovery, not model re-execution or source publication.
At the separately timestamped **14:29 UTC** snapshot, the original eight workers
remain live at **508 / 840** tasks; the four newer native task exits since the
504-task handoff are verified zero. Full checkpoint coverage remains **36 / 60**.
No functional recovery, crossed branch, statistical rule or manuscript result changed.

The subsequent [six complete third-stage pool-B readback](reports/engineering-archive/dense-v3-third-stage-pool-b-evaluations-v1/README.md)
finished at **2026-09-11 22:39:50 UTC**. Complete checkpoint coverage is now
**42 / 60**: all twelve configurations at steps 782, 1563 and 3907, plus the
six original pool-B step-2345 states. Independent raw reconstruction passed at
**22:40:25 UTC**, checking all **588 task values, 588 original exit-zero workers
and 672 score/metadata snapshots**, with all exact rational means reconstructed.
All 36 prior records are unchanged; only six checkpoints / 84 task values are new.
The archived-source replay at **22:40:47 UTC** reproduces three table files
byte-for-byte on this host. These new outcomes are locally archived, not yet in
the previous HF evaluation backups. At the separately timestamped **22:41 UTC**
snapshot the original eight workers are live and **604 / 840** tasks are complete;
pool A continues step 2345 and pool B has naturally entered step 3126. The six
pool-A step-2345 states and all twelve step-3126 states remain required. This
partial rate cohort is not a new optimizer-wide comparison, full trajectory or
mechanism claim. Functional recovery, crossed branches, endpoint inference and
the manuscript remain unchanged.

Those six original pool-B step-2345 outcomes now have a
[separate immutable data-only recovery entry](docs/third-stage-pool-b-evaluation-restoration.md).
All **281 files / 982,548 bytes** were anonymously recovered at **22:54:21 UTC**;
the copied isolated stdlib reader reproduced all 84 raw scores, 84 original
exit-zero workers, six exact means and 90 CSV rows at **22:54:42 UTC**.
Revision `dbd16adcc835c2c6b7537a61b8d511f027a2c351` preserves all twenty other
root entries and seven earlier corrected subtrees, including root attributes.
The [backup archive](reports/engineering-archive/dense-v3-third-stage-pool-b-artifact-backup-v1/README.md)
retains the original failed module-path invocation, corrected 34 adapter / 18
reader tests, successful one-shot upload, console-parse reconciliation and actual
recovery records. This is same-host data recovery, not source publication or new
model/statistical execution. At the **22:55 UTC** snapshot primary evaluation is
**608 / 840**, with eight live workers; all four new native task exits since the
604-task observation are verified zero. Full checkpoint coverage remains 42 / 60.
Functional recovery, crossed branches and the manuscript are unchanged.

The subsequent [complete third-stage readback](reports/engineering-archive/dense-v3-complete-third-stage-evaluations-v1/README.md)
finished at **2026-09-12 00:20:37 UTC**. All twelve configurations at steps
782, 1563, 2345 and 3907 now have complete results: **48 / 60 checkpoints**.
Independent raw-file reconstruction at **00:21:17 UTC** verifies all **672 task
values, 672 original exit-zero workers and 768 score/metadata snapshots**, with
every exact rational mean reconstructed. All 42 previous records remain exactly
unchanged; only six original pool-A step-2345 checkpoints / 84 values are new.
The twelve-row third-stage table includes the six already accepted pool-B states.
A fresh archived-source replay at **00:24:13 UTC** reproduces all three table
files byte-for-byte on this host. The six new outcomes are locally archived,
not yet added to the earlier HF backups. At the **00:24 UTC** snapshot both
original pools are evaluating step 3126, with eight live workers and **672 / 840**
completed task cells. All twelve step-3126 outcomes remain required. No new rate
selection, statistical test, functional recovery, crossed branch, manuscript
finding or source publication was installed.

Those complete twelve step-2345 outcomes now have a separate
[immutable data-only recovery entry](docs/third-stage-evaluation-restoration.md).
All **557 files / 1,959,113 bytes** were anonymously recovered at **00:59:24 UTC**;
the copied isolated stdlib reader reconstructed all 168 raw scores, 168 original
exit-zero workers, twelve exact means and 180 CSV rows at **00:59:53 UTC**.
Revision `fdad53c239d9ed59141fcafbd92a0438bb1a657f` preserves all twenty other
root entries and eight prior corrected subtrees, including root attributes.
Only six pool-A states / 84 values are newly backed up; the pool-B overlap is
not another experiment. The [backup archive](reports/engineering-archive/dense-v3-complete-third-stage-artifact-backup-v1/README.md)
retains 46 adapter / 18 reader tests and actual upload/recovery evidence, including
the original client's successful HTTP 429 backoff and a truncated console chunk.
All **48 complete checkpoint outcomes** now have off-host score backups. At the
separately timestamped **00:59 UTC** snapshot, primary evaluation is **676 / 840**
with eight original workers live. Functional recovery, crossed branches, original
numerical sources, endpoint inference and the manuscript remain unchanged.
This is same-host data recovery, not source publication or a model rerun.

The earlier first-stage handoff also records a [terminal functional-coordinator failure](reports/engineering-archive/dense-v3-first-stage-evaluations-v1/functional-first-state/README.md).
The pretrained GPU worker encoded successfully and the original native save/readback
passed: **one vector state accepted, no features computed**. Creating the next
checkpoint's admission record failed because its slash-containing cell identifier
requires an uncreated parent directory. The original session 39956 is terminal
with **exit 1**, not a live GPU wait. A read-only audit finds sixty nested checkpoint
cells and twelve missing record directories; the old observer's top-level glob also
does not cover future nested records. Original sources, authority, failure and the
accepted raw vectors are preserved. No second model worker, restart, numerical
change or recovery coordinator was launched. A scoped recovery is not yet authorized;
automatic continuations do not grant it. This engineering history stays out of the paper.

The subsequent [complete validation readback](reports/engineering-archive/dense-v3-validation-selection-readback-v1/README.md)
verifies all 49,152 scored rows and the original loss-only selection: AdamW 3e-5,
Muon 3e-4, NorMuon 3e-4. The [selected endpoint table](reports/engineering-archive/dense-v3-validation-selection-readback-v1/tables/selected-endpoints.md)
joins BEIR only after fixing those choices. It is a descriptive comparison, not
a significance test, multi-seed result or mechanism finding. All 48 original
per-row validation output files remain at their authenticated experiment paths.
The small native readback archive is unchanged; a subsequent separate
[immutable evaluation backup](docs/evaluation-analysis-restoration.md) now preserves
all raw final BEIR/validation outputs and original worker records off host.
All 660 files were anonymously recovered, and the recovered scores reproduce all
168 task values, twelve endpoint means, validation metrics and fixed choices.
This is aggregate-score reconstruction, not a new model forward or significance test.

The subsequent [frozen endpoint statistical readout](reports/dense-v3-final-inference-v1/README.md)
now computes both predeclared families using all 168 final task cells. All three
four-rate-average contrasts are inconclusive. For validation-selected recipes,
NorMuon−AdamW is +0.4374 points with a simultaneous 95% interval [+0.1463, +0.7285];
Muon−AdamW and NorMuon−Muon remain inconclusive. The unchanged procedure uses
50,000 paired-task draws, seed 20260903, and three contrasts per family.
Independent scalar replay and source-relocated reconstruction pass. These are
task-level intervals for one training seed, not seed-robustness or causal evidence.
One positive comparison and one inconclusive comparison do not imply that the
two treatments differ. The original whole-840-cell outcome reader and manuscript
acceptance remain unchanged and unpassed; no missing intermediate result is imputed.

Those unchanged endpoint statistics and all three figure formats now have a
separate [immutable recovery entry](docs/endpoint-statistics-restoration.md).
All sixteen files were anonymously recovered at 18:54:17 UTC; independent byte
and table/figure checks passed at 18:55:02. All six comparisons, including the
inconclusive ones, are retained. This is transport of the preceding actual
analysis, not new statistical evidence, model execution or source release.

The [complete native training curves and tables](reports/engineering-archive/dense-v3-training-observations-v1/README.md)
now include every rate, all five stages and original timing/system records. The
independent native numerical read and a fresh copied-input replay pass; all 143
files, including nine outputs and the manifest, reproduce byte-for-byte on this
host. The 35 focused guard tests also pass. Logged loss ends at step 3900, not
3907; sampled gradient norms are not an every-update clipping count. Accepted
times include checkpoint saves and do not demonstrate a large whole-training
speed advantage. That observation milestone did not change source/admission flags or install
manuscript findings.

The [paper-framing revision](reports/paper-review/dense-v3-retrieval-usefulness-v1/README.md)
develops the distinction between weight-space spread and retrieval-useful coordinate allocation.
It specifies task-first margin attribution, helpful mass share, 768-normalized participation,
and the different definitions of truncated- and full-spectrum entropy. Numerical sources,
protocols, generated result includes and active dispatchers are unchanged. The ordinary draft
build has a six-page main text and retains all pending result markers; it is not a completed
paper or a source release. Prior manuscript/handoff bindings remain available in exact before-copies.

A separate [coordinate-sensitivity interpretation check](reports/paper-review/coordinate-utility-interpretation-v1/README.md)
now supplies exact identities and a reproducible dense 768-dimensional counterexample: all three
raw utility features can move in the desired directions while every full-vector shortlist score
and the query/document span rank remain unchanged. This is a synthetic logical boundary, **not a
Muon/AdamW result or training defect**. It supports the existing requirement to connect sensitivities
to full-corpus outcomes; it does not change metrics, statistical rules, required experiments, or
manuscript findings. Do not treat its deterministic numerical bounds as statistical intervals.

The [benchmark-support readout](reports/dense-v3-benchmark-support-v1/README.md),
completed at **22:25 UTC**, authenticates query/qrel support against the exact
revisions in all 168 native final results. NQ has **26 qrel-covered test queries**
(not 26 rows in its 3,127-row query file), versus FEVER's 6,571; each task still
has fixed macro weight 1/14. DBPedia and SCIDOCS also retain 85 and 71 queries,
respectively, with no positive judgment in their pinned qrels. Nothing is excluded.
Selected Muon−AdamW's NQ contribution is +0.19293 macro points, **60.90% of its
net +0.31679 difference**; this is a post-hoc arithmetic decomposition, not a
query-level, causal or significance claim. All six contrasts, negative task
contributions and original inference decisions remain visible. No paper finding,
new inferential rule, GPU rerun or public source release was installed.

## Which experiment and source are current?

The fixed [v3 primary protocol](configs/dense_primary_v3_protocol.json) has SHA
`4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b`.
All runs use the same revised 500K groups, seed 42, one positive/seven negatives,
no in-batch negatives, temperature 0.02, length 8192 and global batch 128.
AdamW rates are 1e-6 / 3e-6 / 1e-5 / 3e-5; Muon and NorMuon rates are
1e-4 / 3e-4 / 1e-3 / 3e-3. Steps are 782 / 1563 / 2345 / 3126 / 3907.

These are fresh full-horizon `verified-v3-*` runs from the same pretrained base,
not continuations of the older scientifically held campaign. The owner's
training-priority exception allowed the frozen, content-bound execution while
deferring code publication. It did not erase draft/release gates or authorize
unreviewed controller, source, HF-deletion or GitHub transitions.

Host-specific paths below are locations, not commands to execute on another host:

| Role | Current location |
| --- | --- |
| Development / paper / handoff | `/root/embedding-optimizer-story-refactor` |
| Frozen 56-file primary assembly | `/root/embedding-optimizer-primary-v3` |
| Actual data / runs / evaluations / active launch records | `/root/embedding-optimizer-v3-experiment` |
| Historical experiment and retained source data | `/root/embedding-optimizer-study` — not the current trainer |
| Primary local models | Experiment `outputs/dense-correctness-v3/dense/verified-v3-*/checkpoint-*` |
| Primary HF models | The **60 exact entries** in [the download index](docs/primary-v3-checkpoints.json) |

The independent primary assembly SHA is
`e603ef175c0f691a1dd83c4d8b474d90da2fc67c8c65f5a77f3c3180e78332b8`.
The current interpreter on this host is `/usr/bin/python`, not the old `.venv`.
The [formal runtime](configs/formal_runtime.json) and imported source identities
must match; a development installation does not certify a formal worker.

## Safe current observations

These **read existing observers**; they do not create jobs, take leases or resume
anything. They are host-specific. If a handle is missing, inspect original
receipts before proposing any action; an observation timeout is not termination.

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /root/embedding-optimizer-v3-experiment/launch/task-observer/status.py
```

The baseline and weight-geometry jobs are terminal; do not poll their stale
live-only handles or rerun them. Validation is also terminal: its former exact
PID 13007/start 298006520 is gone, all twelve original worker exits are zero and
the original completion/selection pass native readback. The coordinator's own
OS exit code is unobserved, not invented. Do not call its old live-only inspector,
restart its watch loop or repeat its empty-pilot preflight. Original pool B has
completed its queue; do not restart it or poll its former task handles. Pool A
has now also completed its queue, and the observer has a native completion record.
Do not poll observer 60083/start 299816818 or either pool's former live-only handles.
No primary evaluation remains queued. The functional coordinator is also
terminal: original PID 42916/start 299159282 is gone, its session 39956 exited 1,
and its original failed.json plus the accepted pretrained records are preserved
in the linked functional handoff. Do not poll its stale live handle or restart the
old queue. Its waiting-for-validation file is historical, not its current state.
The remaining sixty vector states and full feature/inference chain need a separately
authorized recovery; the original numerical protocol remains unchanged.
No scheduling change was made for either readback. The one-GPU priority question is
**unanswered, not approved**. Do not re-ask it, infer approval from an automatic
continuation, stop a lease-holding parent or force-unlock a slot.

## Next actions, in scope

1. Back up the complete trajectory tables/figures as a new immutable data-only
   addition, then anonymously recover and reconstruct them. The complete
   fourth-stage snapshot and its isolated native-score reconstruction are done:
   all 60 outcome backups, all 60 model checkpoints, completed validation/selection
   and all prior snapshots remain unchanged.
   Do not upload native aggregate bundles containing embedded Python as data-only
   artifacts. Complete primary retrieval and its actual eight-table replay are
   done; do not restart workers or repeat completed checkpoints as missing work.
2. Resolve the scoped functional receipt-path recovery authority, preserving the
   failed attempt and authenticating/reusing the accepted pretrained state without
   re-encoding it. Cover every original nested cell in both record writing and
   observation before any new GPU dispatch. Then complete the sixty remaining
   vector states, all 61 feature states and the original interventions/inference;
   join full outcomes to the completed weight measurements. No recovery is approved
   by an automatic continuation, and no frozen numerical source may be changed.
3. Complete the separately gated crossed continuation: two fixed source states
   at step 2345 × two reset operators × three order seeds, all twelve 50K runs,
   five stages each and 168 final full-corpus retrieval units. The current
   [fresh-worker component](reports/engineering-archive/dense-v3-factorial-worker-v1/README.md)
   connects the unchanged factory, full 391-step Trainer, five-stage finalization
   and deep native run reader. Its 57 new / 194 combined checks are bounded CPU
   source/refusal tests and explicitly mocked orchestration, not GPU calibration,
   default-topology GPU admission or a formal run. The 69 parent files remain
   unchanged. No admitted launcher, resource handoff or implicit resume is supplied.
4. Reconstruct the complete scientific outputs from their original artifacts,
   generate the paper's results and figures, then pass the full publication,
   portability, clean-source and authorized release requirements. Do not hand-edit
   generated findings, remove pending markers or replace failed historical hashes.

Keep all rates, stages, tasks, original features and frozen inference rules. A
null or inconclusive result remains reportable; do not select a preferred story.
No implementation/debugging narrative belongs anywhere in the manuscript.

## Recovery and handoff

Use [checkpoint-restoration.md](docs/checkpoint-restoration.md), the authenticated
60-entry index and the standalone recovery script for a new machine. They avoid
the moving HF default branch and the historical checkpoint namespace. File
verification does not deserialize models or prove whole-run/bitwise resumption.
Complete analysis also needs the exact code, data/probes and unfinished outcomes.

The [completed recovery evidence](reports/engineering-archive/dense-v3-recovery-entry-v1/README.md)
includes all-60 anonymous remote metadata verification and one real 20-file,
1.35-GB checkpoint download with independent offline file/seal verification.
It does not claim a second-host experiment or complete source publication.

The complete native weight measurements also have a separately verified
[public snapshot and restoration guide](docs/weight-analysis-restoration.md):
all original spectra, bases, tables, map arrays and figures, at dataset revision
`209b4517e64ac5373db47b04ada39104e9080016`. All 298 files were actually recovered
anonymously and independently verified offline. The
[original backup evidence](reports/engineering-archive/dense-v3-weight-artifact-backup-v1/README.md)
preserves the scope, original metadata and each actual execution record.

The complete [training-observation recovery guide](docs/training-analysis-restoration.md)
now addresses 149 immutable public files at dataset revision
`3c95da08a4c817d5bcd58b5c84df777716a02ca9`. Full anonymous recovery, independent
offline hashing and numerical replay from the recovered inputs pass. The original
upload process nevertheless exited 1 at its unchanged-root check: two literal
new-file LFS rules were appended to `.gitattributes`. A separate read-only check
verifies the exact additions, unchanged old rules/payloads/card/weight subtree,
and every new payload. The original root-byte guard stays failed; no receipt was
forged and no second remote write was made. Read the
[complete evidence and caveat](reports/engineering-archive/dense-v3-training-artifact-backup-v1/README.md).
No program source was published, and this is not a physical second-host experiment.

The [endpoint/validation recovery guide](docs/evaluation-analysis-restoration.md)
now addresses 660 immutable public files at dataset revision
`3883b677f87b1982f06016e9fadb8bb95e0cfc96`. Full anonymous download completed at
17:28:18 UTC and independent recovered-input numerical replay passed at 17:37:09.
The new addition preserved all twenty other root entries and both earlier corrected
subtrees; its strict unchanged-root audit passed without overwriting/deleting any
old payload. This does not erase the earlier training-backup caveat. Read the
[actual backup/replay evidence](reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/README.md).
BEIR task aggregates cannot reconstruct per-query rankings, and the dataset does
not publish executable source or unfinished intermediate/functional outcomes.

The [endpoint-statistics recovery guide](docs/endpoint-statistics-restoration.md)
adds all six interval comparisons, their task tables and PDF/PNG/SVG figures at
revision `13b110a0d067948ff518a0f720ee84c2e2f8c6b1`. All sixteen files / 218,466
bytes passed actual anonymous recovery and an independent standard-library
check of forty CSV rows, 246 numeric table fields, 24 figure fields and all six
interval mappings. The original root entries and three earlier corrected
subtrees stayed unchanged. The first metadata-cache symlink refusal occurred
before any upload and is retained with its exact source; the successful new
attempt materialized ordinary metadata files without weakening the hash guard.
Read the [complete archive](reports/engineering-archive/dense-v3-endpoint-statistics-backup-v1/README.md).
The original endpoint readout and its historical handoff binding are preserved.

The [first-stage evaluation recovery guide](docs/first-stage-evaluation-restoration.md)
now adds 557 immutable files / 1,957,694 bytes at revision
`8555e5849b56862948b0fc0688085708ce3e67cf`. Complete anonymous download passed
at **2026-09-11 04:18:08 UTC**, and independent isolated recovered-score replay
passed at **04:18:43 UTC**: all 168 raw task scores, 168 original exit-zero workers,
twelve means and 180 CSV rows match. All twenty other root entries and four
preceding corrected subtrees are unchanged; no old file was overwritten/deleted,
no new root attributes were written and no executable source was published.
The new local adapters' original alias failures remain in the
[backup evidence](reports/engineering-archive/dense-v3-first-stage-artifact-backup-v1/README.md).
Only the first-stage cohort is newly backed up; all middle-stage evaluations,
functional results and complete source/scientific release remain required.

The current local WIP has **not** been published as a complete GitHub release.
Do not retry the recorded GitHub 403 through another identity. Historical HF
withdrawal was rejected: do not retry, split or bypass it. Never touch `gpu.py`
or its processes; no broad process/GPU-process inspection. Preserve the exact
old stopped controller chain and every original/failed artifact. Detailed
identities, source scopes and remaining authority are in [AGENTS.md](AGENTS.md).

The former default recovery guide and README are preserved in the
[before-copy](reports/engineering-archive/dense-v3-recovery-entry-v1/before/).
They are historical evidence, not instructions to execute the active experiment.
