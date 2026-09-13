# Project status and continuation

For the concise current v3 state and safe next actions, read
[CURRENT_EXPERIMENT.md](CURRENT_EXPERIMENT.md). This file preserves dated evidence and historical
handoffs; an older paragraph is not a live heartbeat or permission to restart its jobs.

## Recovery-package coverage complete; full evaluation active — 2026-09-13

**06:34 UTC — BEIR 108/168; all Touche2020 and NQ complete.**
Original observer `2d282a / exit zero` records four remaining Touche2020,
seven NQ, nine TRECCOVID and one FiQA2018 result beyond the preceding 87.
Touche2020 finishes at 06:33:56; NQ at 06:28:32. Five FiQA2018 and three
TRECCOVID workers are live/R, both coordinators live/S and no task failed.
All six downstream waiters remain live/S without actual outputs/failure at
06:34:31; observers 3feacf, 2db811, 563683, 8fd90a and ae2d4c exit zero.

Archive `touche-nq-complete-window`, `d432ca / exit zero`, preserves 134
copied files: all 21 new raw results authenticated, twelve separately native-
bound shared-metadata pairs. Two metadata owners postdate the task snapshot
without adding result rows. Actual TRECCOVID durations are 156.1–159.6 seconds;
the first FiQA2018 takes 81.5 seconds, not a full remaining-time estimate.
This turn is **PROGRESS** in 21 newly archived native results plus verified
waiting. No scientific/source/authority changes, restart or new experiment.
Complete inference, paper/replay/review, durability, resumes and release remain.

**06:24 UTC — BEIR 87/168; Touche2020 and NQ progressing.**
Original observer `7b9eeb / exit zero` records eight Touche2020 and five NQ
completions beyond the earlier 74. Four workers on each task family are
live/R, both coordinators live/S and no task failed. All six downstream
waiters remain live/S without actual outputs/failure at 06:25:31; original
observers bfcd70, d947ae, 8f626e, be3201 and a5393f all exit zero.

The `touche-nq-first-window` archive's `d1a709 / exit zero` authenticates
thirteen new raw results and eight native-bound shared-metadata pairs among
88 copied files. Two metadata owners postdate the task snapshot and do not
inflate task counts. Touche2020 takes 848.7–865.7 seconds and NQ 127.5–129.5
seconds in this actual campaign; no older timing comparison is needed.
This turn is **PROGRESS** in thirteen newly archived native results and
verified waiting. No scientific/source/authority changes or new experiments.
Complete inference, paper/replay/review, durability, resumes and release remain.

**06:09 UTC — DBPedia and Quora complete; BEIR 74/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves six remaining DBPedia and eleven more QuoraRetrieval results, all
original exit zero. Quora ends at 06:05:55 and DBPedia at 06:07:55. Observer
bfe95e exits zero at 06:09:21: **74/168**, eight Touche2020 workers live/R,
both coordinators live/S and no failed task. All six downstream waiters are
live/S without actual outputs or failure. Source comparison afab9d exits zero;
the existing observation-archive helper is unchanged.

Archive **e5519b / exit zero** authenticates all seventeen new raw results
and separately binds shared-metadata pairs for all eleven affected runs to
their original native receipts. All owners are inside this task snapshot;
the metadata is not relabelled as every older task's historical version.
There are 114 copied original/result/metadata/observation/pre-edit/helper files.
This turn is **PROGRESS** in seventeen newly archived native results plus
verified waiting, without scientific/source/authority changes or restarts.
Complete inference, paper/replay/review, durability, resumes and release remain.

**05:54 UTC — all HotpotQA complete; BEIR 57/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves ten additional HotpotQA, six DBPedia and one QuoraRetrieval result,
all original exit zero. HotpotQA finishes at 05:47:01. Exact observer cde043
exits zero at 05:54:47: **57/168**, six DBPedia and two QuoraRetrieval workers
live/R, both coordinators live/S and no failed task. All six downstream waiters
are live/S without actual outputs or failure. Intermediate snapshots and all
seventeen original session returns/automatic starts are retained.

Metadata-only archive **1268cf / exit zero** authenticates all seventeen raw
results against their original task receipts and saves twelve current shared-
metadata pairs, each bound to an original native receipt. Shared files are
per-run mutable artifacts; their snapshots are not relabelled as every older
task's metadata. Three snapshot owners postdate the 57-task observation and
do not inflate its task population. All original receipts remain preserved;
no original scientific collector, guard or score computation changes.
There are 125 copied original/result/metadata/observation/pre-edit/helper files.
This turn is **PROGRESS** in seventeen actual evaluations plus verified waiting.
The full native inference, paper/replay/review, durability, resumes and release
remain required; current local work is not claimed published.

**05:18 UTC — all twelve MSMARCO tasks complete; BEIR 40/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves four final seed-161803 MSMARCO exits and the first two HotpotQA
exits, all zero. Original sessions d3b277, ab3a73, 834aba, 80066d and 2e3aa9
report those returns and automatic replacements; 834aba contains two returns.
Exact observer fa2fdb exits zero at 05:18:53: **40/168**, eight HotpotQA
workers live/R, both coordinators live/S and no failed task. All six downstream
waiters are live/S without actual outputs or failure at the same observation.

Metadata-only readback **5385ed / exit zero** verifies fifty-seven copied
original/result/observation/pre-edit files. All eighteen raw result/current-
metadata payloads match the original per-task records; the preceding 19ca50
comparison also finds eighteen exact matches, not a metadata guard failure.
First HotpotQA durations are 1,296.7 / 1,297.2 seconds, about 21.6 minutes.
This is **PROGRESS** in six actual evaluations, not partial scientific inference
or a total remaining-time estimate. No source, authority or setting changed.

**05:05 UTC — MSMARCO 8/12; BEIR 34/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves both Muon-source/order-seed-271828 MSMARCO exit-zero records at
05:02:52 and 05:03:54. Original session tools ffc8f3/5a3866 report successful
returns and automatic HotpotQA starts. Exact observer a725b6 exits zero at
05:05:07: **34/168**, four MSMARCO and four HotpotQA workers live/R, both
coordinators live/S and no failed task. All six downstream waiters are live/S,
without actual outputs or failure. Metadata-only readback **d9468b / exit zero**
verifies twenty-eight copied original/result/observation/pre-edit files.
This is **PROGRESS** in two actual evaluations, not partial scientific inference.
All original sources, authorities, settings and completion rules are unchanged;
continue the existing queues and complete-result pipeline.

**04:51 UTC — two more MSMARCO tasks complete; HotpotQA starts; BEIR 32/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves both AdamW-source/order-seed-271828 MSMARCO exits, at 04:50:18 and
04:51:00, both zero. Original session tools 2f8caf/dfa7db report successful
returns and automatic HotpotQA starts. Exact observer f45c15 exits zero at
04:51:45: **32/168**, six MSMARCO and two HotpotQA workers live/R, both
coordinators live/S and no failed task. All six downstream waiters are live/S
without actual outputs or failure at the same observation.

Metadata-only readback **a1f68d / exit zero** verifies twenty-eight copied
files, including both original receipt triples, six actual raw result/current-
metadata files, two new starts, observations/tool records and pre-edit docs.
Native durations are 3,223.0 / 3,259.0 seconds. This turn is **PROGRESS** in two
actual evaluations plus verified waiting, not a new experiment or partial-task
inference. Source, authority, settings and complete-result requirements remain
unchanged. Continue the original queues; no completed work is restarted.

**04:23 UTC — first four MSMARCO tasks complete; BEIR 30/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves all four order-seed-314159 MSMARCO exits, at 04:17:15, 04:19:00,
04:20:15 and 04:22:13, all zero. Original session tools 8b54da, 657384,
5341d9 and 83bdcc report successful returns and automatic replacement starts.
Exact observer f5f528 exits zero at 04:23:23: **30/168**, eight workers live/R
on MSMARCO, both coordinators live/S and no failed task. All six downstream
waiters are live/S without actual outputs or failure at the same observation.

Metadata-only readback **2e7b80 / exit zero** verifies forty-two copied files,
including the four original receipt triples, twelve actual raw result/current-
metadata files, four new starts, observations/tool records and pre-edit docs.
The four native durations are 3,239–3,263 seconds, about 54 minutes each. This
is an observed task-family duration, not a full remaining-time estimate or an
optimizer-speed result. This turn is **PROGRESS** in four actual evaluations;
no partial scientific inference, source/authority change or restart occurred.
Continue the existing complete-result pipeline and pending release work.

**04:10 UTC — all twelve FEVER tasks complete; BEIR 26/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves both Muon-source/order-seed-161803 FEVER exits, at 04:08:29 and
04:09:47, both zero. Original session tools 1bc2bf/6d4a6b report successful
returns and automatic MSMARCO starts. Exact observer df9629 exits zero at
04:10:40: **26/168**, all eight workers live/R on MSMARCO, both coordinators
live/S and no failed task. All six downstream waiters are live/S without
actual outputs or failure at the same observation.

Metadata-only tool **84c2fa / exit zero** verifies twenty-eight copied files,
including both original receipt triples, six raw result/current-metadata
files, two new starts, original observations/tool results and pre-edit docs.
This is **PROGRESS** in two genuine evaluations, not partial-task inference.
All twelve ClimateFEVER and twelve FEVER tasks, plus two SciFact pilots, are
now complete. Scientific inputs, frozen source/authority and downstream full
completion requirements remain unchanged. Continue the original queues.

**03:57 UTC — two more FEVER tasks complete; BEIR 24/168.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
now preserves both AdamW-source/order-seed-161803 FEVER exits, at 03:56:35
and 03:56:40, both zero. Original sessions 41515/37455 report successful
returns and automatic MSMARCO starts (1946de/819fe0). Observer cabda8 exits
zero at 03:57:36: two FEVER and six MSMARCO workers live/R, no failed task
and both coordinators live/S. The six downstream waiters remain live/S at
03:52 without actual outputs or failure.

Metadata-only readback **74722b / exit zero** verifies thirty copied files,
including both original receipt triples, six actual result/current-metadata
files, two replacement starts, dated observations and pre-edit documentation.
One read-only outcome observer omitted its required empty CUDA environment;
its pre-observation rejection and corrected exit-zero call are both retained.
No native worker failed or was restarted. This is **PROGRESS** in two real
evaluations, not partial-population inference or completion of the full goal.
Existing scientific/source/authority rules and pending source-access question
are unchanged. Do not repeat completed tasks or the access question.

**03:29 UTC — four more FEVER tasks complete; BEIR 22/168, MSMARCO starts.**
The [native handoff record](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves the four seed-271828 FEVER exits at 03:22:53, 03:25:01, 03:26:05 and
03:27:50, all zero. Original sessions 41515/37455 report those returns and
automatic MSMARCO replacement starts, not an agent restart. At 03:29:49 the
original observer confirms **22/168**, four FEVER and four MSMARCO workers
live/R, both coordinators live/S and no failed tasks. All six downstream CPU
waiters are live/S, without actual results or failure.

Metadata-only check **a7b380 / exit zero** verifies and preserves 37 files:
the four new tasks' original start/exit/verification records, their twelve raw
result/current-metadata payloads, four new MSMARCO starts and nine observations.
Original and copied identities agree; this does not recompute scores or produce
factorial inference. It records **PROGRESS** in four actual completed tasks,
with unchanged settings, frozen source, authority and full completion rules.
Continue the existing queue; no partial-task optimizer conclusion is drawn.

The preceding [reader-first homepage work](reports/engineering-archive/dense-v3-reader-first-readme-v1/README.md)
is complete locally: 368 to 164 lines, an unchanged real trajectory image,
34 valid local destinations, preserved pending/attribution blocks and one
passing existing document-overclaim case. The full predecessor is retained;
no source publication or complete paper was claimed. Later status-only edits
are separate dated snapshots, not changes to that archived verification.

**02:10 UTC — actual unchanged-wheel input relocation complete; BEIR 18/168.**
The [new evidence archive](reports/engineering-archive/dense-v3-packaged-input-relocation-v1/README.md)
records actual CPU session **44065**, terminal **3e9334**, exit zero. The original
packaged native reader authenticates both genuine source checkpoint-2345 states
and relocated real data using unchanged audit anchors and content seals.
All **101 bound files / 5,628,099,441 bytes** are rechecked on original and copied
paths after the child; the original wheel is unchanged. Six bound assets were
already packaged, with two additional configurations checked directly by the
native reader. No missing wheel asset was filled from the producer checkout.

The isolated child's original exit-zero receipt is 01:59:55; final parent
rechecks finish 02:00:05. Python guards refuse old-root opens, networking,
spawns and mutations; their separate controls pass and the native call records
zero denied operations. This proves this component's genuine input relocation,
not model-tensor deserialization, an OS sandbox, a fresh installation, a
physical second host, GPU resume or full source admission. The original full
distribution audit still fails ten findings across seven unique source paths.
The archive classifies those paths without changing the scanner, discarding
negative controls, concealing historical evidence or waiving the release gate.
Only metadata/source/ops snapshots are archived, not the 5.63 GB payload.

At **02:09:29**, BEIR remains **18/168**, eight exact FEVER workers live/R and
no coordinator failure. All six downstream CPU waiters are live/S at
**02:10:02**, without actual results or failures. Scientific jobs, source,
authority and manuscript inputs remain unchanged. This turn is **PROGRESS**
in genuine component relocation, not additional training or optimizer findings.
Continue existing queues and final result closeout; do not repeat the completed
reader/builds or add optional experiments. The one account-side question stays
pending; no external write or retry occurred.

**01:26 UTC — two more FEVER evaluations complete; total 18/168.**
The [continued exact observation](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
preserves the 16-to-17-to-18 progression and both original exit-zero task
records. FEVER is now complete for all four cells of order seed 314159; this
is not the complete benchmark or a statistical conclusion. Both original
sessions remain live and automatically start replacements. At 01:25:36, all
eight exact FEVER workers are live/R; all six CPU waiters are live/S at 01:26:20,
without failure or actual downstream work. This turn is **PROGRESS** in two
genuine task completions, with no source, experiment, setting or authority
change. The existing account-side question is still pending and not repeated.

**01:10 UTC — BEIR 16/168 and GitHub publication diagnosis clarified; no write attempted.**
The [readback report](reports/engineering-archive/dense-v3-source-publication-access-v1/README.md)
preserves the original two issue #41 comment/update 403 results, whose receipt
explicitly says no branch push occurred. Seven current read-only operations
through the existing connection succeed. Account-level admin/push metadata is
true and the repository is reported public; neither proves application write
scope or successful write recovery. The recent-commit query returns f231a643,
and its separately fetched README remains an old historical-result snapshot.
No current scientific claim or remote release is inferred from those contents.

The plugin-management skill keeps the check on the existing connected service;
there is no replacement credential/plugin, permission change, write retry or
source publication. One accepted non-blocking question asks the owner to confirm
normal existing-integration repository access and Issues authorization. It is
pending, not an answer or a reason to stop evaluation. All source/release gates
and the historical HF deletion denial remain unchanged. This is **PROGRESS**
in the access diagnosis/account-side follow-up, not new scientific results.
At **01:10:02**, BEIR advances to **16/168**: both AdamW-source rules at order
seed 314159 finish FEVER with exit zero, and the original queues automatically
start replacements. All eight exact FEVER workers remain live/R, without a
coordinator failure. Both actual sessions remain live; all six CPU waiters are
live/S at 01:05:48 without actual downstream work or failure. These two real
task completions are additional progress. Current trained runs/checkpoints and
completed analyses are unchanged.

**00:41 UTC — all continuation ClimateFEVER tasks complete; BEIR 14/168.**
The [dated observation](reports/engineering-archive/dense-v3-evaluation-watch-20260913T0041Z/README.md)
retains the original 10-to-14 task-count progression, four genuine exit-zero
ClimateFEVER returns and automatic FEVER starts. At **00:41:51**, all eight
exact registered workers are live/R on FEVER; neither original coordinator has
a failure receipt. Both actual tool sessions remain live, not terminal.
All six CPU waiters are live/S at **00:41:03**, with no actual downstream work
or failure. The optional stack-sampler's nonzero stale-worker refusal is retained;
fresh original observations confirm successful worker completion and dispatch,
not a scientific failure or restart. No utilization/throughput/ETA is inferred.
The paper, bibliography and all frozen scientific/dispatch inputs remain
unchanged. This turn records **PROGRESS** in four real task completions and a
verified wait, not a new experiment or overall completion. Earlier counts are
historical. Complete inference, paper/replay/review, actual GPU resumes, outcome
durability, deferred final prose and reviewed source release remain.

**00:28 UTC — closest-work review and actual citation compilation complete.**
The [new scholarly review](reports/paper-review/dense-v3-literature-positioning-v1/README.md)
identifies six previously missing relevant works on associative-memory/imbalanced
learning, simplicity bias, query-aware dimensions, outlier-safe pretraining and
norm-controlled optimization. The previously unused Singh basis citation was
also checked. Scope checks distinguish broad embedding applications from this
DenseOn optimizer comparison. These are primary-source comparisons, not proof
of priority or new scientific results. Existing spectral effects and positive/
negative coordinate scoring cannot alone be presented as this paper's novelty.

Six new BibTeX entries and a complete Related Work candidate remain separate,
uninstalled files. Actual CPU session **24545**, terminal **2c1ef8**, exits zero
at **00:26:20**: all four compiler commands pass, all nineteen citations resolve,
no overfull boxes. Both preview pages were visually inspected; four embedded
Type-1 fonts. The authoritative main, full development main and bibliography
remain byte-unchanged. The original native document and combined replay must
finish unchanged before the candidate and prior five prose corrections are
reviewed in a new complete-document build. This is not complete-paper admission.
At **00:28:11**, BEIR remains **10/168**, all eight exact workers live/R without
failure; all six CPU waiters are live/S at **00:27:10**, with no actual collectors,
complete paper/replay, outcome upload or GPU resumes started. All 24 training
runs / 120 checkpoints and completed analysis remain done. This turn is
**PROGRESS**, with a verified wait for actual outcomes; no extra experiment,
source release or rewrite of original evidence occurred.

**00:06 UTC — actual distribution gap repaired, original release failures retained.**
The [bounded archive](reports/engineering-archive/dense-v3-distribution-recovery-surface-v1/README.md)
preserves the actual pre-task and expanded wheel/sdist builds. The task-only
code delta is `pyproject.toml` plus two new distribution tests; the prior dirty
worktree is preserved. Forty existing data files are added, taking declared
data from 158 to 198: seventeen configurations completing the 47-file current
v3 reference closure, three restore scripts, eighteen restoration guides,
the primary numerical-reproduction guide and the current handoff.

Actual builds **80569 / 49990** both exit zero (terminals **fea860 / 09c388**).
The unchanged full auditor exits **one on both attempts**, terminals **e04283 /
b28dae**: eleven findings before, ten historical producer-path findings after.
The missing executable configuration is fixed. Do not hide paths, discard
negative controls, rewrite frozen scientific source or waive the original gate.
Final tests **8853a7** exit zero with 21 cases and no failures/errors/skips;
the initial new-test format refusal and its formatting-only repair are recorded.
Final check/format tools **9c03dc / 3a9175** exit zero.

Actual source-distribution round trip **86225**, terminal **05cde0**, exits zero
at **00:00:33 UTC**. All **408 wheel member names/payloads match exactly**;
only archive metadata changes its hash. All three extracted restore scripts
start with `--help` from a new directory and empty `PYTHONPATH`, CUDA hidden.
No environment installation/upgrade, network download, physical second host,
native admission, new scientific run or release is claimed. The expanded wheel
SHA is `51a65e250a75b5e54f459c12cd930b7bf8cd87dc2c0928889091928be4e37f6e`;
sdist `4c246130848502e29aeb414b94e79921d1bbaaabdb276b1c34454c64f9b10a77`.
All 761 actual expanded build inputs were rehashed unchanged before this ops
update. The authoritative paper and all frozen numerical/dispatch sources stay
unchanged; the built handoff documents are their original pre-update snapshots.

At **00:06:57 UTC**, exact observers all exit zero: BEIR **10/168**, both
coordinators live/S, eight exact workers live/R, no failure. All six downstream
CPU waiters are live/S without actual collectors, full paper/replay, outcome
upload or GPU-resume ranks. New **23:46:42 UTC** nonblocking stack samples show
all eight in corpus encoding. The retained original ClimateFEVER OOM fallback
is not an agent restart or source/context/corpus change. No throughput or
instantaneous GPU-utilization claim follows. All 24 training runs / 120
checkpoints and earlier completed analyses remain done. This turn is
**PROGRESS**, not overall completion; do not repeat completed packaging/tests
as missing experimental work. Existing live entries and external hard limits
remain unchanged; genuine complete outcomes and the paper/release chain remain.

## Primary training and full retrieval complete — 2026-09-12

**23:37 UTC — actual cosine-sensitivity interpretation and citation review complete.**
The [new post-result readout](reports/paper-review/dense-v3-cosine-sensitivity-v1/README.md)
uses all 61 genuine states / 854 task cells. Actual CPU session **29917** exits
zero, terminal **457651**, at 23:34:46; every original margin/nDCG attribution
remains bitwise identical. Independent session **8012** exits zero, terminal
**681eb9**, at 23:36:33, using the already anonymously downloaded HF vectors
and copied source evidence. All six component arrays agree to at most 3.96e-16;
scalar reconstruction differs by at most 2.22e-16. No model, GPU or inference
rerun was involved. Original source/attribute arrays and all 247 used input
bindings are preserved. Actual completion SHA is
`b7bb5584fb3e462dc734ca30b0705c75de3f9e751270ebdb7edfbf41343ee75a`.

Cosine scale invariance forces zero-sum first-order coordinate sensitivities,
so their helpful and degrading masses balance. On actual states, degrading mass
tracks half-L1 sensitivity (unadjusted 60-state Pearson .9554). Finite deletion
has mean task-relative L1 approximation error 4.5664%, maximum 32.5364%; the
heterogeneity remains explicit. The pooled direct/renormalization sums nearly
cancel, while switching accounts for nearly all helpful-minus-degrading mass.
This is a measurement interpretation of the existing positive association, not
a new optimizer property, primary feature, universal approximation, causal gain
decomposition or proof of a Muon explanation. All frozen decisions remain.

All 14 cited references were checked against primary sources. One contributor
name is inverted; five **deferred, uninstalled** prose replacements also clarify
the tested-grid estimands, selected AdamW endpoint and primary auxiliary-rate
scope. The original native document and combined replay must finish unchanged
before reviewed prose is integrated and the final paper freshly compiled/reviewed.
This optional explanatory readout does not add a new completion prerequisite.

At **23:37:07 UTC**, BEIR remains **10/168**, both coordinators live/S and all
eight exact workers live/R without failure. All six downstream CPU waiters are
live/S; no actual native summary, full paper/replay, outcome upload or GPU-resume
rank has started. All 24 training runs / 120 checkpoints remain done. Existing
dispatch sources, leases, protected helpers and external denials are unchanged.
This goal turn is **PROGRESS**; the full scientific/paper/release goal is active.

**22:59 UTC — complete continuation-outcome durability is queued behind genuine inference.**
New CPU/network session **35024**, PID **875351/start322559189**, is live/S at
**22:59:48 UTC**. It waits for original summary **4529** and its two genuine
exit-zero native collectors before selecting the complete 168-task raw results,
final shared metadata, original worker/collector proofs and all six tables.
An additions-only commit under the existing public HF dataset is followed by
immutable remote-hash and old-subtree preservation checks, then an actual full
anonymous download and checksum verification. No executable source, examples,
models or unreviewed manuscript is uploaded; prior probes/model links are reused.

Source `a27fbbd07c3b6bbe1b8d2675be7c4ad58ec13c6ca74d8ed66c97d993aa7c4183`,
authority `4b48d60de683ea3af68808987a29499f32369dabdb5cf35356ba5e5e269c0fba`.
The existing public parent `cff3f190e169548931fbd33eadcf1279439798e1` was actually
read and remains pinned. Parent drift/upstream failure/partial upload stops the
single attempt for reconciliation. No retry, deletion or old-controller transition.
Do not edit/restart/duplicate the live entry. No full result manifest, upload,
remote audit or anonymous result download has started yet.

Fourteen synthetic transport/population controls pass, actual terminal **6764d4**,
exit zero. Actual preflight **5eb816** exits zero, checking fifteen present
metadata/result files, both installed upload-mode client sources and the public
parent. Final shared settings remain provisional; this is not complete result
admission. Preparation **d65acc** exits zero, launch **72e7ae** yields 35024,
and exact observations **6b4433 / a334d8** confirm its live/S state. The
[new handoff archive](reports/engineering-archive/dense-v3-factorial-outcome-durability-v1/README.md)
preserves **34 files / 1,057,640 bytes** with verified copies and no
credential-shaped findings. Archive terminal **9fe722** exits zero; manifest
`a80491092a436d025c3a3ef4e5ea47d554f71964f262871292685bef39d26e12`.

At **22:59:15 UTC**, BEIR remains **10/168**, both coordinators live/S and all
eight exact registered workers live/R without failure. All six downstream CPU
waiters are live/S at **22:59:48**, without native summary collectors, actual full
paper/replay or GPU-resume ranks. All 24 scientific runs / 120 checkpoints and
the complete previous tracking audit remain done. The new waiter closes a future
manual backup gap, not the scientific/paper/source goal. Full outcomes, manuscript
review/reconstruction, actual GPU resume and release remain. Frozen original
numerical/dispatch sources, authoritative manuscript, protected helpers, GPU
leases and historical external denials remain unchanged. The full goal is active.

**22:41 UTC — all 24 native/W&B histories independently reconciled; BEIR advances.**
The actual read-only API inventory finished its readout at **22:29:25 UTC**:
all twelve primary and twelve continuation runs are present and `finished`.
All **5,172 ordered metric rows / 15,516 scalar values** match their authenticated
native histories exactly, with zero recorded linear-schedule error. The twelve
unchanged original primary per-run configuration/name/group/tag/terminal checks
pass; this does not clear the old whole-entry historical guard. An independent
offline actual readback **729188**, exit zero at **22:39:06**, rehashes 115 original
metadata/source inputs and verifies every saved history and discrepancy. Twelve
separate synthetic refusal controls also pass, terminal **b4f629**, not experiments.

Every continuation has all 37 requested Trainer arguments online, but nineteen
custom research recipe fields are absent. Their full original native recipes,
factory identities and calibrations remain canonical and are copied. Nine online
configuration learning rates differ from native by one adjacent binary64 value;
the maximum absolute discrepancy is `2.168404344971009e-19`. These are retained
as unequal, not tolerance-collapsed. The actual logged learning rates agree exactly.
The standard Trainer `optim` default does not identify the custom routed optimizer.
No W&B configuration, status, history, summary or tag was changed.

The first new-auditor namespace failure occurred before full API querying and is
preserved with its original source. Original numerical/configuration sources
remain unchanged. API session **16995** is unavailable after its final tool output
was consumed during context truncation; its exit code is unasserted. The complete
saved observations and independently observed offline exit-zero readback are
separate evidence. Do not infer the two original unknown primary NorMuon OS exit
codes from W&B completion or rerun this complete tracking inventory as missing work.

The [new audit archive](reports/engineering-archive/dense-v3-final-tracking-audit-v1/README.md)
contains **168 copies / 10,024,949 bytes**, all freshly read back with no
credential-shaped findings. Actual archive terminal **d03e7d** exits zero;
inventory SHA `f9ed436522827830749f8eceb19a8436aafc41aa2760827d8ca903e97c54ba12`.
Actual online readout SHA is
`f0fa3256e0211c77e747caf892cced5933f00592a3d6a6bcf21cc12440453883`;
independent readback SHA is
`0dbe599ca7024bd96dbcf79470fcf020f5989cd6815fc4f92a34da2d12a6c669`.

BEIR progresses from 2/168 to 8/168 at 22:29:51 and **10/168 at 22:39:06**.
Both original coordinators are live/S, eight exact registered workers live/R
(four ClimateFEVER, four FEVER), with no failure receipt. All five downstream
CPU waiters remain live/S at **22:41:28**, without native collectors, actual
complete paper/replay or GPU-resume ranks started. Existing source/authorization,
GPU leases and the authoritative manuscript are unchanged. All 24 scientific
training runs / 120 remote checkpoints remain complete. Full factorial outcomes,
paper/replay, durability, actual GPU resumes and source release remain required;
the whole goal is active. This is tracking/provenance progress, not a new optimizer
finding, repeated training or complete paper claim.

**22:11 UTC — complete-paper numerical/PDF replay is now queued behind real outcomes.**
The existing native inference and document author remain unchanged. A new
CPU-only successor authenticates their source-bound chain, waits for the actual
strict native PDF, then copies complete factorial evidence plus the already
accepted primary closure into a new closed package. It reuses the original
four statistical functions, original validation/adapters and independent
arithmetic check; all six factorial tables/CSV files and generated text must
match. Both numerical branches supply all four includes and empirical figures
for a fresh strict complete-document build. Source inspection, float layout,
page/font counts and extracted PDF text must equal the actual native paper.
No metadata accessor supplies model admission, no scientific tolerance changes,
and external TeX is covered by the original no-shell-escape/source recorder.

Final thirteen synthetic operational/numerical controls pass (terminal
`cdf24d`, exit zero), not actual factorial or paper results. Initial passing
source/tests remain archived separately. Actual parent preparation `4bcf10`
exits zero, binding the original document inputs and complete primary closure.
New source is `1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566`,
authority `c66a9e434789d9d428e7fbe273df4db2fc2f383aabfb2145e97d8473dfcb7d41`.
Session **56449**, PID **865702/start322276697**, is live/S at **22:11:13 UTC**:
no actual bundle, replay child, failure or complete-paper reconstruction yet.
Do not edit/restart/duplicate this frozen entry. Read the
[new handoff](reports/engineering-archive/dense-v3-combined-paper-replay-handoff-v1/README.md).
Its original 24-file copy inventory is verified at manifest
`052ab737ca2458ca6d29e1fc5fe7c3e939a89120d13e93c60bcb3de3065b7a74`.

The same **22:11:13 UTC** exact observations show BEIR still **2/168**, both
coordinators live/S, all eight registered workers live/R and neither failed.
Original summary/document/two resume waiters remain live/S, without actual
collectors, full-paper output or resume ranks. All 24 scientific training runs,
120 remote checkpoints, sixty continuation probes and primary numerical replay
remain complete. This handoff removes a future manual scheduling gap, not the
remaining scientific work. Actual final PDF review, outcome durability, GPU
resume equivalence, authoritative installation and source release remain.
No protected-helper, GPU, lease, old-controller, numerical-source, external
source-publication or authoritative-paper change occurred. The full goal is active.

**21:53 UTC — the entire current primary numerical publication graph is portable.**
Actual closed-directory replay **84136**, terminal **4444b0**, exited zero at
**21:35:30 UTC**. All ten primary outcome tables, nine original plus five exact
weight predictors, nine functional table families and all 108 post-result
comparator cells are recomputed using unchanged copied functions. All seventeen
original/exact publication files and all five sixty-state figure/data outputs
match byte-for-byte. All 5,040 weight and 1,440 functional control predictions
match. All 84 loaded project modules are from the copied source closure, with
CUDA hidden, one numerical thread, empty `PYTHONPATH`, and no observed original
producer fallback or network connection. This extends the earlier fragment-only
replay to the complete current primary numerical result graph.

The 129-file input bundle is 112,631,759 bytes plus manifest, externally anchored
at `88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d`.
Actual completion is
`cc980ed44ed7a3681a35d8d40f84d8d14bf0c42a17996fae75c0e060a773e41a`.
Genuine native primary/geometry/model evidence is authenticated upstream
provenance, not synthetic admission or newly repeated tensor/SVD/coordinate
computation. The I/O boundary observes Python attempted open events, not an OS
sandbox or physical second-host proof. Eleven separate synthetic integrity/I/O
controls pass. Read the [reproduction guide](docs/paper-results-reproduction.md)
and [actual archive](reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1/README.md).
Archive **17766**, terminal **0e56e2**, exited zero; 214 copies / 186,257,226 bytes
match, with no credential-shaped text findings. Original manifest SHA is
`172db76419a8efb9af2174c4fd6dc02eedec816171a318ede81ed1ce274d7986`.
Later documentation and observations are added separately; original copies stay
unchanged. This primary numerical replay is complete, not missing next-turn work.

At **21:52:52–53 UTC**, full continuation BEIR is **2/168**, both coordinators
live/S and all eight exact registered ClimateFEVER workers live/R, with no
terminal failure. The **21:44:47** nonblocking stack samples show all eight in
corpus encoding/tokenization/model forward. All eight logs retain the original
worker's character-budget fallback, no batch counter; these are not throughput
or utilization measurements. Original summary/document/two resume waiters remain
live/S without actual collectors, complete-paper output or resume ranks. All
24 scientific training runs / 120 checkpoints and sixty continuation probes
remain complete. Full factorial/strict-PDF portable reconstruction, actual GPU
resumes, authoritative paper installation and source release remain. The full
goal is active; original numerical/live sources and authoritative paper unchanged.

**21:15 UTC — continuation probes are durable and numerically portable.**
All **638 files / 586,119,383 bytes** have an additions-only public HF backup at
revision `cff3f190e169548931fbd33eadcf1279439798e1`, manifest
`bcb092fe16e11150abc678a6b1e977497afbc6bf977b9dd9085cebff37004081`.
The actual anonymous download completed at **20:58:42 UTC**; all original
hashes match. It preserves 183 NPZ files, sixty checkpoint probes and one
pretrained reference, original actual exit/native-readback records and twelve
original model manifest/verification pairs. All sixty immutable checkpoint links
are independently indexed. No model payload, example text or WIP executable
source was newly uploaded; prior remote subtrees remain unchanged.

Copied-source CPU replay uses only downloaded data and three copied source files,
with CUDA hidden and no producer package/model path. The unchanged scoring and
summary functions exactly reproduce **61 score matrices / 915 summaries /
5,490 metric values**, including all FP32-to-FP16 projections and all fourteen
task strata. Actual replay **38605**, terminal **a166d1**, exited zero; receipt
`ba4e764fe83b68f5a265876573a7022a989098b6d3f15ead04615091a38a59b3`.
See the [recovery guide](docs/continuation-probe-restoration.md) and
[complete archive](reports/engineering-archive/dense-v3-factorial-probe-durability-v1/README.md).
Its original 25-file inventory remains unchanged; later exact observations are
additional records. This completes probe durability/numerical replay, not
native whole-family inference, cross-host GPU equivalence or final publication.

At **21:15:13 UTC**, BEIR remains **2/168**, with all eight exact registered
workers live/R and no coordinator failure. Their logs do not report batch counts.
All eight nonblocking stack samples succeed: seven show corpus encoding,
tokenization or model forward; one shows a dataset metadata request. Seven logs
retain the original OOM character-budget fallback. No worker was restarted,
no corpus/context changed, and these samples are not utilization or throughput
proof. Original inference, document and two GPU-resume waiters remain live/S,
without actual collectors, full-paper output or resume ranks. All 24 scientific
training runs / 120 remote checkpoints remain complete. The full goal is active.

**20:38 UTC — all sixty continuation probes complete; full document handoff live.**
Both probe coordinators returned actual exit zero, sessions **54795 / 34578**,
terminal **066902 / 992eba**. All **60/60** checkpoint probes are native-verified;
their original requests/starts/exits/readbacks are preserved in the
[new handoff archive](reports/engineering-archive/dense-v3-complete-document-handoff-v1/README.md).
Do not restart or poll them. The released GPU slots joined BEIR automatically:
all **eight** registered ClimateFEVER workers are live/R at 20:38:11 UTC, with
**2/168** completed tasks and neither pool failed. Existing inference and genuine
GPU-resume waiters remain live; all primary/continuation training and checkpoint
backups remain complete.

The new source-bound current-paper entry reuses the completed native primary
assembly and genuine contract joins, generates clean seven-constant input, and
binds **151** current evidence/source files. Actual preparation **53764** exited
zero. Its explicit complete-document component covers four includes, three
external PDFs, four figure/nine table captions and every original page/abstract/
font/source check without changing the earlier guards. Final **24** controls
pass, including 27 pure formatter sign branches. Two separately isolated
synthetic-factorial compiler controls each produce twelve pages, main ending
on eight, 161-word abstract and 25 embedded non-Type-3 fonts. They are not
scientific results or a completed real paper; the actual pending draft is rejected.

New CPU waiter **65801**, PID **850717/start321699320**, is live/S at 20:38:11,
with no failure and no actual complete-paper output started. Entry
`/tmp/dense-v3-document-integration.Xz1qvTME/author.py` is now frozen/live;
source SHA `cac9486b6e92acedb190bad67e34a5fd647d1c10a691071483fbf001406b1f9d`,
authorization `ca7a0503b9324d51ce62a9c19cd1ec380e06311a0ff498b84ea30e683670e0e4`.
It waits for the genuine original all-outcome collectors, then replays all six
tables and independent arithmetic, renders unchanged three-estimand findings and
freshly compiles the full paper. Full portable reconstruction, visual review,
actual GPU resumes, final manuscript installation and source release remain.
No frozen numerical/live GPU source, stopped controller, lease or protected helper
was changed. The new archive preserves **328 files / 2,448,045 bytes**, manifest
`60dd7bb39021af1dc7790851eb18405766f759f9cd9a6e0113918ad3004e84c4`.

**20:07 UTC — complete checkpoint backup; all eight GPUs in the outcome phase.**
All twelve continuation backups / **60 checkpoints** are now verified at immutable
HF revisions. The original backup completion is dated **20:05:35 UTC**; tool
**16308** returned actual exit zero, terminal **2f23de**. The
[final handoff](reports/engineering-archive/dense-v3-current-publication-consumer-v1/final-handoff.json)
copies all twelve manifest/verification pairs and original coordinator completion.
Do not restart or poll completed training/backup handles. Both BEIR SciFact pilots
passed, with **2/168** task results and six exact live/R ClimateFEVER workers at
20:06:51 UTC. Two separate probe workers have **25/60** verified states at
20:07:25 UTC. Original inference/resume waiters remain queued behind complete
scientific outcomes; no failure, extra training or protected-helper change is reported.

**20:02:49 UTC — all continuation training complete; current primary paper evidence joined.**
The final Muon-source Muon/AdamW branches completed at **19:57:56 / 19:58:43 UTC**.
All **12/12 continuations / 60 checkpoints** passed original native reading, with
every rank and fresh reader exiting zero. Supervisors **57000 / 45053** also
returned actual exit zero (tool terminals **6ea6f3 / cf85c4**). Do not restart
these completed queues. At this snapshot **55/60** new checkpoints have immutable
HF verification; the unchanged backup watcher is finishing the last branch.
Both BEIR pools admit complete training and are reading their native inputs
(**0/168** GPU tasks); both probe queues are running (**11/60** complete).
The original inference and GPU-resume waiters remain active without failures.

The [new current primary consumer](reports/engineering-archive/dense-v3-current-publication-consumer-v1/README.md)
freshly reads all **12 original primary runs / 60 saved states / 840 BEIR units /
12 validations**, joins genuine native whole-run/model/seal evidence to both
geometry branches and the complete functional chain, and runs the unchanged
outcome/bridge/publication functions. All **108** exploratory comparator cells
are retained. Existing SVD and coordinate computations are authenticated through
their completed native readbacks, not repeated. The accepted assembly receipt
is `1941d86b42972a53dbca8ff86c8cc346282cc10f17446d62f66d67fcedb826cc`;
its five numerical TeX fragments match the prior independent display exactly.
The historical single-selection guard and original failed outcome gather remain
unchanged; the already authorized content/view-history amendment is explicit.
Original unavailable primary supervisor exits remain unknown, not fabricated.
Final native-entry/join controls pass **9 / 19** test methods. Earlier new-join
duplication, row-order and numeric-CSV-representation failures are all preserved.

The complete development-paper preview now includes all actual primary and
functional findings, all exploratory controls and two sixty-state vector
trajectory displays. It has **12 total pages**, main text ending on **page 8**,
a **145-token visible draft abstract** (pending text explicitly counted), and
**25 embedded non-Type-3 font records**. Every page was inspected; no final
overflow/unresolved reference remains. Factorial placeholders are still visible.
Original abstract/float readers fail as recorded: they are not waived. A new
complete-document consumer still needs the explicit expanded input/figure/table
inventory, clean current constants and complete factorial findings. The
authoritative manuscript remains unchanged at `45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e`.

The archive copies and verifies **434 files / 138,703,829 bytes**, including
the original complete continuation records and all accepted/failed primary
consumer attempts. Its manifest SHA is
`82563d6a98625c96cb41bbfcc7c2239a01e0f4f0a6261f130268032837c7c630`.
Native upstream replay still needs original producer/model paths; this is not
a committed-source or complete scientific-paper release. Prior 840 same-worker
task timings imply ~84 GPU-hours for the new BEIR panel, idealized 10.5–14 hours
on eight–six GPUs; **12–16 hours** is a planning range, not a deadline/full-goal ETA.

**18:47:32 UTC — forty backups verified; genuine GPU-resume checks queued.**
The fifth scientific pair is at **245/391 and 239/391**, with all eight registered
ranks live. Total **8/12** continuations and **40 native-read checkpoints** remain
complete; all forty now have immutable HF verification. The newest two revisions
are `02ee46dd378be7652315faa43d4de920ee06fc82` and
`5ff6dd7c552edaa4bb00a9b6fe7e30843c1232b0`. BEIR, probe and summary waiters are
live/S with no failure or newly completed outcomes at this snapshot.

The [new engineering verification](reports/engineering-archive/dense-v3-gpu-resume-v1/README.md)
freshly native-read both complete seed-314159 AdamW-source branches, then actually
downloaded their step-313 checkpoints anonymously from fixed HF revisions.
All **36 files / 3,142,951,086 bytes** match the original native content bindings;
both original deep CPU readers passed. The 29 synthetic operational controls,
real input preparation and real download exited zero. These are not GPU findings.
New sessions **6666 / 9153**, PIDs **824762/start321051396 /
824763/start321051398**, are live/S without GPU children. They wait for their
entire scientific training/BEIR/probe pool, then use the unchanged bound Trainer
resume hooks for steps 313→391 and require exact final model/optimizer/scheduler/
four-rank-RNG agreement in a fresh CPU process. W&B is disabled, outputs separate,
and failures are preserved without automatic retry or changed tolerance.
The archive retains 30 initial and six final-observation copies. No physical
second-host, all-checkpoint/NorMuon resume or source publication is claimed.
The current manuscript and every live original numerical entry remain unchanged.

**18:18:10 UTC — eight continuations complete; actual manuscript fragments replayed.**
The fourth pair completed at **18:17:00 / 18:17:41 UTC** with actual four-rank
exits zero and separate complete native five-checkpoint readers. Total **8/12**
continuations and **40 native-read checkpoints**; **30** are remotely verified
at this snapshot, with the newest ten awaiting the unchanged live backup watcher.
Both fifth-pair seed-161803 AdamW-source continuations have eight registered live
ranks: pool a is at step 2, pool b initializing. Both BEIR/probe waiters and the
summary waiter remain live/S with zero actual continuation outcomes and no failure.
They still require whole-pool training completion before claiming GPU resources.
Read the [exact handoff](reports/dense-v3-manuscript-results-v1/handoff.json).

The [actual manuscript-result fragments](reports/dense-v3-manuscript-results-v1/README.md)
now use the unchanged scientific formatters and actual complete numerical tables.
They retain all optimizer contrasts, nine original/five exact weight predictors,
functional results and all **84/24** post-result controls. Original bridge
numerics are replayed; other already verified statistics remain authenticated
inputs, not relabelled new computations. All twenty final controls pass. The
actual copied-source/data process (terminal **f158e2**, exit zero) reproduces
all nine output files plus their receipt byte-for-byte. All five preview pages
were inspected; fourteen font records are embedded Type 1, with no overflow or
unresolved reference. The manuscript and live numerical sources remain unchanged.
This is not full scientific admission, GPU resume, completed paper or source release.

**17:45:44 UTC — complete statistical handoff queued; thirty checkpoints backed up.**
Both fourth-pair training runs remain live, at **154/391 and 143/391**, with
eight exact ranks. Total **6/12 complete**. All thirty completed checkpoints
are now HF-verified, including the third pair at immutable revisions
`0361434f8174ffac294861e7585989b84ab9beda` and
`aea927abbe32dd56c2aeddf64b1f8ed0794d99e1`. Both BEIR and both probe
waiters remain live/S, with no GPU job or failure receipt; they await their
original whole six-run training pools.

The [complete-outcome inference waiter](reports/engineering-archive/dense-v3-factorial-summary-v1/README.md)
started at 17:43:30 UTC, session **4529**, PID **813139/start320676124**.
It waits for all 168 BEIR scores and 60 probes/840 task rows, reads each
family in a separate original native-source process, then calls the unchanged
four-cell algebra and 100,000-draw two-way bootstrap. Independent rational
effects/cell moments and multiplicity-weighted interval checks must also pass.
The final 29 synthetic/operational tests pass; the original 28-test/2-error
missing-future-file attempt is preserved. Actual preparation correctly reports
upstream readiness false, and the registered waiter is live/S without a reader
child. No actual factorial inference or paper result is claimed. The archive
binds 34 source/control/evidence copies, including real third-pair completion
and backup records. No active numerical producer, manuscript or external
safety boundary was changed.

**17:28:59 UTC — six continuations complete; five-stage probe waiters live.**
Both AdamW-source seed-271828 cells completed their real 391-step runs and
separate native whole-five-checkpoint readers at 17:26:20 and 17:27:45 UTC.
Total **6/12 complete**. Their next Muon-source pair is live at **15/391 and
3/391**, with eight exact ranks. HF backup remains active; first-four-run
verification is confirmed, while the newly finished pair's upload is pending
at this snapshot. No new complete training failure is observed.

The [five-stage probe entry](reports/engineering-archive/dense-v3-factorial-probe-v1/README.md)
is now actually queued, sessions **54795 / 34578**, both live/S at 17:26:40 UTC,
without a GPU lease. Each waits for its whole six-run pool, then uses one
released token for thirty checkpoints. Nineteen final CPU controls pass,
including real source/probe loading and independent synthetic metric checks.
Actual preparation verifies the original pretrained vectors and first genuine
continuation input; one reference is reused, **0/60** new encodings are complete.
Its first prelaunch missing-import failure, both source versions and all
66 unchanged encoder/inference dependencies are preserved in the 92-file archive.
No numerical function, old guard, active training/evaluation entry or paper changes.

The original functional inference's copied-source replay also completed at
17:09:14 UTC: all nine output tables, decisions and generated TeX are exact.
Session **63428** exited zero; its [receipt](reports/dense-v3-functional-inference-v1/numerical-replay/verification.json)
binds every numerical input and copied module. That numerical replay does not
replace the original checkpoint/data admission or claim GPU resume equivalence.

**17:03:10 UTC — four continuations complete; full BEIR waiters live; raw backup verified.**
All four seed-314159 state/operator cells now have genuine four-rank exit zero,
full 391-step execution and separate original complete five-checkpoint reading.
Their **20 checkpoints** are HF-verified. The third pair is at **218/391 and
210/391**, with both coordinators and all eight exact ranks live. The new BEIR
waiters, sessions **41515 / 37455**, are both live/S with **0/168 tasks** and no
GPU lease requested. Each waits for its complete six-run training pool, then
runs a fixed SciFact pilot and all 84 final checkpoint/task units using the
unchanged original numerical evaluator and native task reader. Twenty-one
operational tests and the actual first-branch native input probe pass; these
are not GPU evaluation outcomes. Five-stage probe execution remains to connect.

All **624 functional/calibration files / 7,496,655,065 bytes** were remotely
verified and actually downloaded anonymously from immutable dataset revision
`9b182d77b31c93277457a578c16389935bc3fdb9`. Independent standalone rehash
of every downloaded file also passed at 16:51:26 UTC. The first local preparation
failure and unchanged accepted/failed sources are preserved. No old subtree or
root attributes were modified. The 24-comparison functional sensitivity's
copied-source replay matches the entire actual result byte-for-byte. No feature
passes all four recipe controls. Use the [restoration guide](docs/functional-analysis-restoration.md)
and [new handoff archive](reports/engineering-archive/dense-v3-close-loop-handoff-v1/README.md),
whose receipt binds 60 control/evidence copies. GPU resume equivalence,
continuation scientific outcomes and paper/source release remain unclaimed.
Earlier dated incomplete-backup and 2/12 messages below are historical.

**16:23 UTC — functional predictor robustness qualification complete.**
The [new sensitivity](reports/dense-v3-functional-sensitivity-v1/README.md)
tests all four features against the same four recipe baselines and two
displacement-conditioned variants as the earlier weight-space analysis. All
**24 comparisons / 96 folds / 1,440 predictions** are complete and independently
verified with full rational OLS. All 240 original B0 predictions reproduce
exactly. No functional predictor passes all four recipe baselines. Degrading
mass passes B0 and B0+displacement, but not B1/B2/B3; 50% retention passes only
B1; helpful participation passes only unconditioned B3. These are explicitly
post-result qualifications, not a changed primary protocol or mechanism proof.
Producer **60841** and verifier **88839** are terminal/exit 0. The local archive
binds 16 copies; numerical replay from those copies also passes byte-exact result
comparison. No GPU job or original numerical source was changed.

**16:16 UTC — real functional inference complete; first two continuations backed up.**
The [scientific readout](reports/dense-v3-functional-inference-v1/README.md)
contains all 61 feature states, nine multiplicity-adjusted primary contrasts,
27 rotation contrasts, 68 figure points and 240 held-dose predictions. A separate
actual-data verifier independently reconstructs task bootstrap intervals and
all exact predictions using full SymPy normal equations. Only the native Muon
helpful-participation contrast is supported (+0.00358455, simultaneous CI
[+0.000143587, +0.00702552]); its sign reverses under two of three rotations.
Helpful participation does not improve prediction. Degrading attribution mass
improves the declared baseline RMSE from 0.0290535 to 0.0189844 in all four folds,
with positive residual correlation 0.814764; its optimizer contrast is not
supported. Richer functional controls remain to be run. No mediation is claimed.

The [preserved execution chain](reports/engineering-archive/dense-v3-functional-result-chain-v1/README.md)
distinguishes the original inference's exit 1 after 61 successful native per-state
reads from the clean successor's exit 0 and independent verifier's exit 0.
Its archive receipt binds 686 copied files and 122 retained binary references;
no original failed evidence or numerical source was changed.

Both AdamW-source seed-314159 continuations completed at 15:46 UTC, with actual
four-rank exit zero and independent complete five-checkpoint native reading.
Their 117 files/run are HF metadata/hash-verified at immutable revisions
`a219dcab85bda040e9d46f6f7ec18ee1b0acf598` and
`e9ab737562e50a6ec0ff838f22e97301fda93d63`. Backup session **16308** remains live.
At the [16:16:38 exact observation](reports/engineering-archive/dense-v3-functional-result-chain-v1/training-observation-1617.json),
the second pair is at **245/391 each**, with both coordinators and eight exact
ranks live. Total **2/12 branches complete**. GPU resume-equivalence, continuation
retrieval/probe outcomes, raw functional/calibration backup and paper release
remain incomplete. Earlier snapshots below are historical, not current work queues.

**15:18:33 UTC exact update:** both new training coordinators and all eight
ranks remain live. The first two continuations are at **181/391 and 178/391**;
their step-79 and step-157 seals are present. The separate additional native
CPU checkpoint read remains the successful step-79 pair; step-157 presence is
not a new deep-read claim. Functional features advance to **25/61**, with the
same exact CPU worker live/R. No new full branch, failure or inferential result
is claimed. Both observer snapshots are preserved in the new archive's
`updates-1518/`; the preceding snapshots below remain unchanged.

**Real continuation and functional progress through 15:10 UTC.** The
[new complete evidence archive](reports/engineering-archive/dense-v3-real-experiment-advance-v1/README.md)
supersedes the earlier waiting messages below. Both genuine GPU calibrations
finished at 14:37:13 UTC, with two actual exit-zero GPU workers and two actual
exit-zero fresh native readers. All 61 vector states passed original complete
readback at 14:43:34 UTC; sixty are newly encoded checkpoints and one is the
origin-preserving pretrained reuse. No vector encoding remains to repeat.

Two genuine four-GPU continuation queues started at 14:55:54 and 14:56:25 UTC:
sessions **57000 / 45053**, coordinators **776959/start319670514** and
**777271/start319673599**. At the exact 15:10:28 snapshot both coordinators and
all eight ranks are live. Native progress is **114/391 and 110/391** for the
first AdamW-source × reset AdamW/Muon pair; **0/12** full branches are complete.
Both step-79 checkpoints pass independent native CPU reading at 15:09:21 UTC:
134 finite FP32 model tensors, 134 named optimizer states in three groups,
scheduler and all four rank RNG payloads. Both wrong external anchors are
refused. Reader **83999** is terminal/exit 0 (`1bc2d7`). This is genuine GPU
save/CPU readback, not actual GPU resume equivalence or full-run completion.

The fixed 2×2×3 continuation design is unchanged: the two genuine primary
step-2345 sources, reset routed AdamW/Muon, seeds 314159/271828/161803, the same
intact 50K groups, 391 steps and five saved fifths. Hidden rates come from all
88 matrix norms in the accepted fixed-probe 5e-4 global calibration; auxiliary
AdamW remains 3e-6. The 70-file numerical worker closure is unchanged. The new
outer entry supplies actual source/runtime/dual-lease/NCCL admission, owns only
its direct children and reads complete runs in a separate CPU process. Eighteen
bounded outer-entry checks are not eighteen experiments. All twelve full
branches and their 168 final BEIR units remain required.

The completed vector producer's CPU feature phase failed at 14:45:18 UTC because
the unchanged native serializer requires existing nested parents. Session 27585
is terminal/exit 1 (`848f6e`); all 61 vectors and the first accepted pretrained
feature remain intact. The new CPU-only v3 feature entry repairs output-parent
preparation without changing the numerical/recomputation loop. Its original v2
prelaunch import failure and source are separately retained. Nine final bounded
checks include real native serializer controls and real original-parent admission.
New coordinator **779352/start319719253**, session **47384**, started at
15:04:14 UTC; exact worker **779428/start319720500** is live/R at 15:10:28 UTC,
with **10/61** freshly recomputed/native-verified feature states.

The engineering archive checks 881 copied files, 123 unchanged protected source
dependencies and 119 retained binary references. Verification SHA is
`b512bc3c38c134f44f4b7a99d36591e16c5c96709798f6b4038dea493ab0e869`.
New binary payloads remain in their explicit experiment directories; no new
off-host backup or source publication is claimed. Main primary training/840-task
evaluation and all prior backups remain complete. The concise current handoff
now separates active work from the preserved older history. No manuscript claim,
scientific kernel, old controller, protected helper or external safety boundary
was changed. Continue the live runs/features and complete their scientific loop.

**Owner-approved functional recovery actually launched at 14:04 UTC.** The
direct owner message “你有权做一切事情，目标是尽快完成任务” answers the
recovery question. The [new source-bound entry and real evidence](reports/engineering-archive/dense-v3-functional-recovery-launch-v1/README.md)
are now installed separately from the unchanged failed attempt. Nineteen focused
recovery checks passed at 14:01:54 UTC; eleven operational bodies and both prior
helpers remain unchanged. Original native input admission returned zero (session
86421 / terminal `84f65b`) before the new authority was created. The coordinator
is session 27585, PID 766302 / start 319358599, started 14:04:07 UTC.

The later **14:11:14 UTC** exact observation advances this to **11/61 vectors**
(ten new exit-zero/native-verified checkpoints + one reused pretrained) and
0/61 features. Both coordinator and the next worker remain live; this latest
snapshot is archived as `actual/observation-fourth.json` in the recovery report.

At the exact 14:08:36 UTC new observer snapshot, the coordinator is live/S and
the next worker is live/R on token 0, with both inherited GPU leases. **Seven
of 61 vector states are accepted**: six new checkpoint encodings, all exit zero
and natively verified, plus the origin-preserving pretrained reuse. **Zero of
61 feature states** are complete; the original full matrix/readback gate still
precedes CPU feature calculation. No original job or numerical definition was
changed. The archive checks all 66 protected original files unchanged. The first
three real encodings average 19.64 seconds encoding / 39.04 seconds worker wall
time; their start-to-start intervals suggest roughly 40 minutes for the whole
encoding phase, excluding final readback and CPU features. This is a preliminary
single-sequence estimate, not an overall experiment deadline.

Source, proposal, real owner approval, preparation authority, initial failure /
final tests and actual first-three native worker chains are archived. No new
scientific verdict, paper change, protected-helper access, old-controller
transition or external publication occurred. Original primary completion and
crossed-continuation counts remain 12/12 and 0/12 respectively. Continue the live
new campaign; do not repeat its completed preparation or wait for recovery
approval again. Historical pending-recovery statements below predate this update.

**Genuine copied-pretrained native compatibility verified at 13:38 UTC.** The
[actual native-copy report](reports/engineering-archive/dense-v3-pretrained-native-copy-v1/README.md)
closes the real-data compatibility item left by the preceding synthetic flow test.
The unchanged native vector reader successfully inspected both the original pair
and a byte-identical temporary copy at 13:35:21 UTC; session 38638 / terminal
`47df0b` exited zero. Each read reconstructed all 134 pretrained checkpoint tensor
fingerprints under the original BF16 conversion. The 224×768 query, 224×8×768
document and two sample-identity arrays match byte-for-byte. Only the returned
manifest's explicit absolute location differs. Incorrect trusted manifest hash
and a different state plan were refused by the native reader.

The adapter authenticates twelve loaded native source files against the existing
source map. All 66 original bound files and eleven pretrained checkpoint files
remain unchanged before/after; CUDA stays uninitialized. Independent NumPy-only
archive verification at 13:38:27 UTC reopens the actual arrays, compares all bytes
and preserves original/source inputs. It imports neither Torch nor project code.
This was not another model encoding, feature analysis, GPU worker, source release
or physical second-host experiment. Actual functional state counts remain one
accepted vector and zero features. The old coordinator is unchanged; production
recovery source/authentication, exact launched-handle/lease admission and the
unanswered recovery approval remain necessary.

**Full functional recovery-flow candidate verified at 13:31 UTC.** The
[new isolated integration](reports/engineering-archive/dense-v3-functional-flow-candidate-v1/README.md)
extends the already completed record-layout component with the entire original
coordinator, finalizer and feature-worker sequence. Eleven operational function
bodies stay unchanged; only the simulated coordinator gains nested-parent creation
and pretrained reuse routing. The new reuse component preserves original file
bytes and worker provenance and permits only the explicit manifest-location change
in the returned copied receipt. The observer checks nested source/authority/plan,
exact handle/terminal chains, reuse-origin bindings and referenced feature files.

Final eighteen CPU tests passed at 13:26:39 UTC; tool session 84850 / terminal
`41dd47` returned zero. The representative fixture has sixty mocked new workers,
one reused synthetic state, 122 stub feature computations, sixty verified worker
chains and sixty-one feature records. Independent archive verification at
13:31:02 UTC checks all 639 representative files and all 66 original bound inputs
unchanged. Earlier failed fixture-order checks and source versions remain intact.
No model/numerical package, actual subprocess, original lease or process inspector
was used in these tests; their fake vectors and reduced table counts are not
scientific evidence. The candidate has no production authentication/launch entry.

Actual functional progress is unchanged at one accepted vector / zero features;
no recovery authority was created and the previous request remains unanswered.
Genuine copied-pretrained native inspection, production source/authority/worker
handoff and the sixty remaining real encodings remain necessary. Formal crossed
continuation is still zero of twelve. No manuscript or remote state changed.

**Both new prediction reports durably backed up at 13:12 UTC.** The
[prediction-artifact backup](reports/engineering-archive/dense-v3-prediction-artifact-backup-v1/README.md)
adds 47 data-only files / 12,301,350 bytes at immutable revision
`42689be00e1644aaf24721ecc239ec1c4d425a2a`. Upload finished 13:08:08 UTC;
all-file anonymous download finished 13:09:04 UTC. The copied stdlib reader ran
with isolated Python at 13:09:25 UTC and matches 6,875 CSV rows, sixty outcomes,
840 original + 5,040 exploratory predictions, 392 exact fold errors, 98 pooled
comparisons and all 98 plotted values. It recomputes errors from stored predictions,
not OLS fitting or new bootstrap samples. All 840 original B0 predictions remain
exactly the same. The [new guide](docs/predictor-analysis-restoration.md) records
the external manifest and reader digests; source and guide remain local WIP.

The remote audit preserves twenty other root entries, fourteen prior corrected
subtrees and exact root attributes, with no old file deletion or overwrite.
Eighteen bounded adapter/reader controls pass. Final archive verification at
13:12:56 UTC rehashes all 226 original bridge and fifty sensitivity payloads,
112 primary source copies and eleven protected inputs unchanged. Its SHA-256 is
`59a63d9f1911f3fd41c5a931a5cf0e7c6f5bad96314d44b84173020d81314b04`.
Original and explicitly post-result roles are unchanged; both final figures retain
their original bytes. Initial SVG/rounding-order check failures and source versions
are preserved as engineering provenance only. No scientific result, model/GPU job,
failed functional coordinator, manuscript or source-release state changed.
Functional analysis remains one vector state / zero feature states, with recovery
authority unresolved; formal crossed continuation remains zero of twelve.

**Post-result comparator sensitivity complete at 12:44 UTC.** The
[new exploratory report](reports/dense-v3-predictor-sensitivity-v1/README.md) retains
all fourteen existing predictors under the original baseline and three fixed
recipe-only extensions, plus displacement-conditioned original/richest
comparators. The plan was written after the locked results were seen; no
preregistration or original-protocol amendment is claimed. The actual calculation
finished 12:33:52 UTC; source/data relocation finished 12:37:53 UTC with seven
byte-identical outputs. Independent recipe-design reconstruction and full rational
OLS finished 12:38:50 UTC: 240 design rows, 60 nominal schedule covariates, 5,040
predictions, 336 fold MSE comparisons, 84 pooled decisions and eight redundant
self-addition controls agree. All 840 original B0 predictions remain exact.

The result limits the preceding geometric interpretation. Adding three raw-rate
columns to the original baseline lowers RMSE from 2.905349 to 0.651744 without
reading trained weights; cumulative displacement then worsens prediction to
0.795009. Its large B0 gain is not comparator-robust. Full-spectrum segment entropy
worsens B0 + displacement from 1.003975 to 1.260228, but helps under B3 + displacement
(1.420297 to 1.347976). No predictor passes all four unconditioned baselines.
Other features change flags as well; all 84 comparisons remain visible in tables
and an all-feature heatmap. Richer comparators can overfit/extrapolate poorly:
these are 8–25 parameter models fitted to 45 training rows, not new independent
experiments or model-selection validation. The right conclusion is comparator
sensitivity, not that all geometry is useless or that displacement is the mechanism.

Final archive verification at 12:44:30 UTC preserves all 226 preceding analysis
payloads, all 112 original source files and eleven protected inputs. The final
figure is visually checked and contains no Type 3 fonts; both display versions
have identical data. Both new analysis reports remain local, pending immutable
data-only backup. Endpoint effects, original locked results, manuscript, training,
retrieval, failed functional coordinator and all external state are unchanged.

**Actual weight-to-retrieval prediction complete at 12:24 UTC.** The
[new actual report](reports/dense-v3-weight-retrieval-v1/README.md) joins every
checkpoint by native run identity, retained step and original seal. Both geometry
branches and all 840 task scores match their accepted immutable inputs. All nine
original features and five distinct exact sensitivity features retain the locked
four-dose holdouts, baseline, exact arithmetic, undefined handling and support rule.
The first actual calculation finished 12:15:02 UTC. Independent full augmented
rational OLS reproduces 840 predictions, 56 fold MSE comparisons, 14 pooled decisions
and 14 associations; independent input mapping checks all 840 feature values and
300 original/exact correspondences. Copied source and inputs reproduce all 17
outputs byte-for-byte at 12:18:57 UTC. Final archive verification at 12:24:51 UTC
also rechecks 112 original source files and eleven protected inputs unchanged.

The baseline pooled RMSE is 2.905349 nDCG points. Cumulative displacement/weight
reduces it to 1.003975, improving 4/4 folds; stable-rank features improve 0/4 folds
in both measurement families. Full-spectrum segment entropy reduces RMSE to
2.347064 with 3/4 improved folds. Five original and one exact predictive flag are
true, not six independent discoveries or significance tests. The specified additive
baseline omits nonlinear learning-rate and optimizer interactions; displacement
may proxy these effects. No functional utility, mediation, causal or richer-baseline
claim is established. All-feature PDF/PNG/SVG plots are generated and visually
checked; source/report remain local WIP. No model, retrieval, original protocol,
manuscript, functional coordinator, remote publication or formal consumer changed.

All twelve corrected primary runs, sixty retained checkpoints and **840 / 840**
checkpoint/task retrieval units are complete. Baseline remains **14 / 14** and
validation **12 / 12**. Original pool B completed at 08:59:47 UTC and pool A at
10:47:25 UTC; the original observer has a terminal completion record. No primary
training/evaluation task remains to run. Do not poll or restart retired handles.

The [complete trajectory report](reports/dense-v3-complete-trajectories-v1/README.md)
contains all actual curves and eight CSVs, original completion records and copied
sources. Native sixty-state readback returned zero at 10:51 UTC; complete numerical
readout at 10:52 UTC and copied-source replay at 11:08 UTC agree on all eight CSVs.
The preceding 54 records and all six endpoint comparisons are unchanged. Generated
selected-configuration trajectories provide descriptive process observations, not
new significance, training-seed replication or mechanism claims.

All sixty model checkpoints and **all sixty complete evaluation outcomes** now
have verified off-host backups. The final six newly backed outcomes are included
in the [full fourth-stage snapshot](docs/fourth-stage-evaluation-restoration.md):
557 files recovered anonymously at 11:23:59 UTC, with 168 raw scores / twelve exact
means reconstructed by the isolated copied reader at 11:25:49 UTC. Twelve older
corrected subtrees and twenty other root entries remain unchanged. Complete
curves/tables now also have their [separate data-only backup](docs/retrieval-trajectories-restoration.md):
all 26 files / 1,093,911 bytes recovered at 11:40:10 UTC and independently reconstructed
at 11:40:28 UTC. Its raw-score index joins all five stage snapshots, rechecks 2,888
previously recovered files and matches every one of 840 scores. Revision
`dbcd12376f483347cc570584abc588503c0a5fa0` preserves thirteen older corrected
subtrees and all twenty other root entries. No new model or bootstrap execution
occurred. Functional analysis remains **1 / 61 vector states, 0 / 61 feature
states**; its failed coordinator is not recovered or implicitly authorized to
restart. Formal crossed continuation remains **0 / 12**. Paper, source publication
and clean-host scientific reconstruction remain incomplete. The safety restrictions
on protected helpers, stopped controllers, external writes and historical removal
are unchanged. All older counts below are historical evidence.

## Historical handoffs — evaluation was active at the recorded times

**Factorial model/argument factory checked; GPU continuation still pending — 2026-09-10.**

The new internal `factorial_v3_factory.py` generates all twelve fixed recipes,
uses the unchanged primary model/loss loader and constructs the separate bound
factorial Trainer only inside an existing four-rank NCCL worker. Its 68 parent
files remain unchanged. Final tests are **137 passing cases** (30 new factory
plus 107 existing run-component cases), not independent training experiments.
Both genuine sources load on CPU with all 134 FP32 tensors unchanged; the final
call rejects 20 configuration/tokenizer counterexamples. Its receipt SHA is
**10516d79cb814d44fc6fda8904a7349f194228f54e3094a857fa5aa27cb21fe8**.
The recipe metadata uses explicitly synthetic rates; no genuine GPU calibration,
forward/backward, formal Trainer or branch was executed. Read
`reports/engineering-archive/dense-v3-factorial-factory-v1/README.md` in the story
tree. All five owned CPU handles are terminal. Do not repeat these completed
checks as missing work. Real GPU calibration/admission, worker/source/resource
handoff, complete branches/outcomes, portability and scientific release remain.

At 11:03:02 UTC primary BEIR is **16 / 840**, baseline **14 / 14**, with eight
exact live/R workers and no failure receipt. At 11:03:59 validation is **7 / 12**
and functional is **0 / 61**, both live/S. The one-GPU priority request remains
unanswered. All 12 primary runs and 60 backups remain complete. No GPU task,
frozen numerical/dispatch source, historical controller or manuscript changed.

**Source-bound factorial run component checked; real GPU admission pending — 2026-09-10.**

Two new internal modules now bind the genuine v3 source, both authenticated
calibration chains, complete 50K text view, reset operator/order seed and all
68 source files to native run checkpoints and complete-run reading. The bound
Trainer owns the original query/document prompt collator and five-stage callback;
its inherited numerical training, optimizer and checkpoint/resume hooks are unchanged.

The final **107 focused cases pass**. The actual full 50K / 21-field branch read
and both 134-tensor genuine source-weight reads pass with CUDA hidden. The final
`actual/actual-view-fifth.json` SHA is
**3e87392057fdca49460f1c40883881183dc9126d64a7940aef27c0fd62b6baf4**.
Three versioned four-rank CPU writer-control calls also pass, using explicitly
fake Trainer callers and synthetic payloads; these are NOT actual Trainer/GPU
runs. Their default deep reader rejects the synthetic files. Initial fixture
errors and three newly found container-check omissions are preserved with their
original failed tests and source versions; the final cases reject those inputs.

Read `reports/engineering-archive/dense-v3-factorial-run-contract-v1/README.md`
in the story-refactor tree. All this milestone's owned CPU calls are terminal.
Do not repeat completed input/component checks as missing work. Actual GPU
calibration, default-topology four-GPU bound Trainer/save/readback verification,
twelve genuine branches, complete outcomes, cross-host run/calibration transport
and final scientific/publication admission remain required. This is not a launch
CLI, scheduler, resource handoff, committed-source release or optimizer finding.

At the exact task snapshot **10:33:32 UTC**, primary BEIR is **16 / 840**, baseline
**14 / 14**, and all eight exact primary workers are live/R with no failure receipt.
Validation remains **7 / 12** at 10:29:41; functional is live/S at 10:33:43 with
**0 / 61** states, waiting for validation. Existing dispatch/numerical sources
are unchanged. The one-GPU priority request remains unanswered, not approved.
All twelve corrected primary runs and sixty remote backups remain complete.
No historical controller/helper/HF/GitHub boundary or manuscript was changed.


**Genuine v3 calibration component checked; GPU execution still pending — 2026-09-10.**

The new internal `factorial_v3_inputs.py` and `factorial_v3_calibration.py` connect
the genuine corrected source/data evidence to fresh gradient-history production,
complete direction-norm reading and all twelve reset continuation cells. The four
hidden rates are derived from all 88 matrix norms by the unchanged global
Frobenius rule, not hand-entered or selected using retrieval. Gradient and final
calibration receipts require externally supplied content bindings. Neither saved
calibration moments nor historical gradients become branch initialization.

Both actual-input consumer reads passed on their own source versions. The final
`actual/actual-inputs-third.json` has SHA
**3640dc3a0deaa142d7e09eabfcc8450ab786db1b7302e04f470518557a63176f**.
It authenticates 66 assembled files, including all 56 unchanged primary files,
both native source checkpoints and the exact 50K / 32-row input evidence.
The final **57 focused tests** pass. Actual tiny-tensor replay is checked against
an independent FP64 Adam reference and eight actual primary Muon updates; complete
calibration-to-twelve-cell wiring remains explicitly synthetic. Initial 45- and
56-case results, original source versions, and the first audit-wrapper schema
failure are preserved, not aggregated into extra experiments.

Read `reports/engineering-archive/dense-v3-factorial-calibration-v1/README.md` in
the story-refactor tree. All its CPU test/input calls are terminal. Do not repeat
them as missing work. No GPU calibration, calibrated primary rate or formal branch
is claimed, and the component has no scheduler. Its internal GPU functions require
a separately admitted exclusive worker; they do not waive source/runtime/release,
resource or primary-completion gates. Real GPU save/readback, four-GPU DenseOn
branch verification, formal run identity and whole-run consumers remain required.

At the exact observer's **09:27:30 UTC** snapshot, primary BEIR is **12 / 840**,
baseline **14 / 14**, and all eight exact primary workers are live/R with no failure
receipt. Existing dispatch and numerical sources are unchanged. The unanswered
one-GPU request still is not authorization for a handoff. Complete primary training
and all sixty remote backups remain done; no scientific winner is claimed.

**Genuine factorial inputs checked; no formal branch admitted — 2026-09-10 09:01 UTC.**

The new independent data audit reconstructs the fixed 50K branch selection and
all 32 calibration rows in their original order. All 21 fields match the revised
500K parent; none of the six replaced parent positions occurs in either selection.
No resampling or historical-gradient reuse occurred. Both genuine corrected
step-2345 source states (AdamW 3e-5 and Muon 3e-4) load with all 134 saved tensors
bitwise unchanged. Each state admits both empty reset optimizer constructions
with identical 88 / 1 / 45 routing and no inherited moments or gradients.
This is CPU loading only, not a BF16/GPU forward, calibrated rate or formal run.

Read `reports/engineering-archive/dense-v3-factorial-inputs-v1/README.md` in the
story-refactor tree. The independent `verification.json`, SHA
**9e92896fd14855ea1635f5571e824fc38c9071297e7fa3b4b848a55db08aa526**,
rehashes 49 original data inputs, 40 genuine checkpoint files, both original
immutable receipt pairs and the eight archived evidence files. All 24 focused
comparison controls pass. Original evidence and the four ops preimages are
preserved. These completed checks must not be repeated as missing work.
The v3 outer calibration/input consumer, genuine GPU calibration, four-GPU
DenseOn branch verification, twelve formal branches and complete outcomes remain
required. No frozen numerical source, original protocol or manuscript changed.

At the separately timestamped 09:00:29 UTC task snapshot, primary BEIR is
**10 / 840 accepted cells**, baseline **14 / 14**, with eight exact primary workers
live/R. Validation is still **7 / 12** at 09:01:01 UTC; the functional coordinator
is live/S with **0 / 61** encoded states at 09:01:02 UTC, waiting for validation.
No failure receipt is present. Existing sources/dispatchers remain unchanged.
The one-GPU priority question is still unanswered; no handoff is authorized.
The complete twelve-run primary training and sixty remote backups remain done.

**Zero-update baseline complete; primary retrieval and analysis queues active — observed 2026-09-10 08:01 UTC.**

Corrected training is **12 / 12 complete**, all **60 / 60 checkpoints** are
HF-verified, both original weight-geometry branches are **60 / 60 complete**, and
the all-state update map/full-dimensional endpoint readout are complete.
Do not rerun those completed jobs. Weight spectra alone are not useful embedding
dimensions or a retrieval mechanism. Read `launch/weight-geometry/COMPLETE.md`
and `launch/descriptive-update-map/README.md` in the experiment.

**The zero-update baseline is now 14 / 14 complete.** The original producer
finished complete native verification at **07:56:44 UTC**. Its exact coordinator
**7860 / start 297755793 is absent**, independently observed at 07:59 UTC.
Do not use its old live-only read command or restart/re-evaluate it. Coordinator
OS exit remains unobserved/null; all fourteen actual task workers exited zero.
Fresh original full-result readback matches the producer exactly, including all
16 unique final result files. Full-suite macro nDCG@10 is **0.5129607143**
(**51.2961 / 100**), a baseline reference, not an optimizer contrast.
Read `/root/embedding-optimizer-v3-experiment/launch/baseline-evaluation/COMPLETE.md`.
Complete provenance and the real readback command are in the adjacent
`complete-reference-handoff.json`. Final shared baseline settings are now complete;
earlier provisional task snapshots stay unchanged.

Primary retrieval is **8 / 840 accepted task cells**, with **0 / 60** complete
fourteen-task primary checkpoints at this observation. The original queues
naturally reused the two baseline tokens for final-checkpoint Muon 1e-4 and
NorMuon 1e-4 ClimateFEVER jobs. Eight exact primary workers were R at 08:00:58 UTC.
No source, corpus, context, scoring rule or original scheduling policy changed.

The incremental native task observer is live: **60083 / start 299816818**,
session **6884**. Its 24 bounded tests and actual first/incremental readbacks pass.
It reads new original exit-zero results without dispatching, restarting, selecting
optimizers, computing partial-task means or changing GPU leases.
Read `launch/task-observer/README.md`; use its **`status.py`** for exact observer
liveness and separately timestamped counts. Do not duplicate or edit this watcher.
Its initial launch/readback history is in `launch/task-observer/handoff.json`;
new immutable task observations accumulate under `run/accepted/`.

Full validation remains **7 / 12**, original coordinator **13007 / start 298006520**
live/S at 08:01 UTC. Functional vectors/features remain **0 / 61**, with the original
functional coordinator **42916 / start 299159282** waiting for all twelve validations.
Both sources and their original scientific definitions remain frozen.

**The one-GPU priority request is still pending, not approved.** Read
`launch/validation-handoff/PRIORITY_REQUEST.md`. Existing BEIR dispatch claims
released slots faster than validation's polling; the baseline's final slots were
again reused by primary BEIR. Do not re-ask the question, infer approval from an
automatic continuation, stop a parent that retains leases, force-unlock a slot,
truncate a task or perform an unreviewed handoff. Current jobs continue unchanged.

Next complete the existing full 840-cell matrix, all-twelve validation selection,
61-state functional encoding/features and held-out-dose inference. The scoped
crossed continuation, complete scientific consumer, portable source/artifact
release and defensible NAACL manuscript remain required. No overall optimizer
winner, useful-dimension explanation, complete paper or committed release is
claimed. Engineering incidents stay out of every manuscript section.
Preserve all frozen sources, original/failed evidence, old stopped controllers,
protected-helper prohibition, GitHub 403 and rejected historical HF erasure limits.

**CPU factorial Trainer integration complete in its bounded scope — 2026-09-10 08:35 UTC.**

This supersedes the historical paused/preparation-only Trainer descriptions below.
All thirteen four-rank CPU attempts are terminal: eleven successful and two original
failures preserved. All three full 50K seeded loaders, both actual 391-step routed
optimizers and all five saves pass. Fresh AdamW step-79 and Muon step-313 resumes
reach endpoints bitwise equal to their uninterrupted references, including named
moments and scheduler state. Separate unclipped/no-dropout CPU probes compare
every parameter's actual gradients against an independent FP64 global mean on
128- and 80-group updates; both pass the unchanged 5e-6 / 5e-4 tolerances and
reject quarter/fourfold controls. The stochastic resume fixture remains separate.
Eighteen new placement cases and the existing 82 optimizer cases pass.

Read `/root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-factorial-trainer-integration-v2/README.md`.
Its `complete.json`, SHA **c4320ee481871ae4dc24df2b0ac1b266fe1d9544d389b39b73d3b6b822880630**,
binds 1,088 archived files. Original failures/sources/receipts are retained.
The only development component repair permits an identical empty step-zero
optimizer placement no-op required by Accelerate; nonempty/unbound restore stays
rejected. No primary arithmetic, live source, protocol or manuscript changed.

Do not rerun these completed CPU tests as missing work. They are not formal runs
or default-topology DenseOn/four-GPU admission. Genuine v3 source-state, calibrated
rates and branch-data binding, actual DenseOn distributed verification, whole-run
consumer and twelve formal branches remain required. Primary BEIR remains active;
validation/functional priorities and the unanswered one-GPU request are unchanged.


**Earlier functional queue snapshot — observed 2026-09-10 06:01 UTC.**

All corrected training remains **12 / 12 complete**, with **60 / 60 remotely verified
checkpoints**. No new training or re-training is pending. This block supersedes older
live-training and weight-worker counts below.

- The original approximate weight branch is **60 / 60 complete**, with all twelve
  run-level native saved-weight numerical readbacks passed. Actual worker exit is
  **0** at 05:31:31 UTC. Its four tables contain **60 / 60 / 660 / 10,560** rows
  (checkpoint geometry / optimizer-pair summary / run-pair overlap / subspace health).
  The completed output hashes were independently reread; that reread did not repeat
  tensor computation. **Do not restart or poll the completed approximate worker.**
- The separate exact-spectrum/projector branch has **25 / 60 stages verified**.
  Its original worker **36023 / start 298866251** is live, state R, under the unchanged
  supervisor **36021 / start 298866243**. Session **72001** remains live for that branch.
- Full validation is **7 / 12** with no failure/selection/completion receipt. All eight
  exact authorized baseline/BEIR GPU workers are live on eight distinct tokens.
  Those sources/coordinators remain frozen and unchanged.
- The new complete **61-state functional campaign is queued and live**, not encoded:
  coordinator **42916 / start 299159282**, session **39956**, state S, exact argv verified.
  It waits for all twelve full validations, then uses at most one normally released
  GPU with **both inherited lease namespaces**, before CPU-only full feature replay.
  Actual vector states **0 / 61**; actual feature states **0 / 61** at this snapshot.

Read `/root/embedding-optimizer-v3-experiment/launch/functional-dimensions/RUNNING.md`.
The immutable functional input record authenticates all twelve genuine completion
proofs, the exact relocated five-file/224-row probe and all 61 original state plans.
Its **25 bounded operational tests pass**, including a real temporary-file child
FD-lifetime check; they are not GPU encoder or feature acceptance. Fifty-three
original functional dependencies and all encoding/numerical definitions remain
unchanged. The source-bound dispatcher is
**3b02f1d8486f3c2b41d0403cbf291580d0ca32868b8b29cb05514a22770257f5**;
authority **d72527595e2c963eae7fd46b0a1de2bd83d1ea15a863ae467a1065cfc5b2f336**.

Do not edit, restart or duplicate the live functional dispatcher, its input source,
authority or numerical dependencies. The first actual GPU state still requires its
native loading/save/readback acceptance; on failure preserve outputs and stop new
dispatches, with no automatic retry or tolerance change. The frozen protocol keeps
BF16 GPU encoding, FP32 raw vectors, literal FP64 coordinate deletion, all 80 masks
and all three endpoint rotations. No validation/BEIR score chooses a state.

The new exact read-only observer is `functional-dimensions/observe.py`. Evidence and
commands are preserved in
`launch/observations/functional-queue-and-complete-approximate-weights-20260910.json`.
Continue existing full BEIR/validation, exact geometry and this queued functional
campaign. Original DRAFT/single-selection guards remain unchanged and unpassed;
separate operational evidence is not a committed-source or scientific release.
No old stopped controller transition, protected-helper access, GitHub retry,
historical HF deletion or manuscript modification occurred.

**Earlier CPU-weight launch snapshot — observed 2026-09-10 05:15 UTC.**

All twelve corrected training runs and all sixty remote checkpoint backups remain
complete. Full-corpus BEIR is running unchanged; full validation has seven of twelve
models scored, no selection yet and no failure receipt at this observation.

The new CPU-only all-rate/all-stage weight campaign is live. Both branches retain
all 60 checkpoints and 88 hidden matrices: original approximate measurements plus
the separately declared exact-spectrum/projector measurements. Actual approximate
production and native saved-weight numeric replay have accepted **15 / 60 stages**
(AdamW 1e-6, 3e-6 and 1e-5); exact production/replay is active, with no complete
run-level verification receipt yet at this snapshot. These are real primary
measurements, not the earlier synthetic/diagnostic fixtures or an optimizer verdict.

- Supervisor: PID **36021**, start ticks **298866243**; session **72001**, live.
- Approximate worker: PID **36022**, start ticks **298866251**.
- Exact worker: PID **36023**, start ticks **298866251**.
- Both workers: parent 36021, nice 10, four CPU threads, CUDA hidden; no GPU leases.
- Frozen worker SHA **5ab87568a1a9050e423c2e778a4fda8b53f906eec2ee2f5a824e2522c09c0ec6**.
- Frozen authority SHA **44aad4192ed0871ff9bf0be370ba7f1f73e7f3e789f2c65e92481bd7d884bff4**.

Read `/root/embedding-optimizer-v3-experiment/launch/weight-geometry/RUNNING.md`.
The exact owned reader and source/launch/first-result evidence are archived in
`launch/observations/weight-geometry-started-20260910.json`. Nineteen actual-input
operational admission tests pass. Original geometry producers/readers and both
measurement contracts remain unchanged; their original DRAFT/single-selection
guards are not waived. Extra completion provenance is outside native result bundles.

Do not edit/restart/duplicate either live CPU branch or its supervisor. Preserve
partial outputs on failure and inspect the exact source-authorized handles.
Continue both complete geometry grids and existing evaluations. Functional-vector
encoding remains required under its frozen BF16/GPU protocol; do not silently
substitute CPU encoding or equate a spectral rank with useful retrieval dimensions.
No optional factorial/paper-tool work should delay these actual measurements.

Nonblocking stack observations at 05:10–05:13 show the owned BEIR workers in
tokenization, model forward and full-corpus search. One nonblocking sample failed
to copy a moving Python object; the same exact worker was still live and its retry
succeeded. This was an observation failure, not a worker failure; no restart,
signal, pause, helper inspection or numerical-source change occurred. No whole-
evaluation throughput or optimizer-quality conclusion follows from these samples.

**Training-completion archive: 2026-09-10 04:34 UTC.**
**All 12 / 12 corrected v3 training runs are complete; 0 active / 0 queued.**
**All 60 / 60 checkpoints are remotely verified: 1200 files,
89,889,820,336 logical bytes.** Full-corpus BEIR and full validation are now
actually running. This block supersedes all older live-training counts,
waiting-for-resource descriptions and old tool/session entry points below.

The source-bound primary design remains unchanged: DenseOn only, the same revised
500K queries and seed 42, one positive plus seven fixed negatives, no in-batch
negatives, context 8192, four rates per optimizer, one full epoch and all five
stages 782 / 1563 / 2345 / 3126 / 3907. No training restart, initialization change,
horizon reduction or numerical-source change occurred. Historical pre-v3
12-run / 60-checkpoint results remain scientifically held and are not these runs.

**Actual final training acceptance**

The unchanged completion readers accept all twelve recipes and all sixty stages.
The ten earlier runs retain their original actual exit-zero and exact-view proofs.
The final two have independently observed termination and full-artifact/exact-view
acceptance; their OS exit codes remain **unobserved/null**, not inferred zero.

| Final run | Actual termination | Full-artifact verification | Original PID / start ticks |
| --- | --- | --- | --- |
| NorMuon 1e-4 | 04:24:31 UTC | 04:25:04 UTC | 1959407 / 295778135 |
| NorMuon 3e-3 | 04:26:08 UTC | 04:26:33 UTC | 1959573 / 295779725 |

Both original processes are now absent. The original unchanged observer performed
all-five-stage deep checks (134 parameter states per stage), full-horizon metadata
and complete 500K exact-view checks. Its separate accepted receipts are:

- `orphan-completion-recovery/receipts/verified-v3-normuon-1e-4.artifacts-verified.json`,
  SHA **97217c4a2394d44b82c49c8c0b0d2dddfe306f264c09a423d91004c930594241**.
- `orphan-completion-recovery/receipts/verified-v3-normuon-3e-3.artifacts-verified.json`,
  SHA **6b52984def4b11732a934df00c9d400b1a0923fb44948f567a2642e4dce52b44**.
- `orphan-completion-recovery/both-final-worker-artifacts-verified.json`,
  SHA **86da178b23df0e0663724d0194682dac4a5e3863c478d2ceffd261211824f03e**.

Original single-selection/released-contract guards remain unchanged and unpassed.
Never fabricate original final-pair `.exited.json` / `.view-verified.json` records.
The separately scoped completion proof must remain explicit in downstream consumers.

**Final durability and online logs**

The unchanged backup watcher produced these final immutable commits:

- NorMuon 1e-4 step 3907: **c7f1694bbff6a3c6643c5b66a0e9e1f333c61417**,
  1,351,494,773 bytes.
- NorMuon 3e-3 step 3907: **6a224f167062031521092590b63c37b7a0a1ed1b**,
  1,351,494,611 bytes.

Public repository: `qcz/embedding-optimizer-study-checkpoints`.
Only the `corrected-dense-correctness-v3/dense/4400f1ce.../` namespace is this
accepted campaign. Final `backup-receipts/all-60-checkpoints-verified.json`
is present, SHA **c758542395a2439ff6be3186e178963647d856fed88772ffc91e21dcd7cae80d**.

Actual integration **97444 is terminal/pass at 04:28:33 UTC**: all 60 inventories
and 180 immutable upload/final/audit receipt hashes agree; the accepted 58-record
prefix is preserved. All twenty local files in each of the two new final states
were freshly rehashed. Old large states were not redundantly rehashed, and this
integration is not a new manual network or deep-tensor audit. The original watcher
already supplied the independent immutable-commit remote audits.

Final W&B read **50224 is terminal/pass at 04:28:40 UTC**: both final runs are
`finished`, each with all **391** expected ordered metric rows through step 3900,
finite loss/gradient/LR and zero independently computed LR-schedule error.
Step-3907 completion is established by artifacts, not by inventing a W&B row.
The actual twelve-run/native-pilot read **73245 is terminal/pass at 04:29:39**.

The recovered backup PID **1990603 / start 296921594** and completion observer
PID **1993005 / start 297067443** are now absent, with their final completion
records present. **Do not restart completed training, backup or observer.**
Read-only endpoint watcher **cell 163 is terminal**; it stopped when the first
original worker disappeared. That expected monitoring transition was reconciled
against actual completion evidence, not treated as a training failure.
Old sessions 52244 / 97554 / 79029 / observer 685 and the two-live-worker reader
are no longer current monitoring entry points.

**Actually running evaluation coordinators — frozen, no duplicates**

| Coordinator | PID / parent / start ticks | Current phase |
| --- | --- | --- |
| Primary BEIR pool A | 597 / 1 / 297430092 | Training admitted; real pilot passed; full corpus running |
| Primary BEIR pool B | 598 / 1 / 297430092 | Training admitted; real pilot passed; full corpus running |
| Zero-update baseline | 7860 / 1 / 297755793 | Real pilot passed; full corpus running |
| Full 4096-row validation | 13007 / 1 / 298006520 | First full pilot passed; remaining models running |

Do not edit, restart or duplicate these live sources/authorizations or the original
evaluation/numerical modules:

- `evaluation-handoff/dispatch.py`: **5b8a89bed5e0eaa02a12585ee3f6c883ca3e162b550898d68322321ee41d8427**;
  authorization **2351e225387f42fe607d7e002a0717de79a78cba9f3c93f66652510083f0be7c**.
- `baseline-evaluation/baseline.py`: **338199de60336e374619245cbcf207f0085fe111288d91053695845ec0eb86c0**;
  authorization **22c9db02912a9b985a9f8118c5d42e703a2fcb3aac74ff2a5ef351aa3d81b89c**.
- `validation-handoff/validation.py`: **d2fc55c67670ea5b42cacb055f48379ef278a465ce86d95808231ab86049d913**;
  authorization **6a8dcdd579bd6c2f6ea82e3f4cd808f88e92e9a28b9e97f41ffa6447eb0e1f6f**.

The primary 56-file source assembly and separate exact 62-file validation assembly
remain frozen. The original source/data/runtime preparations and bounded tests are
complete, not missing work. All resources continue to use both per-GPU lease
namespaces; children inherit both FDs, and parent closure never explicitly unlocks
a surviving child. No broad process inspection or protected-helper access.

At **04:34 UTC**, eight exact source-authorized worker identities are live on
eight distinct GPU tokens. This is assigned-worker/liveness evidence, not a
measurement of instantaneous GPU utilization or whole-evaluation throughput.

| GPU token | Work | PID / start ticks |
| --- | --- | --- |
| 0 | Baseline ClimateFEVER | 27950 / 298605233 |
| 1 | Baseline FEVER | 27951 / 298605234 |
| 2 | AdamW 3e-5 final, ClimateFEVER | 28934 / 298609661 |
| 3 | AdamW 1e-5 full validation | 31447 / see its authenticated started receipt |
| 4 | Muon 3e-4 final, ClimateFEVER | 29609 / see its authenticated started receipt |
| 5 | AdamW 1e-5 final, ClimateFEVER | 29612 / see its authenticated started receipt |
| 6 | NorMuon 1e-3 final, ClimateFEVER | 29613 / see its authenticated started receipt |
| 7 | AdamW 3e-6 final, ClimateFEVER | 29616 / see its authenticated started receipt |

Workers are dynamic. Re-read exact `*.started.json` / `*.exited.json` under
`evaluation-handoff/pool-{a,b}/jobs`, `baseline-evaluation/run/jobs` and
`validation-handoff/run/jobs`, verify source/auth, then PID start ticks and argv.
Do not reuse a finished worker PID or infer failure from a stale live-only reader.
The archived worker reader does this without enumerating unrelated processes.

**Actual first results and their limits**

All three fixed SciFact pilots exited zero and pass the unchanged native task
reader at the pinned task revision, full decontaminated corpus and common 8192
context. These are individual task measurements, not optimizer-quality conclusions.

| SciFact pilot | nDCG@10 | Actual worker elapsed |
| --- | --- | --- |
| Zero-update DenseOn | 0.87144 | 23.09 s |
| AdamW 3e-5, step 3907 | 0.87996 | 23.00 s |
| Muon 3e-4, step 3907 | 0.87919 | 23.00 s |

Primary progress is **2 / 840 task cells**, baseline **1 / 14 tasks**, and
**0 complete fourteen-task primary checkpoints** at this observation.
Native shared `run_settings.jsonl` hashes remain provisional until all fourteen
tasks finish. The native reader discloses unused undefined auxiliary nAUC values;
these are not used as outcomes. No omitted corpus, shortlist replacement, test-
selected LR, available-case mean or small-pilot extrapolation is authorized.

The baseline is the authenticated zero-update DenseOn-unsupervised revision
0edbd55684eb782bce55ee74c95b25c97cbe7f43. Its inference package differs from the
pristine release only in the declared 512-to-8192 maximum-sequence default;
weights, pooling, prompts and tokenizer remain unchanged. It is not one of the
sixty trained checkpoints.

**Full validation has actual GPU/save/readback acceptance now.**
At 04:34 UTC the unchanged coordinator has verified three complete models:
AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4; AdamW 1e-5 is active. A fresh CPU native
readback **91928 is terminal/pass at 04:34:11** for the first two models, each
with all 4096 rows, eight original scores, six preserved metrics and authentic
input/plan/completion identities. No model was re-encoded by that readback.

The first two overall mean validation losses are 0.1856933563 and 0.1795032038.
Maximum scalar loss-replay errors are 3.44e-7 and 2.41e-7 respectively, under the
unchanged original verification tolerance. These are bounded observed measurements,
not a selected-LR comparison or evidence that validation loss mediates retrieval.
All **twelve** complete validations remain required before the original minimum-
mean-loss selector runs; exact ties use lower LR, with no BEIR or margin input.

The actual first full validation runtime pilot is now complete. Do not repeat the
13 component tests, 62-file assembly, 4096-row preparation or prior GPU diagnostics
as missing work. See the unchanged `validation-handoff/DISPATCH.md` for details.
Its historical preparation-only wording is superseded by this real execution.

**Authoritative completion/handoff archive**

`launch/observations/all-twelve-training-sixty-checkpoints-evaluation-running-20260910.json`
is **907,061 bytes**, SHA
**1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3**.

Readback verifies 101 bound immutable files, all 180 checkpoint receipt hashes,
the preserved 58-record prefix, actual twelve-run/native-pilot acceptance,
both original final termination/artifact proofs, final W&B logs, two complete
validation readbacks, eight exact live worker observations and four old ops
preimages. Mutable logs/shared settings are not immutable release bindings.
An overlong first read-only inspection command failed before process creation;
its actual error is retained, and the compact reference-based retry passes.
No training, scoring, numerical source or scientific rule changed.

Current read-only commands are embedded under:
`actual_complete_training_and_three_pilots.command`,
`actual_final_termination_and_artifact_receipts.command`,
`actual_gpu_workers.command`, and
`actual_coordinators.{evaluation,baseline,validation}.command`.
Use the completed numerical/checkpoint/native readbacks as evidence, not jobs to
rerun. For current monitoring use the exact coordinator/worker readers.

The prior accepted 58-state archive remains unchanged:
`launch/observations/sixth-pair-fourth-checkpoints-20260910.json`,
313,785 bytes, SHA **b55dec48357c556cd359b193fe080fa859fd45e37a4e6ee93daf0fbe13621d35**.
The full-validation launch parent bda09ca9..., baseline launch parent 7d469be9...
and original training/observer/evaluation parents also remain preserved.

**Next work**

Continue these existing evaluation jobs through the full fixed 840-cell grid,
fourteen-task zero-update baseline and all-twelve validation selection. Reconcile
any real worker failure through exact records; preserve partial outputs and do not
blindly relaunch, overwrite results or resume an old stopped queue.

Training priority is satisfied for the complete twelve-run primary matrix.
Necessary outcome and reached-weight/functional-utility analysis can now advance
alongside evaluation without touching live numerical kernels. Preserve all rates,
stages, genuine feature definitions and separate completion evidence; do not
waive old source/release guards to admit new results. Optional factorial or paper-
tool development must not delay actual evaluation and required primary analysis.

The paper, full evaluation, weight-space findings and reproducible publication
remain incomplete. No overall optimizer superiority, committed-source release
or full scientific completion is claimed. Engineering incidents stay out of every
manuscript section. Keep the approved old stopped controller chain and ledger
unchanged. GitHub's prior 403 and the rejected historical HF erasure remain hard
boundaries: no retry/bypass, no historical deletion, no protected-helper access.

## Earlier milestones — historical, not current dispatch instructions

The third pair started on September 8 at **21:03:12 / 21:06:45 UTC**, using unchanged workers,
source, data and recipes. Both passed all four ranks' full identity admission and
have finite metrics. Initial W&B read **19048** is terminal/pass: 8 / 5 expected
rows through the actual read-time endpoints 70 / 40, no missing/duplicate rows,
finite values and zero LR error. Both have now saved steps 782, 1563, 2345 and 3126
and continued training toward step 3907. Each queue has four runs including its current one. All twelve full-
horizon cells remain required; no complete new validation/BEIR result is claimed.

Observation cell **210** is terminal with the actual transition verified; it did
not stop any training. Exact sessions 52244/97554/79029 remain live at 21:13.
The preceding cell 205 is also closed. Do not restart completed children or
duplicate the current queues. Complete evidence, original full-run receipt hashes,
final W&B logs, actual new worker IDs and read-only commands are in
`launch/observations/second-pair-complete-third-pair-running-20260908.json`.
No training-source or recipe changes were made. These are training/durability
acceptances, not optimizer-quality findings or a scientific publication.

Post-warmup W&B read **69593** is terminal/pass at 21:56: Muon has all 44 expected
records through 430, NorMuon all 41 through 400, with no missing/duplicate rows,
finite metrics and zero LR error. The later local read observes steps 442 / 417.
Observation **227** is terminal with that actual warmup milestone; training and
backup sessions remain live, not stopped. Exact commands/results and the unchanged
twenty-checkpoint durability reference are in
`launch/observations/third-pair-post-warmup-20260908.json`. No new checkpoint or
scientific result was claimed by that earlier observation, and no training source/recipe was changed.

Both current step-782 checkpoints now passed actual deep model/optimizer/scheduler
state reads **64977 / 46665**: 134 parameter states, three groups [88,1,45], scheduler
step 782, sealed payloads unchanged before/after, no CUDA context created. Their
immutable HF commits are **96c42189cee58a52615f1fd834a17d40d2f8cd4c** (Muon 1e-3,
1,350,863,552 bytes) and **916108e06bf2eef4c4aac301dec71c7673ebcc5d** (NorMuon 1e-3,
1,351,429,635 bytes). At that first-stage milestone, durability was **22 checkpoints / 440 files /
34,135,413,704 logical bytes**. Integration **36953** is terminal/pass at 22:44:
all twenty-two exact receipts, identities, inventories and recorded remote hashes
match; every new local file was rehashed. The immutable network audits were the
unchanged watcher's, not repeated by this integration.

Post-save W&B read **32912** is terminal/pass: all 84 / 82 expected rows through
actual endpoints 830 / 810, finite metrics and zero LR error. Both local logs
continued after saving, to 859 / 841 at 22:47. Observation **233** is terminal with
that milestone; all three actual training/backup sessions remain live. Exact
commands, deep-receipt hashes and full results are in
`launch/observations/third-pair-first-checkpoints-20260908.json`. These are single-
stage checks, not current-run completion or optimizer-quality findings. No source,
recipe or original gate changed; the next save at that time was step 1563.

Both current step-1563 checkpoints now passed immutable HF audits:
**e41efce14ec736bb7061f9d68cee51a97c32a776** (Muon 1e-3, 1,350,879,852 bytes) and
**cadfa256088822cbf383bfc5b1847107a7533619** (NorMuon 1e-3, 1,351,445,940 bytes).
At that second-stage milestone, durability was **24 checkpoints / 480 files / 36,837,739,496 logical bytes**.
Receipt integration **33974** is terminal/pass at 00:16: all twenty-four exact
receipts, identities and inventories match, and every new local file was rehashed.
The unchanged watcher performed the immutable network audits; this integration
is not another network or deep-tensor audit. The unchanged automatic whole-run
reader will deep-read all five stages at actual completion.

Post-save W&B read **20932** is terminal/pass: all 159 / 158 expected rows through
actual endpoints 1580 / 1570, finite metrics and zero LR error. Observer **251**
is terminal with the real second-checkpoint milestone. Actual sessions
52244 / 97554 / 79029 remain live, and local logs continued to 1602 / 1598 at 00:17.
Full read-only commands, receipt hashes and accepted evidence are in
`launch/observations/third-pair-second-checkpoints-20260909.json`.
Neither current run is complete, and no retrieval or optimizer-quality finding
follows from these checks. Those second-stage checks are complete, not missing work.

Both current step-2345 checkpoints now passed immutable HF audits:
**245fd8fd2ad6d70d3e073ae80a7b7afd1e2cddb4** (Muon 1e-3, 1,350,896,116 bytes) and
**606cf33223e0b98d2098dd1b5b2653339d362035** (NorMuon 1e-3, 1,351,462,207 bytes).
At that third-stage milestone, durability was **26 checkpoints / 520 files / 39,540,097,819 logical bytes**.
Receipt integration **64329** is terminal/pass at 01:49: all twenty-six exact
receipts, identities and inventories match, and every new local file was rehashed.
The unchanged watcher performed the immutable network audits; this integration
is not another network or deep-tensor audit.

Post-save W&B read **41976** is terminal/pass: all 236 / 238 expected records
through actual endpoints 2350 / 2370, finite metrics and zero LR error. Observer
**265** is terminal with the actual third-checkpoint milestone; training was not
stopped. Actual sessions 52244 / 97554 / 79029 remain live, and local logs reached
2373 / 2384 at 01:50. Exact commands, receipt hashes and accepted evidence:
`launch/observations/third-pair-third-checkpoints-20260909.json`.
Neither current full run is complete; the unchanged automatic reader will
deep-read all five stages at actual completion. At that earlier observation the
next save was step 3126. Those completed checks are not retrieval or optimizer-quality findings.

Both current step-3126 checkpoints now passed immutable HF audits:
**f4cd570f0f8af0e154756700a2813b07a0fbefcf** (Muon 1e-3, 1,350,912,414 bytes) and
**aab2fac6abb3542e378601215bd86ceff59c6d3c** (NorMuon 1e-3, 1,351,478,522 bytes).
Current durability is **28 checkpoints / 560 files / 42,242,488,755 logical bytes**.
Receipt integration **85587** is terminal/pass at 03:23: all twenty-eight exact
receipts, identities and inventories match, and every new local file was rehashed.
The unchanged watcher performed the immutable network audits; this is not another
network or deep-tensor audit.

Post-save W&B read **34479** is terminal/pass: all 314 / 317 expected records
through actual endpoints 3130 / 3160, finite values and zero LR error. Observer
**273** is terminal with the real fourth-checkpoint milestone, not stopped training.
Actual sessions 52244 / 97554 / 79029 remain live; local logs reached 3153 / 3174
at 03:23. Exact commands, receipt hashes and accepted evidence:
`launch/observations/third-pair-fourth-checkpoints-20260909.json`.
Continue toward final step 3907 without code cleanup or optional work. Both current
runs remain incomplete; whole-run acceptance still requires the automatic full
reader and explicit exact view-history audit. The existing queues then dispatch
AdamW 1e-6 / AdamW 3e-6; do not launch duplicate workers or restart coordinators.

CPU-only backup session **79029**, PID **1599515**, remains live and unchanged.
All ten first-pair checkpoints passed original local admission, immutable HF
payload comparison and the independent remote auditor: **200 files /
15,715,145,492 logical bytes**. The final-step commits are
`be7ad202180a32b9bf8941cbe858ec2b5f4f9500` (AdamW) and
`c594909339b51c495c0c0eefa9d3fd3de8a65521` (Muon), under the new public
`qcz/embedding-optimizer-study-checkpoints` v3 prefix. All ten matching upload,
final and audit receipts remain in the experiment's `launch/backup-receipts`.

The completed second pair's step-782 checkpoints are also remotely verified, at commits
`59efd55774a8ccc054802041844f2b309d8d7eda` (NorMuon) and
`5ee3804cc2358b32f70531792adb10e678459399` (AdamW 1e-5). At that first-stage milestone,
durability was **12 checkpoints / 240 files / 18,858,675,262 logical bytes**. Both actual first-stage
deep model/optimizer/scheduler checks pass (26277/78930, terminal): 134 parameter
states each, groups [88,1,45] / [89,45], scheduler step 782. Sealed payloads are
rechecked before/after, unchanged; no CUDA context was created. Exact receipts and
hashes are in the active continuation handoff. The paired evidence is recorded in
`launch/observations/second-pair-first-checkpoints-20260908.json`. This is single-stage
training acceptance, not by itself whole-run completion or an optimizer-quality finding.

Both new step-1563 checkpoints passed immutable remote audits, at commits
`895318dd58f5efd51cdf4d3c4ac238d40e28aa73` (NorMuon) and
`3b29a182baaf5f9acea3045d1d7dd7b2a3a0e6ab` (AdamW 1e-5). At that second-stage milestone,
durability was **14 checkpoints / 280 files / 22,002,237,679 logical bytes**. Read-only receipt
integration 34931 is terminal/pass at 16:28: all fourteen exact commits, run identities,
authority, file names, sizes and recorded remote hashes match. All files in the two
new local payloads were rehashed and match. This integration reads the original
watcher's successful immutable-commit remote audits; it is not another network audit
or a new deep-tensor check. The complete evidence and command are in
`launch/observations/second-pair-second-checkpoints-20260908.json`.

Both new step-2345 checkpoints passed immutable remote audits, at commits
`85bb81ac24c935aa5880591135d82815f114b54e` (NorMuon) and
`0568fa4c506b03db9376a9487f7851c7990ef43b` (AdamW 1e-5). At that third-stage milestone,
durability was **16 checkpoints / 320 files / 25,145,832,682 logical bytes**. Read-only receipt
integration 78085 is terminal/pass at 18:01: all sixteen exact commits, identities,
authority, file lists, sizes and recorded remote hashes match; every new local
step-2345 file was rehashed and matches. This is the same bounded receipt/local-payload
check, not a fresh network audit or deep-tensor check. Exact command and evidence:
`launch/observations/second-pair-third-checkpoints-20260908.json`.

Both new step-3126 checkpoints passed immutable remote audits, at commits
`7d3441e6360c02718b0f4a83fd8b91087a35de9a` (NorMuon) and
`adcc06c044590db2f6daab8dda5a1e59ecb2df7e` (AdamW 1e-5). At that fourth-stage milestone,
durability was **18 checkpoints / 360 files / 28,289,460,294 logical bytes**. Read-only receipt
integration 28441 is terminal/pass at 19:34: all eighteen exact commits, identities,
authority, file lists, sizes and recorded remote hashes match; every new local
step-3126 file was rehashed and matches. This retains the same bounded receipt/local-payload
scope, not a new network or deep-tensor audit. Exact command and evidence:
`launch/observations/second-pair-fourth-checkpoints-20260908.json`.

Both second-pair final step-3907 checkpoints now passed immutable remote audits:
`61218ad3c3569eb5a169cb58be8df879c7a531c8` (NorMuon 3e-4, 1,351,494,941 bytes) and
`0e7fbc537a8fd8e69b2cda07ab8dd8f254868836` (AdamW 1e-5, 1,792,165,282 bytes).
At second-pair completion, durability was **20 checkpoints / 400 files / 31,433,120,517 logical bytes**.
Receipt integration **85915** is terminal/pass at 21:10: all twenty exact commits,
identities, inventories and receipt hashes match; every new final-stage local file
was rehashed and matches. The immutable network audits are the original watcher's;
this is not another network/deep-tensor audit. Evidence is in the completed-pair
transition observation above. That earlier observation preceded the third pair's first checkpoints.

Final second-pair W&B read **65067** is terminal/pass at 21:07: both runs are
finished, with all 391 expected records through step 3900, finite metrics and zero
LR error. Full-run records separately prove step 3907 / epoch 1.0. Their complete
`.view-verified.json` SHAs are `ee4031e2a49b66e1fd5938acd2d3acc5df79eb7cc9573caed6fd19d737bc2535`
and `ce708b79780804b074e0c0fceea853a7e5d02fb31160e671693e01bb26b377af`.

The final first-pair W&B read 15759 is terminal/exit 0 at 13:15 UTC. Both runs
are finished; each has all 391 expected metric rows through step 3900, with no
missing/duplicate logging steps, finite metrics and zero independently calculated
LR error. Full-run state records separately prove terminal update 3907 and epoch
1.0. Read checks 52020/60286 and observation cells 95/105 are also terminal.
The second pair's earlier fourth-stage check 93685 is terminal/pass at 19:33: NorMuon has
317 expected rows through step 3160 and AdamW has 315 through step 3140, each with
finite metrics, exact cadence and zero LR error. These are the actual read-time
endpoints, not a claim of whole-run completion. The later local log observation
proves both continued after their fourth saves. Midway read 15435 is also terminal/pass
at 17:11, through the actual endpoints 1970/1950, and is retained in the third-stage observation.
Earlier quarter-progress evidence remains in
`launch/observations/second-pair-quarter-progress-20260908.json`.
Earlier checks 44641/72239/46612/21852/66149/17767 and observation cells 134/140/159/165/177/187 are terminal;
the three actual training/backup sessions remain live. No source or training parameter was changed during
this observation. Use NorMuon's
actual `unfused-bfloat16-additive-eps-v2` ID suffix, not the guessed Muon suffix
used by the failed read 7457. That was a read-target error, not lost training logs.

## Resolved post-training view-history guard error

Original coordinators **28273 and 55261 are terminal/exit 1**, not live jobs.
The training children succeeded; the subsequent old v3 completion guard compared
the library fingerprint after one column selection against the actual frozen
training path, which makes two identical selections. Exact reproduction gives raw
`0c6bd82f699a563c`, once `5a2cdf9a1ae149bc`, twice `0c29f5d460d1b4d7`.
Both original completion records correctly contain the latter.

The new read-only audit verifies all 18 immutable input files before and after
reconstruction and full equality of all 500K Arrow rows, metadata, column order,
features, format and absence of row indices. It passes, as do eight bounded
refusal cases including altered content carrying a forged matching fingerprint.
The original complete payload/deep-state reader was then rerun unchanged for
both actual runs (14423/39675, terminal/pass). Separately scoped
`<run>.view-verified.json` receipts record that complete check plus the explicit
two-selection-history audit. Original failed results and completion records are
preserved; no original `.verified.json` pass was fabricated.

The new remaining-ten-run coordinator reuses the unchanged original worker,
both GPU lease namespaces and exact stopped-controller checks. Only the faulty
post-training history assumption changed in this named operational amendment.
The frozen training source, data, original launcher, backup watcher, draft
protocol and historical guards remain untouched. Later code consolidation must
integrate this exact content/history correction into downstream consumers;
their unchanged original complete-run fingerprint guard still rejects these tags.
Do not silently relabel it a passed original guard, scientific release or Git commit.

Read `/root/embedding-optimizer-v3-experiment/launch/view-history-continuation/RUNNING.md`
for current handles, all source/authority/proof hashes and exact queue order.
Do not edit or duplicate the live continuation or watcher, restart the old queues,
touch the protected helper, change the old stopped chain, retry GitHub writes or
retry historical HF deletion. The operational incident belongs only in engineering
provenance, never in the manuscript. Optional tooling and code cleanup remain
deferred while training proceeds. The historical CURRENT_PROGRESS.json still
counts held old artifacts, not these new runs.

The earlier W&B page-size readback issue was also resolved read-only; preserve
`launch/observations/wandb-readback-20260908T105218Z.json`. Use page size 1000
for this bounded horizon and check actual cadence; do not deduplicate or fill
records. The previous preparation-only states below are historical and superseded.

## Bottom line

**September 7 priority change:** the owner now expressly asks for rapid primary
training once correctness is confirmed. Optional factorial/paper-tool development
is paused, not a primary-training prerequisite. The unchanged audited training
core and v3 consumers are assembled in `/root/embedding-optimizer-primary-v3`.
All 104 relevant training tests pass; the actual source-bound v3 contract and
pinned runtime load pass; the complete real data/base read and all twelve recipe
identities pass preflight. See
[the launch preparation](reports/engineering-archive/dense-v3-primary-launch-v1/plan.md).
No formal run has yet started. The local training-code commit exception remains
the explicit outstanding owner choice; do not infer it from a preselected answer.

At 15:02 UTC the exact project-lock check returned exit 0: twelve existing locks
were unheld and unchanged, four primary-namespace paths were absent, and the three
approved old dispatchers remained stopped before/after. This is a momentary lock
observation, not a claim that all GPUs are idle. The new primary and old common
GPU-lock roots differ; the reviewed launcher must protect both namespaces for the
selected pool throughout training without changing or restarting the old chain.
The numerical/data checks are complete; the already-asked local commit exception
is still unanswered. No formal training was launched during this continuation.

**The project is still before the new valid primary experiment.** The old 12 training runs and
60 scheduled checkpoints finished and are preserved, but remain on scientific hold. No complete
v3 run, accepted primary optimizer contrast or finished NAACL paper exists.

Prepared data, training-identity consumers, validation, outcome aggregation, geometry and retrieval
bridges pass their bounded checks. The portable raw-vector reader passes both complete full-width
cold audits, including independent replay and numerical refusals. The original-outcome reader
also passes both complete cold audits and 62 focused cases; its first test/launcher failures
remain preserved, with production numerics unchanged.
The new original-geometry reader now passes 67 focused cases and the **2,548-case** full
regression. Both complete cold audits 13684/20415 are terminal/pass, including all 28 semantic
refusals on both. Joint original/functional reconstruction and its preservation acceptance
are complete. Primary evidence generation now passes its focused and synthetic layout checks;
the latest whole-development regression exits zero (JUnit reports 3,065 tests,
3,056 serialized testcase elements, no failures/errors/skips). Next is actual factorial integration and complete primary publication,
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

## Current joint verification

The [joint raw-to-inference reader](reports/engineering-archive/dense-v3-joint-reconstruction-v1/README.md)
has completed both cold audits. Its corrected 25 focused tests pass. The complete synthetic
fixture and independent joins, original 540 predictions, functional 240 predictions, task-family
intervals and figure-point checks have returned successfully. Full regression 47895 is now
terminal/pass: **2,573 cases**, zero failures/errors/skips. Cold audit 18914 is also
terminal/pass: all 477 numerical outputs and all 21 semantic refusals are verified.
Independent complete replay **74613** is also terminal/pass, observed at 08:21 UTC. Follow
[the exact call record](reports/engineering-archive/dense-v3-joint-reconstruction-v1/RUNNING.md).
The independent replay matches all 477 outputs and passes all 21 semantic controls again.
All joint audit/test handles are finished. Validator 26171 exited zero and generated
validation.json with SHA-256 e6fd1dd99d14d8eec86937f63352a7e5b75083c4ef519d2b07fd56e8b8e85073.
It verifies the completed pair and preservation bindings at the bounded synthetic scope.
No new primary result or manuscript value exists.

The initial fixture finished all 61 full-768D states, then failed when its new builder omitted the
same CSV type decoding used by the actual inference author. A separate unit found a mismatch with
the frozen author's provenance file order. Both corrections affect only new integration code;
all failed sources and the original completed raw components are preserved. Reused synthetic raw
components are explicitly anchored, not new checkpoint admissions; the cold reader recomputes all
three branches. The new full-suite result does not accept the distinct training candidate.

## Latest completed work

The new [factorial optimizer/batching components](reports/engineering-archive/dense-v3-factorial-routing-v1/README.md)
pass **103 combined tests** in development and with the exact corrected candidate
factory. Actual pretrained loading verifies all 134 tensors, identical 88/1/45
named routing and empty states; toy saved/resumed trajectories match exactly.
Actual upstream sampler checks exposed 16 omitted groups in a direct 50K reuse.
The new balanced sampler preserves all 50,000 groups and the correct final
80-group mean; its actual project-InfoNCE gradient checks pass. This is not a
distributed Trainer or complete checkpoint-resume verification. Whole regression
**21015 is terminal/exit 0** and corrected-source focus **64963 is terminal/pass**,
both observed 14:18 UTC. No call in that archive remains live. The old 500K
cardinality control is unaffected by this particular tail. Original kernels,
manuscript and controller ledger remain unchanged. The next required work is the
actual Trainer and distinct run/completion identity, not more baseline reruns.

The new [active manuscript document component](reports/engineering-archive/dense-v3-manuscript-consumer-v1/README.md)
has **154 focused tests passing** and a complete fresh synthetic PDF check. It verifies
exact generated inputs, active constants, actual abstract expansion (166 words in the
complete fixture), all nine captions, fresh compiler inputs, page limits and fonts.
A counterexample shows why the old nominal abstract reservation was insufficient;
the actual sample's primary finding has 32 tokens against its old 30-token allocation.
No real result or new constants were installed in the paper. The component does not
admit primary/factorial data, and is not the final strict consumer. Whole regression
**5462 is terminal/exit 0**, observed 13:05 UTC; 77334, 65097 and 3676 are also terminal/pass. Preserve
the five source-current files and follow the archive's exact call record.

The [v3 evidence authoring entry](reports/engineering-archive/dense-v3-primary-publication-v1/README.md)
has **63 passing focused tests** and complete generation from the single joint synthetic
population. It retains all primary/secondary/dynamics/systems/geometry/functional tables,
actual v3 selection and system definitions, exact decisions and undefined results. Its actual
CLI refuses missing primary runs and simulated admissions before creating outputs. The new
CSV wire-type and full-paper table-label failures are preserved; only new adapters changed.

A complete synthetic paper copy now passes layout: main text ends on page 7, all required
floats are correctly placed, the conservative abstract budget is 178/200 words, and there
are no overfull boxes or Type 3 fonts. Pages 4, 5 and 9 were visually checked. Original
functional LaTeX is retained byte-for-byte; its manuscript-specific version adds only the
required table label. No synthetic result was installed in the real paper.

Whole-development regression **44536 is terminal/pass**, with all 2,636 cases passing and
zero failures/errors/skips. Preservation validator **65642 also exits zero**; its bounded
synthetic-preparation receipt is SHA-256
34a28795f0a6b1682971e8e0424137224a18b4ffafa80f4a5f3bae38e7d65e8c. Source-bound
portable publication and strict manuscript-consumer integration remain incomplete.

The [new publication archive/reader](reports/engineering-archive/dense-v3-publication-reconstruction-v1/plan.md)
has 27 passing focused cases, including actual missing-primary refusal and relocated
source-only contract loading. Full synthetic cold reconstruction **40446 is terminal/pass**:
487 outputs, 83 archived package imports and all seven rehashed output refusals.
Whole-development regression **28989 also passes all 2,663 cases**, zero failures/errors/skips.
Independent full replay **64182** is now terminal/pass: all 487 outputs match and
all seven output controls are refused again. Validator **30075** also exits zero;
acceptance SHA-256 1b9044c4506b483d1dddadb54819c0a7ae5809f15103b743ae6d98b5e0ad2e62.
No handle in that archive remains live. Follow its exact call record; no actual primary admission,
strict manuscript integration, physical cross-host proof or scientific finding is supplied.

The [new exact-branch integration](reports/engineering-archive/dense-v3-exact-reconstruction-v1/plan.md)
adds full-spectrum/projector reconstruction and the separate five-feature sensitivity
to the complete original/functional path. Its 50-case focused retry passes, with the
first two test-message assertion failures preserved. All 1,584 real retained
diagnostic spectrum records reproduce exactly. Independent full-fixture scalar/set
checks and SymPy's 300 exact predictions also pass. These checks are terminal.
Full cold reconstruction 79178, whole regression 82844 and independent replay 70022
are now terminal/pass: all 503 outputs match and all 15 controls pass on both cold
attempts, while all 2,713 full-suite cases pass. Validator 78568 also passes; acceptance
SHA-256 d9888b172648555b54895c8a0b0acc751a3b8a345b7872ebf929ebf7038f5c59 binds only this
synthetic/diagnostic scope. No handle in that archive remains live. No frozen numerical
kernel, protocol, manuscript, live training source or dispatcher changed.

The new [exact publication entry](reports/engineering-archive/dense-v3-exact-publication-v1/plan.md)
retains all original evidence and adds the separate five-feature sensitivity to the
actual checkpoint-backed publication path. Two additional input-timing counterexamples
found and verified a fix in the new adapter; the full 95-case focused suite passes.
Both complete synthetic generations pass with ten identical outputs. The complete
v3 PDF layout passes (main end page 7, conservative abstract 178/200, eight required
labels, no overfull boxes/Type 3); three pages were visually inspected. The old-layout
failure is preserved. Whole regression 7103 is now terminal/exit 0. Its complete XML
reports 2,808 tests with no failures/errors/skips; the nine-element difference from
the 2,799 serialized cases is preserved, as in the preceding milestone. No accepted complete active-v3 paper consumer
or checkpoint-backed positive publication exists.

The earlier [14-case renderer candidate](reports/engineering-archive/dense-v3-primary-renderer-v1/README.md)
and its four layout samples remain unchanged as provenance. The new entry above extends
its checked behavior; neither archive supplies real optimizer findings.

[Original geometry reconstruction](reports/engineering-archive/dense-v3-geometry-reconstruction-v1/README.md)
has completed both full numerical checks:

- All 60 native-shape synthetic states, 5,280 matrix records, 10,560 health records and four
  original tables are covered. All 660 run-pair overlaps and 60 optimizer groups are recomputed;
  raw weight metrics/spectral health are retained measurements, not freshly remeasured.
- An independent set/rational oracle verifies 14,480 coordinate bases and all pair/group/entry
  aggregates, including 132 undefined pairs. These are explicit hypothetical fixture values.
- **67 focused cases and all 2,548 full regression cases pass**, without failures/errors/skips.
  The first cold audit 13684 passes all 28 fully rehashed semantic controls using only 72
  archived package imports. Independent replay **20415 is also terminal/pass**: all eight
  numerical outputs match and all 28 controls fail as required. These checks do not establish
  physical cross-host execution or complete bridge/inference/publication reconstruction.
- A missing zero-norm-statistic consistency check was found and added in the new reader;
  all nine real diagnostic checkpoint rows still match exactly. The initial ten counterexamples
  and a separate Python test-helper failure remain archived; scientific kernels are unchanged.

[Original-outcome reconstruction](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
now passes its bounded local numerical checks:

- Replays all 12 × 4,096 raw validation records, reconstructs validation-only selection, reparses
  all 840 pinned BEIR task files and exactly reconstructs ten complete original outcome tables.
- 62 focused cases and the **2,481-case full retry** pass without failures, errors or skips.
- Complete cold audit **68209** and independent replay **40528** are both terminal/pass. Their
  14 numerical output files match exactly; each child uses only 71 archived package modules and
  refuses all 21 fully rehashed semantic controls. Network and original source/producer paths
  remain blocked; the old editable-install search entry is removed before package import.
- The first missing-file test-message assertion and blocked editable-path audit are preserved
  with exact source and inputs. Only the new test and audit launcher changed; no production
  validation, statistical or outcome-reconstruction kernel changed after the first smoke.
- Scores, checkpoint admission, timing and hardware remain explicitly synthetic. The cold
  fixture retains previously verified real validation row identities; unit tests generate their
  own identities for self-contained execution. No model encoding, retrieval, primary admission,
  physical cross-host proof or manuscript installation is supplied.

[Raw-vector reconstruction work](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
remains unchanged at its earlier **bounded numerical verification** boundary:

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
  It is not the complete integration fixture used by the new reader above; its minimal
  placeholders remain unchanged and are not accepted as full checkpoint metadata.

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
opening the old paths. Both raw-vector and original-outcome readers now reconstruct their local
numerical outputs. Original geometry reconstruction also passes both complete cold audits;
joint original/functional bridge and inference reconstruction passes both cold audits.
Its combined preservation acceptance and primary publication remain separate requirements.

## Accepted preparation and what remains

| Surface | Current evidence | Remaining requirement |
| --- | --- | --- |
| Shared data | Revised 500k / 4,096 views; only 6 / 52 groups changed; complete independent reconstruction | Formal v3 deployment; no semantic-decontamination claim |
| Training core and identity | Actual short full-model GPU/maximum-context/continuation checks; sealed payload readers | One assembled released checkout and all full-horizon primary runs |
| Validation / outcomes | Real bounded scorer checks, complete-grid admission, independent statistical checks and full synthetic portable raw-outcome replay | Actual full validation and 840 primary BEIR units |
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

The post-acceptance twelve-run rerun question is now answered by the owner's
September 7 instruction: confirm correctness and promptly train. The local
training-code commit exception remains unanswered; automatic continuation is not
approval. Preserve old draft parents and final-paper gates. A separate reviewed
training release is being prepared; do not silently edit old bindings or transition
the stopped controllers. Commit/merge/push restrictions remain until explicitly resolved.
GitHub writes returned 403; no retry or alternate credentials. This local WIP is not in a clean
remote clone. HF withdrawal is safety-rejected; no deletion or history erasure occurred.

The [artifact snapshot](CURRENT_PROGRESS.json) counts legacy files, not live jobs or accepted
scientific results. It must be refreshed from the live experiment directory. Automatic 60-checkpoint
coverage is distinct from the separate independent ten-run HF audit (1,010 files / 82,373,100,558 bytes).

## Exact next sequence

**The owner-prioritized primary launch supersedes the older ordering below.**
Use the assembled corrected source, finish local training-source freeze/release
under the outstanding commit exception, verify relocated data and GPU leases,
then start AdamW 3e-5 and Muon 3e-4 on the two four-GPU pools and complete all twelve
declared recipes. Preserve and remotely verify every stage. Do not wait for the
optional factorial Trainer or completed paper to begin the primary experiment.
All scientific acceptance and final-publication requirements remain later work.

1. Whole regression **21015 is terminal/exit 0**; its complete XML is preserved,
   reporting 3,065 tests and 3,056 serialized cases with no failures/errors/skips.
   Preserve the factorial-routing six source-with-batches files. Its 103-case
   corrected-source focus 64963 also passes. The older 5462 and 7103 have finished; do not
   poll or restart either as missing work. The count difference remains explicit. The
   exact-sensitivity generation and layout pass their bounded checks; the separate full
   exact reconstruction and validator also pass. Do not repeat completed parents. The original/
   functional publication readers 40446/64182 and validator 30075 already pass; do not
   poll or restart them. All older standalone/joint/publication-preparation checks are
   complete. Preserve every source and failed attempt.
2. Finish complete versioned strict manuscript-consumer integration. Its new document
   component now checks actual expansion and fresh compilation, not upstream scientific
   admission. Actual v3 routed-factorial and reviewed release are required dependencies;
   finish those real paths before exposing a complete outer consumer. The new local-role
   publication reader passes its first full synthetic reconstruction; this is not real
   primary admission or final publication. Follow its manuscript-consumer-boundary.md:
   the old entry also depends on historical discovery/causal contracts, so replacing
   two publication helpers is insufficient. Keep the original nine geometry features,
   separate exact-feature sensitivity, all functional inference rules and raw inference bytes.
   A synthetic reconstruction is not primary admission; neither publication draft nor old
   parent hashes authorize a release. Do not use producer-location fallback or bypass the
   existing pending manuscript checks. The actual manuscript remains unchanged.
3. Integrate the actual routed factorial Trainer and balanced 50K loader separately;
   the primary-only policy rejects the routed control. Named optimizer and sampler
   components pass their checks but do not supply actual Trainer hooks, complete
   run identity or deep whole-run admission.
   Complete the reviewed assembled-source/runtime/release-parent handoff without refreshing old
   hashes into false compatibility. Existing completed diagnostics need no repeated baseline runs.
4. Obtain outstanding authority/access, execute all twelve primary runs in new namespaces and
   verify durable checkpoints. Then complete actual validation/BEIR, analyses and crossed continuations.
5. Write the paper from those accepted findings, render figures/tables and verify all manuscript,
   clean-clone, reconstruction and distribution gates before authorized release.

## Evidence navigation

- [Complete original geometry reconstruction](reports/engineering-archive/dense-v3-geometry-reconstruction-v1/README.md)
- [Complete raw-vector reconstruction checks](reports/engineering-archive/dense-v3-vector-reconstruction-v1/README.md)
- [Complete original-outcome reconstruction checks](reports/engineering-archive/dense-v3-outcome-reconstruction-v1/README.md)
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
