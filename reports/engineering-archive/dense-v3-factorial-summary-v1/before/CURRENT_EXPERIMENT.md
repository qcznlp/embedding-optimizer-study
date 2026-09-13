# Current DenseOn experiment and handoff

Dated snapshot: **2026-09-12 17:28 UTC**. This is not a perpetual heartbeat.
Read [AGENTS.md](AGENTS.md) for safety/authority boundaries. The previous long
handoff is preserved [unchanged in the new evidence archive](reports/engineering-archive/dense-v3-real-experiment-advance-v1/before/CURRENT_EXPERIMENT.md);
its older waiting messages and completed handles are historical.

## Goal and current outcome

Deliver a defensible **NAACL paper and reproducible repository/model-analysis
artifacts** on how AdamW, Muon and NorMuon change DenseOn weight states,
functional representation utility and retrieval. Dense only; no LateOn or blog.

The owner explicitly said **“你有权做一切事情，目标是尽快完成任务”**.
Recovery approval is answered. Real experiments are now advancing under new
source-bound entries. Do not repeat completed preparation or ask for that
approval again. The protected helper, stopped historical controllers, original
evidence, scientific definitions and external safety denials remain protected.

| Component | Actual verified status |
| --- | --- |
| Corrected primary training | **12/12 complete**, all five stages, **60 checkpoints** |
| Primary full-corpus BEIR | **840/840 complete**; pretrained baseline **14/14** |
| Primary validation selection | **12/12 complete**, fixed validation-only selection |
| Primary weight geometry/update map | Both approximate and exact branches **60/60 complete** |
| Native training dynamics | All **4,692 observations / 60 stages** read and replayed |
| Weight-to-retrieval prediction | Original 9 + separate exact 5 features, all 60 states, complete |
| Post-result comparator sensitivity | All **84 comparisons / 5,040 predictions**, complete and labelled exploratory |
| Functional vectors | **61/61 complete**, original full native readback at 14:43:34 UTC |
| Functional feature computation | **61/61 complete**, full native per-state recomputation and readback |
| Functional inference | **9 primary + 27 rotation contrasts, 240 predictions**; separate actual-data independent verification complete |
| Functional post-result controls | **24 comparisons / 1,440 predictions**, independently verified; no feature passes all four recipe baselines |
| Genuine GPU calibration | **2/2 complete**, both GPU and both fresh CPU reader exits zero |
| Crossed continuations | **6/12 complete**; fourth pair at **15/391 and 3/391**, eight exact ranks live |
| New checkpoint verification/backup | First **20/20** native deep reads and immutable HF metadata/hash checks complete; backup watcher active |
| Continuation full-corpus BEIR | Two complete-pool waiters active; **0/168** tasks, no GPU lease requested yet |
| Continuation five-stage probe | Two complete-pool waiters active; **0/60** new checkpoint encodings, one native-verified reused reference complete |
| Functional numerical portability | All nine table families, all decisions and generated TeX reconstructed exactly from copied source/data |
| Functional/calibration durability | **624 files / 7.5 GB** remotely verified, anonymously downloaded and independently rehashed |
| Paper / source release | Draft only; full results, portability and release still incomplete |

The new [actual execution/evidence report](reports/engineering-archive/dense-v3-real-experiment-advance-v1/README.md)
contains source, owner approval, calibration, real worker records, preserved
failures and independent first-checkpoint reads. These are not fixture results.
The [latest training observation](reports/engineering-archive/dense-v3-factorial-probe-v1/entry/training-observation-1730.json)
and [completed functional result chain](reports/engineering-archive/dense-v3-functional-result-chain-v1/README.md)
retain actual process exits, original failures and successful independent readers.
The first six full branches provide actual production-topology/save/readback
verification; actual GPU resume-equivalence remains separate and unclaimed.

## What is running

| Work | Exact owned coordinator | Tool session |
| --- | --- | --- |
| Training pool a, tokens 4–7 | PID **776959**, start **319670514**, started 14:55:54 UTC | **57000** |
| Training pool b, tokens 0–3 | PID **777271**, start **319673599**, started 14:56:25 UTC | **45053** |
| Completed-run HF backup | PID **787110**, started 15:46:52 UTC | **16308** |
| Final BEIR waiter a | PID **802691**, start **320351837**, started 16:49:37 UTC | **41515** |
| Final BEIR waiter b | PID **802714**, start **320351956**, started 16:49:38 UTC | **37455** |
| Five-stage probe waiter a | PID **809602**, start **320566873**, started 17:25:20 UTC | **54795** |
| Five-stage probe waiter b | PID **809603**, start **320566875**, started 17:25:20 UTC | **34578** |

Both training pools run disjoint balanced six-cell queues. Current fourth pair:
`factorial-v3-muon_state-adamw-seed271828` (pool a) and
`factorial-v3-muon_state-muon-seed271828` (pool b).
All four seed-314159 cells and both AdamW-source seed-271828 cells are complete.
All eight current exact training ranks
are live at the snapshot. CPU feature session **47384** is terminal/exit 0.
Inference successor **37685** and independent verifier **49034** are also
terminal/exit 0; original inference **58754** is preserved terminal/exit 1
after its complete 61-state prefix. Do not poll or relaunch those consumers.

The original vector-recovery coordinator **27585** is terminal/exit 1, after
completing every vector and encountering a CPU feature-save path failure.
Its vector completion is accepted; its partial feature attempt remains intact.
Do not restart it or re-encode vectors. The separate new CPU-only entry repairs
the output-parent preparation without changing scientific computations.

### Safe observation

Use the archived read-only observers with a **new, non-existing absolute output
path** on this experiment host. They bind only these new coordinators and their
exact children; they do not launch work or enumerate processes/GPUs.

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  reports/engineering-archive/dense-v3-real-experiment-advance-v1/training/preparation-and-observations/observe_training.py \
  --output /tmp/dense-v3-training-status-NEW.json

```

Do not restart a coordinator whose directory already exists. A failure stops
its own queue, preserves payloads and does not authorize overwriting/resuming.
Do not poll retired primary/baseline/validation/evaluation observer handles.
For the new BEIR waiters only, use the read-only
`/tmp/dense-v3-factorial-evaluation.IOV7MK93/observe.py --output /tmp/NEW.json`.
The source-bound evaluation entry and actual input probe/21 operational tests
are preserved in the [latest handoff archive](reports/engineering-archive/dense-v3-close-loop-handoff-v1/README.md).
The [new five-stage probe archive](reports/engineering-archive/dense-v3-factorial-probe-v1/README.md)
preserves its live entry, all 66 unchanged numerical sources, actual inputs and
19 final operational/numerical checks. Its narrow observer is
`/tmp/dense-v3-factorial-probe.ryorjXJK/observe.py --output /tmp/NEW.json`.
Both probe waiters use at most one token per wholly completed training pool;
they do not displace training or alter the live BEIR entry. Do not restart them.

## Fixed continuation experiment

Two genuine primary step-2345 sources, AdamW 3e-5 and Muon 3e-4, each receive
reset routed AdamW and Muon under order seeds **314159, 271828, 161803**:
**2 states × 2 operators × 3 seeds = 12** complete branches.

Every branch uses the same intact **50K** groups, one positive/seven negatives,
no in-batch negatives, temperature .02, maximum length 8192, four GPUs ×
microbatch 8 × accumulation 4, global batch 128 and final 80-group tail.
There are **391 steps**, **40 warmup steps**, linear decay, clipping 1,
weight decay .01 and checkpoints **79, 157, 235, 313, 391**.
FP32 parameters and the unchanged BF16/FA2 numerical policy are retained.
Auxiliary parameters use AdamW LR **3e-6**. No NorMuon crossed branch is declared.

| Source | AdamW hidden LR | Muon hidden LR |
| --- | ---: | ---: |
| AdamW state | 0.00015039350105964107 | 0.0018331886661728941 |
| Muon state | 0.00015137826846307303 | 0.0018296038299112601 |

Rates come from the actual original fixed-probe calibration of all 88 hidden
matrices, targeting global update/weight **5e-4**, excluding weight decay.
They are not hand-tuned against retrieval and do not imply equal updates
throughout training. Calibration is not an optimizer-quality result.

Each full branch must pass real four-rank completion and a separate native CPU
whole-run read. All twelve final checkpoints then need **168 full-corpus BEIR
task evaluations**, plus the declared five-stage probe outcomes. The three
fixed final contrasts are source-state, reset-operator and interaction effects;
positive interaction alone is not within-state benefit, dominance or mediation.

## What the completed primary evidence supports

The primary four-rate-average optimizer comparisons remain inconclusive:
all three simultaneous intervals include zero. Validation-selected endpoint
macro nDCG@10 points are **58.9263 AdamW, 59.2431 Muon, 59.3637 NorMuon**
(pretrained **51.2961**). Selected NorMuon–AdamW is **+0.437** points with its
secondary task-level simultaneous interval **[+0.146, +0.729]**; the selected
Muon–AdamW interval includes zero. Learning-rate cells are not independent seeds.

Selected Muon/NorMuon point estimates exceed selected AdamW at all five stages;
their 40% points exceed AdamW's final point. This is descriptive, not a wall-time
saving claim or cross-seed mechanism finding. Training time was similar across
optimizers; lower optimizer-state memory is a distinct engineering measurement.

The declared weight-prediction baseline supports displacement and some entropy
predictors, but the [post-result sensitivity](reports/dense-v3-predictor-sensitivity-v1/README.md)
shows comparator dependence: no original feature passes all four tested recipe
baselines. Neither spectrum shape nor a one-step proxy explains retrieval by
itself. The [complete functional inference](reports/dense-v3-functional-inference-v1/README.md)
finds only one supported primary contrast: Muon increases native helpful
participation by **0.00358**, but its direction reverses in two of three rotations.
Helpful participation does not improve held-dose retrieval prediction. Degrading
attribution mass does improve the declared baseline in **4/4 folds** (RMSE
**0.02905 to 0.01898**) with a **positive** residual association. The corresponding
optimizer contrast is inconclusive: this is not evidence that Muon wins by
reducing harmful coordinates. The now-complete [functional controls](reports/dense-v3-functional-sensitivity-v1/README.md)
show that this predictive signal also depends on the baseline. No functional
feature passes all four recipe comparators. Crossed continuation outcomes
remain necessary before proposing an explanation.

Scientific narrative: **retrieval differences → reached weight states and
trajectories → functional utility → held-out-dose prediction**, complemented
by the bounded state-by-reset-operator continuation. Do not select a preferred
story, equate prediction with mediation, infer arbitrary-basis invariance from
three rotations, or put implementation-error narrative anywhere in the paper.

## Exact locations and preservation

- Development/paper: `/root/embedding-optimizer-story-refactor`.
- Original completed primary assembly: `/root/embedding-optimizer-primary-v3`.
- Unchanged 70-file continuation assembly:
  `reports/engineering-archive/dense-v3-factorial-worker-v1/actual/source-final`.
- New training entry: `/root/embedding-optimizer-v3-experiment/launch/factorial-training-v1`.
  Source SHA `d11d3281523b1573ce7fcd650e10c538cfa84022b5db92117559d47213e57e53`;
  authority SHA `00c52f0bec979fb163a734ad0f9eadcd6ea09f7393133fd35c65bc62878ce11f`.
- New training outputs: `/root/embedding-optimizer-v3-experiment/outputs/dense-v3-state-operator-v1`.
- Complete calibration: `/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-factorial-calibration-v2`.
- Complete vectors: `/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-functional-dimensions-recovery-v1/vectors`.
  Manifest SHA `9bf90d652d54ee6b9b571bf9dc88cd919da8c78bb70db0d476e307733e8c23e7`.
- New CPU feature entry/output:
  `/root/embedding-optimizer-v3-experiment/launch/functional-features-recovery-v3`,
  `/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-functional-features-recovery-v3`.

Primary checkpoints and completed prior scientific tables/raw scores are already
backed up. Use [checkpoint restoration](docs/checkpoint-restoration.md),
[weight restoration](docs/weight-analysis-restoration.md),
[trajectory restoration](docs/retrieval-trajectories-restoration.md) and
[prediction restoration](docs/predictor-analysis-restoration.md).
The first four new runs (twenty checkpoints) are remotely verified in
`qcz/embedding-optimizer-study-checkpoints`, revisions
`a219dcab85bda040e9d46f6f7ec18ee1b0acf598` and
`e9ab737562e50a6ec0ff838f22e97301fda93d63`,
`0f464bbcc98760bbcffbe78b8c5af45c87a60d12` and
`32541a1533957c9507c675884db150961f98eae0`. The watcher backs up each of
the twelve branches after that branch's native whole-run completion. Its entry is
`/tmp/dense-v3-factorial-backup-launch.75Chtt86/backup.py`; do not restart it.
**Calibration and recovered functional payloads are now fully backed up and
actually downloaded/verified.** Use [functional restoration](docs/functional-analysis-restoration.md):
all 624 files / 7,496,655,065 bytes at immutable dataset revision
`9b182d77b31c93277457a578c16389935bc3fdb9`. The full download is not a
source release or a physical second-host GPU-resume test.
Do not repeat prior backups or infer source publication from a local archive.

## Next work, in order

1. Continue the two genuine training queues. Confirm full 391-step/native complete
   reads, preserve every stage, and back up new checkpoints. Finish required
   genuine GPU recovery verification without relabelling the completed CPU checks.
2. Functional inference, its numerical replay and all-four-feature post-result
   recipe-control sensitivity are complete and independently verified. The
   complete raw/results backup is verified. Do not repeat these completed jobs.
   Retain all unsupported and undefined findings.
3. The **168-task BEIR and 60-checkpoint probe waiters are already live**;
   do not create duplicate queues. Verify their real workers/results as they
   run, then connect full outcomes to the three unchanged factorial estimands.
   Use released capacity without displacing an active
   training pool or violating its dual leases. A pool's entire six-cell queue
   must complete before evaluation uses it. Do not change the locked outcomes.
4. Complete source-bound scientific consumption, portable reconstruction,
   generated paper results/figures and final paper/reproducible-source release.
   Keep generated pending markers until actual gates pass; do not hand-edit claims.

The full goal remains active. No protected-helper access, old-controller
transition, GitHub write retry, HF deletion retry or scientific gate waiver is
authorized by these new operational entries. Code publication remains deferred
under the owner's September 8 instruction and the existing publication gates.
