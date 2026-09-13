# Current DenseOn experiment and handoff

Dated snapshot: **2026-09-12 15:18 UTC**. This is not a perpetual heartbeat.
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
| Functional feature computation | **25/61 native-verified** at 15:18:33 UTC; new CPU worker live |
| Genuine GPU calibration | **2/2 complete**, both GPU and both fresh CPU reader exits zero |
| Crossed continuations | **2 running / 0 of 12 complete**; first pair at **181/391 and 178/391** |
| Real first checkpoint verification | Both step-79 saves pass fresh native CPU model/moment/scheduler/four-rank-RNG reading |
| Paper / source release | Draft only; full results, portability and release still incomplete |

The new [actual execution/evidence report](reports/engineering-archive/dense-v3-real-experiment-advance-v1/README.md)
contains source, owner approval, calibration, real worker records, preserved
failures and independent first-checkpoint reads. These are not fixture results.
The latest [training](reports/engineering-archive/dense-v3-real-experiment-advance-v1/updates-1518/training.json)
and [feature](reports/engineering-archive/dense-v3-real-experiment-advance-v1/updates-1518/features.json)
observations are separately retained. Both step-157 seals are now present, but
only the first step-79 pair has received the additional independent read so far.
The first two full branches also provide actual production-topology/save/readback
verification; actual GPU resume-equivalence remains separate and unclaimed.

## What is running

| Work | Exact owned coordinator | Tool session |
| --- | --- | --- |
| Training pool a, tokens 4–7 | PID **776959**, start **319670514**, started 14:55:54 UTC | **57000** |
| Training pool b, tokens 0–3 | PID **777271**, start **319673599**, started 14:56:25 UTC | **45053** |
| CPU functional features | PID **779352**, start **319719253**, started 15:04:14 UTC | **47384** |

Both training pools run disjoint balanced six-cell queues. First pair:
`factorial-v3-adamw_state-adamw-seed314159` and
`factorial-v3-adamw_state-muon-seed314159`.
All eight exact training ranks are live at the snapshot. CPU feature worker is
PID **779428 / start 319720500**, child of 779352.

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

CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  reports/engineering-archive/dense-v3-real-experiment-advance-v1/functional/feature-preparation-history/observe_features.py \
  --output /tmp/dense-v3-feature-status-NEW.json
```

Do not restart a coordinator whose directory already exists. A failure stops
its own queue, preserves payloads and does not authorize overwriting/resuming.
Do not poll retired primary/baseline/validation/evaluation observer handles.

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
itself. Full functional and crossed-continuation evidence remains necessary.

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
**New continuation checkpoints, calibration and recovered functional data still
need their own immutable off-host backups.** Do not repeat completed prior backups
as missing work, or infer source publication from a local archive.

## Next work, in order

1. Continue the two genuine training queues. Confirm full 391-step/native complete
   reads, preserve every stage, and back up new checkpoints. Finish required
   genuine GPU recovery verification without relabelling the completed CPU checks.
2. Finish the active CPU feature matrix, then compute/reconstruct the fixed
   nine-contrast task-family inference and four named functional predictors.
   Keep all rates/stages and unsupported/undefined findings.
3. Connect the genuine continuation checkpoints to the declared probe and full
   168-task evaluation. Use released capacity without displacing an active
   training pool or violating its dual leases. Do not change the locked outcomes.
4. Complete source-bound scientific consumption, portable reconstruction,
   generated paper results/figures and final paper/reproducible-source release.
   Keep generated pending markers until actual gates pass; do not hand-edit claims.

The full goal remains active. No protected-helper access, old-controller
transition, GitHub write retry, HF deletion retry or scientific gate waiver is
authorized by these new operational entries. Code publication remains deferred
under the owner's September 8 instruction and the existing publication gates.
