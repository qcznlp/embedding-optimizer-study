# Project status and handoff

Last updated: 2026-09-05 18:35 UTC

This is the current decision page for the DenseOn-only NAACL project. Read [AGENTS.md](AGENTS.md)
before acting. Historical milestones and superseded deployment proposals are preserved in the
[byte-identical earlier handoff](reports/handoff-history/PROJECT_STATUS-20260905-0247.md); they are
not current launch instructions. The manuscript under `paper/` is the only publication deliverable.

## Full-model NorMuon check — 2026-09-05 18:35 UTC

The new [full-model reference probe](reports/engineering-archive/normuon-full-reference-v1/README.md)
narrows the epsilon finding: **352/352 hidden-matrix update comparisons are bitwise equal** to
the authenticated official functions on one real DenseOn checkpoint and the fixed first two
materialized training rows. All 88 hidden matrices are covered with saved/fresh state and loss
multipliers 1/0.25. Actual input lengths are 11–611 tokens; no truncation or model update occurs.
Both scale conditions activate clipping, so this is not evidence that quarter-scaling is harmless.

The small-matrix 44/96 nonconformance result remains valid, but **its impact on actual primary
training is not established**. All measured BF16 denominators in this full-model probe are equal;
do not generalize that to all gradients. The diagnostic reads a digest-authenticated independent
download, preserves every source payload and all model/optimizer state, and uses CPU only.

The duplicate-normalization failure and scientific hold remain decisive. No GPU handoff approval,
pause, deployment, primary retraining or HF withdrawal has occurred. Existing evaluation ownership
and the invalidity of the old zero-step migration remain unchanged. This is engineering provenance
only, not manuscript content or an optimizer-quality result.

## Code verification update — 2026-09-05 18:06 UTC

**The code is not cleared for scientific use.** Training has now finished **12/12 runs**, with
**60/60 resumable and backed-up checkpoints**. Main has completed its first four post-training
steps and is evaluating BEIR; factorial still waits for main. The older running/10-run counts
below are superseded by this update and the fresh artifact-only snapshot.

The [new code-correctness audit](reports/engineering-archive/dense-code-correctness-v1/README.md)
retains the decisive duplicate-normalization failure and adds an independently authenticated
NorMuon reference check: 44/96 small-matrix update cases fail conformance because local code
uses a clamped norm while upstream adds epsilon. The actual primary-training impact is unknown.
The existing isolated normalization candidate has not been deployed.

Twenty-four four-rank CPU checkpoint continuations (both Trainer variants, three optimizers,
dropout 0/0.1, two restart points) restore initial weights, optimizer, scheduler, consumed rows
and per-rank RNG exactly, using a disclosed CPU-only deserialization mapping adapter. Subsequent
calculation is not bitwise identical; one live NorMuon case has a row-second-moment difference
above the prior gradient-audit tolerance. The full 1,281-test regression suite and 25 new guard
tests pass, but neither overrides these failed/incomplete numerical acceptance checks.

The owner was asked whether to pause automatic evaluation for GPU correctness checks; no reply
or pause has occurred. No production source, checkpoint or controller was modified. Keep the
scientific hold; do not launch GPU checks into the active evaluator. The old seven-file main
migration requires an inactive **zero-step** ledger: those old preconditions no longer hold.
It must be revalidated for the completed-step/evaluation state after any authorized handoff.
Source publication, runtime deployment, retraining and HF withdrawal remain unapproved.

## Critical normalization finding — scientific promotion on hold

The new [four-rank CPU/Gloo audit](reports/engineering-archive/dense-ddp-accumulation-v1/README.md)
fails on the unchanged production Trainer: raw gradients are **0.25000001 of the independent
128-query reference**. Trainer and Accelerator each divide by four. Both installed library files
match their exact official release tags byte-for-byte. A paired, explicitly compensated CPU-only
diagnostic passes all six gradient/clipping checks for three optimizers; it is not a deployed fix.

The effect cannot be dismissed as a fourfold learning-rate change or guaranteed harmless scaling:
3,169 of 3,910 sparse logged gradient norms from ten complete runs fall in the derived interval
where the declared clipping behavior can change. This is not a measurement of retrieval impact.
Earlier data, payload, loss-formula and serialization passes retain their narrow validity, but they
do not certify the previously untested accumulated Trainer path. **Do not promote these runs into
paper conclusions pending resolution.** No automatic retraining, deletion or relabelling is authorized.

The earlier request to pause the last two runs was unanswered; both have since finished, and
automatic post-processing is evaluating BEIR. All checkpoints remain preserved. The current owner
question is whether to pause evaluation for GPU correctness verification. No pause has occurred.
The former post-processing-only handoff is not sufficient to resolve the training-stack issue.
Engineering findings stay out of the manuscript.

### Repair prepared and verified on CPU — not deployed

The [single-normalization candidate](reports/engineering-archive/dense-normalization-candidate-v1/README.md)
retains Trainer's actual accumulation count and removes Accelerator's second divisor. Both the
tiny fixture and a real, independently downloaded DenseOn checkpoint pass the actual four-rank CPU
audit for AdamW, Muon and NorMuon: three groups of 128/128/32 distinct queries, raw/clipped gradients,
actual learning-rate cadence and identical final weights across ranks. All 134 full-model parameter
tensors are checked; the maximum raw element error is 4.054e-6 under the unchanged tolerances.
The deliberately wrong constant-divisor control fails on the tail. Full-model source payloads remain
unchanged, and diagnostic optimizers start fresh: this is not primary optimizer-state resumption.

The candidate and 36 focused guards are isolated; the complete code regression suite passes 1,256
tests. These checks do not execute or certify GPU/BF16/maximum-length training and do not measure
the scientific impact of earlier updates. Live numerical source, controllers and checkpoints have
not been changed. The next required step is an authorized GPU-path check, followed by an explicit
replication/impact decision; the earlier pause request remains unanswered.

## Goal and scientific boundary

Determine how AdamW, Muon and NorMuon change the reached weight states, whether those changes alter
the functional use of embedding dimensions, and whether that explains retrieval outcomes.
The intended evidence sequence is **weight trajectory → dimension utility → retrieval**, followed
by a crossed continuation that measures endpoint contrasts for a fixed source-state/operator pair.
It does not decompose the primary-training gain. This is a hypothesis-driven structure, not a claim
that the chain is already established.

Only the current 12-configuration primary matrix and its audited follow-ups can supply paper
findings. Do not reuse invalidated exploratory scores, infer retrieval quality from update spectra
alone, or assert that AdamW has an inherently better next step. One seed and one base model also
limit generalization; four learning rates are not four independent random seeds.

The owner excludes all implementation-error, packing and padding narratives from the entire
manuscript, including the appendix. Engineering evidence stays in repository reports. No separate
blog and no new LateOn training or evaluation are in scope.

## Verified snapshot

The artifact-only [current snapshot](CURRENT_PROGRESS.json) is dated 18:06 UTC on September 5.
It is a durable milestone, not a heartbeat. The experiment host's local clock is UTC+8.

| Component | Verified state | Still required |
| --- | --- | --- |
| Primary execution | **12/12 runs finished:** four configurations for each optimizer | Resolve correctness findings before scientific promotion |
| Checkpoints | **60/60 resumable and remotely covered**; all twelve complete-run backups exist | Preserve all stages; backup is not scientific validity |
| Accumulated Trainer gradients | Live path **failed**; isolated repair passes tiny and full-model CPU groups 128/128/32 for all three optimizers | Authorized GPU verification, correction deployment and scientific-impact/re-run decision |
| Training/data integrity | New 18:09 UTC audit: all 60 stages pass deep payload checks; all 500,000 stored row identities agree; frozen bindings match | Numerical/GPU correctness is not implied by payload validity |
| Independent HF verification | **1,010 files / 82,373,100,558 bytes** match ten original immutable upload commits | Future checkpoints and follow-up artifacts |
| Raw weight/displacement geometry | **50 stages / 4,400 hidden-matrix records**, all ten completed runs | Final two runs, full subspace analysis and retrieval bridge |
| Primary retrieval | No complete 840-unit BEIR grid or outcome manifest | 12 runs × 5 stages × 14 tasks, validation selection and inference |
| Functional dimensions / crossed continuation | Implemented and tested; real primary outputs not produced | 61-state exports, dimension analyses, then frozen factorial |
| Paper | Narrower scientific wording and conceptual figure integrated locally; 1,202 tests pass | Approved deployment, real findings and final release |

All twelve runs now record step 3907 and all five scheduled checkpoints. The snapshot contains
zero fatal training-error markers; this is an artifact observation, not exhaustive runtime
certification. The existing post-training controller owns the eight-GPU evaluation handoff.
Do not start another matrix or an uncoordinated GPU diagnostic.

### Current evidence

- Both remaining runs now also have sealed step-1563 backups: NorMuon 1e-3 at
  `9119d9b288560cad403a05de93ba7e5f55373c85` and 3e-3 at
  `1c2cfec908ffc826aab8e81d1e00f76659932467`. Their original uploader receipts verify 34 files /
  2,702,861,785 bytes. This does not extend the independent ten-run audit or certify training semantics.

- Both remaining runs now have sealed step-782 backups, each with 17 digest-verified files:
  [NorMuon 1e-3](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints/commit/7d0b82a88daa2b38a29a438cfd1792a53272c578)
  and [NorMuon 3e-3](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints/commit/0d196dede6b5953d16414fb41efaec205830729d).
  Their uploader receipts cover 2,702,829,188 bytes in total. This additional coverage does not
  silently extend the separate ten-complete-run independent audit or its 50-stage geometry count.
- [Ten-run integrity audit](reports/experiment-integrity/primary-audit-ten-run.json): repeats the
  full 500k-row linkage check and deep validation of all 50 completed checkpoints. It does not
  re-download upstream texts or numerically resume every checkpoint.
- [Ten-run HF content audit](reports/experiment-integrity/huggingface-digest-audit-ten-run.json):
  no missing/extra files, size or digest mismatches; local payloads remain stable.
- [Ten-run geometry archive](reports/experiment-integrity/primary-geometry-ten-run-archive.json):
  102 unique model/metadata inputs and 60 output files / 10,146,027 bytes verified. The unchanged
  frozen CPU entry point added both completed NorMuon trajectories. The additions-only
  [HF commit](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/commit/489c606076b0f8a0ef82adc7b85f1fb29da794bc)
  adds 12 files / 2,030,743 bytes and preserves all 48 previously archived geometry files.
  Anonymous checks verified the ten underlying NorMuon weights at their original public commits.
  No old path, repository card or history was modified.
- [Integrity report and scope](reports/experiment-integrity/README.md): also records real CPU
  loss/gradient/encoding checks and three-checkpoint optimizer-state continuation checks.
  These are not distributed-training, GPU mixed-precision, maximum-length or optimizer-quality
  results. Passing engineering tests does not complete the scientific experiment.
- [Independent checkpoint restoration](reports/experiment-integrity/checkpoint-download-normuon-3126.json):
  a forced HF download of NorMuon 3e-4 step 3126 verifies all 18 files / 1,351,464,827 bytes.
  Offline encoding and CPU serialization/continuation of all 134 parameters and optimizer states
  pass in a fresh directory without reading a live checkpoint. This is the same physical host,
  not a four-GPU resume or benchmark. The new [restoration guide](docs/checkpoint-restoration.md)
  replaces the broad historical-download instructions. That restoration milestone passed 1,150
  tests; its receipt and prior 1,124-test recovery receipts remain unchanged. New source is still isolated.

### Scientific claim review — integrated locally, not deployed

The [factorial claim review](reports/paper-review/factorial-claims-v1/README.md) reproduces three
synthetic counterexamples with the unchanged summary/bootstrap. The current automatic wording
can call effects additive despite a significantly negative interaction, imply beneficial
co-adaptation from positive interaction alone, or infer state-effect dominance from an inconclusive
operator estimate. Those inferences are not justified by the frozen contrasts.

The narrower [manuscript/renderer integration](reports/paper-review/factorial-claims-v2/README.md)
now reports the actual endpoint estimands, distinguishes calibration-probe matching from a realized
first update, and retains every estimate, interval, threshold and decision label. The conceptual
figure is aligned and checked for box-text overflow; its development placeholder contains no
invented scores. **1,202 full-suite tests pass**, including 41 claim/figure checks and a complete
explicitly synthetic paper build. The real pending draft ends main text on page 6, with an expanded
abstract bound of 178/200 words and no Type 3 fonts or prohibited implementation narrative.

An exact fourth publication-only preparation layer preserves the previous three layers, all 47
commands and the same main gate. Prior source bytes and the first candidate's 34-test receipt remain
archived unchanged. No runtime deployment or source publication occurred. Strict paper release
still correctly fails because actual primary/dimension publications and three scientific includes
are pending; these mathematical/textual checks are not optimizer findings.

## Execution and approval boundary

The live checkout is `/root/embedding-optimizer-study`, main commit
`f231a6430712388778f32ad1736a4cb6de3bec3e`. The isolated paper/analysis work is
`/root/embedding-optimizer-story-refactor`, branch `narrative/weight-space-spine`.
Set `PYTHONPATH` explicitly to the chosen checkout's absolute `src`; the installed package can
otherwise resolve to live main. Never merge the whole isolated branch into a running finalizer.

At 05:32 UTC, the main ledger is waiting for training (10 complete, ten backups, zero
post-training steps); the factorial ledger is waiting for main (zero steps). The independent
sealed-backup supervisor covers 52/60 checkpoints with zero cycle failures.

| Contract | Running on the experiment host | Prepared, not deployed |
| --- | --- | --- |
| Main post-training controller | `4152531e...` | Seven-file recovery target `4380a363...` |
| Dimension/factorial successor | `6605090d...` | Local `41fdf48c...`; host projection `7aafd7af...` |

### Critical execution finding

Three engineering defects affect future post-training work: evaluation provenance consumption,
table-header rendering, and interrupted-controller resume/log preservation. The
[exact seven-file repair and migration runbook](reports/engineering-archive/main-resume-v1/README.md)
preserves the original 17 commands and scientific settings. Its migration defaults to read-only,
requires the exact inactive waiting/zero-step ledger and a free existing lease to apply, and
archives original evidence. The four-layer successor amendment preserves all 47 prepared commands.

The earlier [recovery validation](reports/experiment-integrity/main-recovery-successor-validation.json)
records **1,124 full-suite tests, 81 focused checks and 116 documentation/contract checks passing**.
That earlier receipt is unchanged; it is not a test run of future edits. The actual successor dry
run launches nothing because main is incomplete. No live source deployment, migration or controller
takeover has occurred.

The [current publication-wording validation](reports/paper-review/factorial-claims-v2/validation.json)
binds the integrated source, 1,202-test run, 160-check documentation refresh and four-layer projection.
Its read-only controller dry run still finds main incomplete. This changes neither the earlier
repair's evidence nor the runtime approval boundary.

The new training/post-processing pause request is unanswered. Three earlier, separate approvals
also remain unresolved:

1. Narrow post-processing controller handoff to deploy the validated repair. An owner question was
   sent; continued goal execution is not approval. Training and `gpu.py` must remain untouched.
2. Publishing isolated work-in-progress source while the current repository's pre-commit rule
   forbids pending manuscript results. Do not self-relax that rule or push this branch unapproved.
3. [Erroneous HF artifact withdrawal](reports/hf-obsolete-cleanup/README.md): the rejected plan
   selects 9,629 files / 366,252,912,201 logical bytes. No deletion occurred. Do not retry or split
   it to bypass exact-scope confirmation. Historical erasure additionally requires preserving
   shared objects and migrating pinned current-backup dependencies; refresh the now-stale
   dependency audit before any approved removal.

## Safe continuation order

1. Resolve the new normalization finding and requested safe pause before promoting any results.
   Until the owner answers, follow only artifacts/exact handles and preserve backups; do not signal
   jobs, launch new evaluation, re-train, or modify the existing controllers. The original progress
   entry point reports execution artifacts, not a scientific validity decision.
2. The [training/analysis runbook](docs/dense-no-packing-retrain.md) permits incremental frozen
   CPU geometry for newly complete runs, with partial status and no final summary. Independently
   verify and preserve new checkpoint/analysis payloads. Do not invent interim optimizer verdicts.
   The claim-wording integration is now prepared under its exact fourth publication-only layer.
   Preserve numerical decisions and all 47 successor commands; this work grants no runtime
   deployment or WIP publication authority.
3. After explicit handoff approval, recheck live identities and leases and follow the narrow
   seven-file migration runbook. Do not substitute an older header-only, six-file or JSON-only
   transition. Do not signal training jobs or deploy from uncommitted scientific source.
4. The main finalizer requires 12 deeply complete runs before validation, all 840 BEIR task units,
   full geometry/bridge/outcome analysis and its 17-step completion gate. Early evaluation on a
   released pool requires a single-owner handoff: manual workers do not share the current main
   controller's eight-GPU ownership. Two runs remain, so the owner's last-run condition is not met.
5. After the exact repaired main is fully complete, perform the approved zero-step successor
   transition and recompute the host contract. Follow [dimension utilization](docs/dimension-utilization.md)
   and [state-by-operator continuation](docs/state-operator-factorial.md): eleven primary/dimension
   steps then the unchanged original 36 factorial steps. No historical exports as substitutes.
6. Render actual findings and verify complete source-bound evidence, portable reconstruction,
   inference and manuscript release gates. Commit/push only under the repository's approval rules.
   A draft build or synthetic complete-results fixture does not satisfy scientific release.

Never inspect, read, edit, signal, stop, replace or otherwise touch `gpu.py` or its processes.
Do not inspect broad process lists, launch duplicate controllers, relax frozen gates, overwrite
existing checkpoints, print credentials, restart LateOn, or import implementation incidents into
the paper.

## Where to continue

- Operational and evidence index: [README.md](README.md), [integrity audits](reports/experiment-integrity/README.md).
- Frozen primary training: [execution](configs/dense_no_packing_execution_protocol.json),
  [preflight parent](configs/dense_no_packing_preflight_protocol.json),
  [evaluation](configs/dense_no_packing_evaluation_protocol.json),
  [analysis](configs/dense_no_packing_analysis_protocol.json),
  [outcomes](configs/dense_no_packing_outcome_protocol.json).
- Scientific manuscript plan: [Dense-only NAACL plan](docs/naacl-dense-paper-plan.md),
  [paper build and release](paper/README.md).
- Public progress: [GitHub issue #41](https://github.com/qcznlp/embedding-optimizer-study/issues/41);
  live curves: [W&B](https://wandb.ai/stevezenguom/embedding-optimizer-study).
- Remote preservation: [checkpoints](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints),
  [analysis artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts).
- [Handoff history](reports/handoff-history/README.md) preserves the previous long status page
  byte-for-byte. It is an audit trail, not an alternative current execution plan.
