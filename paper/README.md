# NAACL manuscript

This directory contains the result-safe manuscript for the optimizer study. It uses the official
ACL style files pinned to commit `d5adc823ff0f80f98c80405ca0ab66c68e684409` of
[`acl-org/acl-style-files`](https://github.com/acl-org/acl-style-files). The style files are fetched
into an ignored local directory so this repository does not silently fork the conference template.

Build the review-format PDF with:

```bash
cd paper
make
```

`results.tex` is deliberately the only place where numerical result macros may enter prose. A
macro containing `\ResultPending` marks an unresolved completion gate and renders visibly in red.
The prospectively frozen headline and interpretation rules live in
`../configs/paper_claim_protocol.json`. They bind the experiment and intervention protocols by hash,
require all family/optimizer contrasts, and define when a confirmatory interval permits positive,
negative, or only inconclusive language.
If a bound narrative source needs a factual documentation correction, the protocol retains the
original freeze context and records a content-hashed amendment with the evidence visible at the
time; the audit exposes that amendment and rejects an unrecorded source change.
The paper is not submission-ready while any such marker remains.

Audit the checked-in constants against the frozen matrix, materialized dataset, and strict
weight-space manifest with `embed-optim-audit-paper`. The command reports unresolved evidence during
drafting; `embed-optim-audit-paper --strict` becomes a hard final-submission gate.

After every strict evidence report exists, run `embed-optim-render-paper-results`. It replaces only
the five headline macros, writes `reports/paper-results.manifest.json`, and binds the resulting
`results.tex` bytes to the claim protocol, all evidence manifests, and all source tables. The strict
paper audit rejects a manual headline edit or a stale generated manifest.

`make -C paper` also invokes this renderer with `--if-ready`: an incomplete experiment retains the
visible audited draft placeholders, while a complete evidence matrix renders the frozen headlines
before LaTeX compilation. This keeps an already-running final handoff compatible with the same
completion gate without weakening failures caused by malformed or stale complete evidence.

## Evidence contract

| Claim family | Authoritative evidence | Required gate |
|---|---|---|
| Discovery training and systems behavior | canonical Trainer/W&B histories, completion records, and both manifests under `reports/training-dynamics/` | 24 runs, 120 checkpoints, 9,384 history rows, six systems summaries, and two source-bound figures pass audit |
| Discovery retrieval behavior | `reports/coverage.json`, strict aggregate tables and plot sidecars | exactly 1,680 decontaminated-BEIR units |
| Discovery time-to-quality | `configs/retrieval_dynamics_protocol.json` and `reports/retrieval-dynamics/summary_manifest.json` | 120 checkpoint means and all 1,680 source task files pass hashes; first passage uses the prospectively locked AdamW-median rule with right censoring and discloses its 160/1,680-unit freeze timing |
| Integrated weight trajectories | `reports/weight-space/summary_manifest.json` | 24 runs, 120 checkpoints, verified model inputs |
| Common-state update geometry | `reports/common-state/summary_manifest.json` and exact-spectrum manifest | 20 anchors, 1,760 gradients, 5,280 transforms, 360 spectra |
| Representation and score geometry | strict manifests under `reports/representation-space/` | two pretrained references plus 120 checkpoints per probe tier |
| Immediate causal intervention | `reports/functional-intervention/manifest.json` | 20 anchors and 58,240 paired sample records |
| Accumulated causal branch | `reports/short-branch/summary_manifest.json` | 18 runs, 90 checkpoints on both frozen probes |
| Routing fairness | `reports/hybrid-adamw/summary_manifest.json` | eight hybrid runs and 112 final BEIR units |
| Confirmatory retrieval claims | `reports/confirmatory/summary_manifest.json` | three new seeds, 18 runs, 252 final BEIR units |
| Final outcome rendering | `reports/outcome-summary.manifest.json` | all routing, immediate-direction, accumulated-branch, and confirmatory tables are source-hashed into the final blog marker region |

Checkpoint correlations are always described as observational. Causal language is reserved for
same-state interventions and shared-start branches. The exploratory seed is never pooled with the
three confirmation seeds as though all four were prospectively sampled.
