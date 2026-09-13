# Complete-outcome statistical handoff is queued

At **2026-09-12 17:45:43 UTC**, the inference waiter was live/S:
session **4529**, PID **813139 / start 320676124**, started at 17:43:30 UTC.
Neither real result collector has started. No factorial statistical result or
manuscript installation is claimed.

The actual producer waits for all **168 full-corpus BEIR results** and
**60 checkpoint probes / 840 task rows**, from both complete original pools.
It then reads the two evidence families in separate processes, using their
unchanged native numerical/source namespaces. This avoids mixing the original
primary evaluator with the separately copied probe source assembly.

The BEIR collector requires every actual exit-zero task worker, complete run
and input records, pinned model/task/runtime identities, and a fresh call to
the original native full-result reader after shared settings have finalized.
The probe collector requires every actual worker exit and complete pool,
then rereads all raw vectors and reconstructs every stored array, score and
six-metric task/overall summary through the unchanged original functions.
Neither collector launches GPU work or adopts partial/historical populations.

Only then are the unchanged source functions used to produce all six original
tables: 168 seed/task scores, four cell summaries, 126 seed/task contrasts,
three estimand summaries, 60 probe checkpoints and 840 probe task rows. The
three fixed estimands remain source-state, reset-operator and interaction.
Each uses the original **100,000-draw, seed-20260904, two-way seed/task cluster
bootstrap** and linear-percentile marginal 95% interval. These are three
marginal intervals, not a simultaneous family; interaction is not mediation.

A separate calculation must reproduce all 126 effects using exact rational
four-cell algebra, all four cell moments, and all three complete bootstrap
intervals using multiplicity weights and independently implemented percentiles.
Its numerical tolerance is 2e-15 and the original decisions must agree.
The original estimates are never replaced with the verifier's values.

## Checks and preservation

The final **29 synthetic/operational tests pass**, including all fixed table
cardinalities, independent full resampling, negative/zero effects, rejection of
missing/duplicated or altered cells, and no collector execution while waiting.
These are not real factorial outcomes. The first 28-test attempt had two errors:
the new waiting code had not distinguished missing future files from the old
generic helper's nonordinary-file error. Its entire source/test output is
preserved. The corrected new collector explicitly distinguishes absence while
retaining symlink, source, completion and numerical guards unchanged.

Real preparation exited zero (terminal `423821`) with upstream readiness
**false**, correctly. The real registered waiter started and remains alive,
but its actual BEIR/probe readers and statistical computation are still pending.
No live training, evaluator, probe source or manuscript was modified.

The accompanying training snapshot at **17:45:44 UTC** shows **6/12 complete**,
both fourth-pair runs live at **154/391 and 143/391**, and all eight exact ranks
present. All six completed runs' **30 checkpoints are HF-verified**. The newly
completed third-pair revisions are:

- AdamW state / Muon / seed271828: `0361434f8174ffac294861e7585989b84ab9beda`,
  verified at 17:29:37 UTC.
- AdamW state / AdamW / seed271828: `aea927abbe32dd56c2aeddf64b1f8ed0794d99e1`,
  verified at 17:32:57 UTC.

The evaluation and probe snapshots both show their two waiters live/S, no GPU
jobs and no failure receipt. They still await completion of their original
whole six-run training pools.

## Exact live entry

- Directory: `/tmp/dense-v3-factorial-summary.BQ08HjeP`.
- Source SHA: `4be265ce12cc649a595afaec9e5ef8b68a50eaa347ccb583d4d32ba3e100c5a1`.
- Collector SHA: `988c71fd9ad24576a017864ce9b99eeeb0ed6e5d48327483607b185fe4175366`.
- Authority SHA: `47485b8a16af1b0be419e67c4cf45ede08cdacbd01059e42357e16c97b9eb92c`.
- Tests SHA: `207ef057c74fc6173b3950a01fb705db42bb420bf156a66244284ebd4a6322e3`.
- Future output: `/root/embedding-optimizer-v3-experiment/analyses/dense-v3-factorial-inference-v1`.

Do not edit/restart this waiter. Read only its exact registered handles:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /tmp/dense-v3-factorial-summary.BQ08HjeP/observe.py --output /tmp/inference-status-NEW.json
```

The resulting scientific tables still require real-data interpretation, portable
publication reconstruction and the manuscript/source release gates. This
operational handoff neither waives those gates nor touches the protected helper,
old controllers, denied GitHub write or denied HF deletion.
