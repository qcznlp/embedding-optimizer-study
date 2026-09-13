# Complete first-stage configuration grid

The unchanged native reader finished at **2026-09-11 03:20:45 UTC** with
**24 / 60 complete fourteen-task checkpoint evaluations**: every one of the twelve
declared configurations at steps **782 and 3907**. The original pool-A six step-782
states are new in this observation. The previous eighteen records are unchanged
overlapping evidence, not additional runs or replications.

## First-stage results

Every row uses all fourteen original pinned full-corpus BEIR tasks with equal
weight. Values are nDCG@10 multiplied by 100. Rows follow optimizer/rate order,
not performance order. These are descriptive checkpoint scores, not a new
statistical comparison or score-based selection.

| Optimizer | Learning rate | Complete tasks | Step-782 macro score |
| --- | ---: | ---: | ---: |
| AdamW | 1e-6 | 14 / 14 | 54.2569 |
| AdamW | 3e-6 | 14 / 14 | 56.4409 |
| AdamW | 1e-5 | 14 / 14 | 58.2073 |
| AdamW | 3e-5 | 14 / 14 | 58.5774 |
| Muon | 1e-4 | 14 / 14 | 58.2296 |
| Muon | 3e-4 | 14 / 14 | 58.8121 |
| Muon | 1e-3 | 14 / 14 | 58.1859 |
| Muon | 3e-3 | 14 / 14 | 54.2492 |
| NorMuon | 1e-4 | 14 / 14 | 58.0118 |
| NorMuon | 3e-4 | 14 / 14 | 58.7539 |
| NorMuon | 1e-3 | 14 / 14 | 58.5261 |
| NorMuon | 3e-3 | 14 / 14 | 53.4834 |

All **36 checkpoints at steps 1563, 2345 and 3126 remain required**. Having the
first and final stages does not establish the full training trajectory, convergence
speed, seed robustness, useful dimensions or an optimizer mechanism. The
[previous final-checkpoint inference](../../dense-v3-final-inference-v1/README.md)
is unchanged; no new significance test, preferred-rate selection, interpolation
or manuscript finding is introduced here.

## Actual readback evidence

The [native bundle](actual/complete-checkpoint-readback.json) contains all
**336 raw task values** and their original worker/metadata snapshots. Its
5,015,697 bytes have SHA-256
ed537f9678dba6af8b32de3ad04a24a5a0275e6c2105b87bc727b546045630e0.

The [unchanged reader](source/readback.py), SHA-256
765f00465728ad6e18d299f2e064070d6cc525775a5bf1c7e71273c9497ca5b2,
authenticates both 56-file training assemblies, original dispatch authority,
training-completion parents and actual checkpoint artifacts. Each original
inspect_evaluation reread equals its complete-checkpoint receipt, with all
fourteen declared task identities/revisions/splits and matching original
exit-zero workers.

The [independent raw-score reconstruction](actual/independent-raw-reconstruction.json)
finished at **03:23:54 UTC**. It checks 336 raw nDCG values, 336 original successful
worker records, 384 raw-score/metadata snapshots and every exact rational mean.
All eighteen entries from the
[previous intermediate archive](../dense-v3-first-intermediate-evaluations-v1/README.md)
are identical as complete parsed records, including raw snapshots; that file's
original hash is unchanged. **Only six states and 84 task values are new.**

The CPU native reader exited zero without initializing CUDA or recomputing model
embeddings, rankings or retrieval scores. [commands.json](commands.json) retains
actual commands, terminal results and the reconciled native exits.
[verification.json](verification.json) binds this archive and the handoff update.
This is local preservation, not a new HF upload, committed source release or
second-host replay.

## Execution handoff

At the separately timestamped **03:26 UTC** observation, primary coverage is
**336 / 840**, all eight original evaluators are live and both pools have entered
step-1563 ClimateFEVER. No evaluator, numerical source, authorization, priority or
historical controller was changed. See [observations.json](observations.json).

The functional campaign briefly obtained a normally released GPU, accepted its
pretrained vector state, then terminated on a second-state receipt-path error.
Its exact **one accepted vector state, zero feature states, original failure and
recovery boundary** are documented separately in
[the functional handoff](functional-first-state/README.md). Do not report that
coordinator as still waiting or restart it automatically. The eight primary
evaluators continue unaffected.

The [before-copy](before/CURRENT_EXPERIMENT.md) preserves the preceding handoff.
All 840 primary cells, 61 functional states, twelve crossed-continuation branches
and the complete paper/reproducibility requirements remain in scope.
