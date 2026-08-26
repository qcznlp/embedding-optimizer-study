# Exact checkpoint-trajectory summaries

These tables summarize the exact-statistics tier of the weight-space analysis over all 24 runs and
120 checkpoints:

- `checkpoint_trajectory.csv`: one row per run/checkpoint;
- `run_trajectory_summary.csv`: final displacement and the five-checkpoint coarse path per run;
- `summary_manifest.json`: coverage, shared data fingerprint, source-manifest hashes, and output
  hashes.

The source records used `--sketch-rank 0`: Frobenius norms, row/column balance, Gini coefficients,
energy concentration, and checkpoint displacements are exact, while singular-spectrum fields are
disabled. Full or randomized spectra are a separate analysis tier.

`coarse_checkpoint_path_length` joins the pretrained initialization and five saved checkpoints. It
is not the optimizer's per-step path length, and checkpoint displacement is not an optimizer update.
Use common-state gradient/update probes before making causal claims about AdamW, Muon, or NorMuon.

Regenerate the tables with:

```bash
embed-optim-summarize-geometry \
  --geometry-root results/weight-space \
  --output-dir reports/weight-space \
  --verify-inputs
```
