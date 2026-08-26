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

For the preregistered high-resolution spectrum tier, select exact checkpoints and tensors instead
of materializing every matrix. The following example computes full singular spectra because rank 768
reaches the smaller dimension of every selected matrix. Within this selected sequence,
`delta_from_previous` means displacement from the previous selected checkpoint.

```bash
embed-optim-geometry \
  --run-dir outputs/dense/muon-lr1e-4 \
  --reference /path/to/pinned/DenseOn-unsupervised \
  --output-dir results/weight-space-spectra/dense/muon-lr1e-4 \
  --steps 782 2345 3907 \
  --tensor-regex '^0\.layers\.(0|10|21)\.(attn\.Wqkv|mlp\.Wi)\.weight$' \
  --sketch-rank 768
```
