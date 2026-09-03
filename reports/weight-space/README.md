# Exact checkpoint-trajectory summaries

These tables summarize the exact-statistics tier of the weight-space analysis over all 24 runs and
120 checkpoints:

- `checkpoint_trajectory.csv`: one row per run/checkpoint;
- `run_trajectory_summary.csv`: final displacement and the five-checkpoint coarse path per run;
- `optimizer_pair_contrasts.csv`: matched-learning-rate final Muon/NorMuon displacement and row-energy
  ratios;
- `optimizer_pair_contrast_trajectory.csv`: the same strict contrasts at all five checkpoints;
- `summary_manifest.json`: coverage, shared data fingerprint, source-manifest hashes, and output
  hashes.

The source records used `--sketch-rank 0`: Frobenius norms, row/column balance, Gini coefficients,
energy concentration, and checkpoint displacements are exact, while singular-spectrum fields are
disabled. Full or randomized spectra are a separate analysis tier.

## Descriptive Muon/NorMuon signal

Across all eight same-learning-rate pairs at step 3,907 (two model families by four learning
rates), the NorMuon-to-Muon pretrained-reference displacement ratio is 1.000668–1.003879. At nearly
the same displacement scale, the parameter-weighted row-norm CV ratio is 0.232758–0.463783 and the
top-1%-row energy ratio is 0.659264–0.730320. Both model families show the same direction. This is
consistent with NorMuon redistributing trajectory energy across rows rather than merely shrinking
the overall displacement.

The five-checkpoint table contains 40 matched pairs. Across the complete trajectory, displacement
ratios stay within 0.995607–1.003879, row-norm CV ratios within 0.166608–0.463783, and top-1%-row
energy ratios within 0.585108–0.730320; the direction never reverses.

In the narrow displacement range shared with AdamW, Muon is not more row-balanced: the nearest
descriptive DenseOn points have row CV 0.1972 (Muon) versus 0.0951 (AdamW), and LateOn has 0.2012
versus 0.1011. NorMuon is 0.0894 and 0.0933, respectively, at almost the same displacement as its
Muon counterpart. These post-hoc checkpoint matches are not causal comparisons. They motivate
separating Muon's singular-spectrum hypothesis from NorMuon's row-balancing hypothesis.

![Matched Muon and NorMuon checkpoint geometry](optimizer_pair_contrast_trajectory.svg)

Regenerate the vector figure with:

```bash
embed-optim-plot-geometry
```

The complementary phase plot includes AdamW and uses relative displacement on a logarithmic axis.
The gray band is only the scale range observed for all three optimizers; it is not a matched causal
experiment.

![All-optimizer checkpoint geometry by displacement scale](optimizer_geometry_phase.svg)

Regenerate it with:

```bash
embed-optim-plot-geometry-phase
```

These are one-seed, integrated checkpoint trajectories, not individual optimizer updates or causal
effects. The common-state virtual updates, matched-scale controls, short branches, and confirmatory
seeds in the paper plan are required before turning this pattern into a mechanism claim.

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

For the prespecified high-resolution spectrum tier, select exact checkpoints and tensors instead
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
