# Complete corrected DenseOn retrieval trajectories

This immutable data-only snapshot contains all 840 task scores from twelve
optimizer/rate configurations at five retained checkpoints, eight numerical CSVs,
the complete PDF/PNG/SVG trajectory figure and its data, descriptive stage tables,
original completion observations, accepted endpoint statistics and a five-snapshot
raw-score recovery index. No learning rate, stage or task is omitted.

Scores use fourteen-task macro nDCG@10. Plot and descriptive-table values multiply
this by 100. Curves connect retained 20/40/60/80/100% points; intervening steps are
not measured. The normalized trapezoidal area covers only 20–100%, not wall time
or an imputed initialization. Stars indicate final-validation selection, never
BEIR test selection. The existing six endpoint comparisons are unchanged.

These are descriptive trajectories and previously computed task-level inference,
not new training seeds, a new bootstrap, proof of useful-dimension allocation or
an explanation of a retrieval mechanism. Full-horizon validation was used to
select configurations; a 40% crossing does not prove 60% deployable time savings.

The data index at `provenance/raw-score-index.json` identifies five immutable HF
snapshots covering all sixty checkpoint outcomes. Each gives the dataset, exact
revision, content-addressed prefix and external manifest identity, plus the native
receipt and fourteen score-file identities for each checkpoint. Index construction
rechecked all 2,888 previously anonymously recovered files (37,364,806 logical
bytes) and reconstructed every one of 840 raw scores, matching the complete table.
The endpoint snapshot additionally preserves original validation data. Existing
overlapping partial-stage snapshots are not needed for this complete five-stage
recovery and remain unchanged.

Authenticate this snapshot's `artifact_manifest.json` using the digest from the
trusted handoff, then verify every payload. The separately supplied local stdlib
reader rebuilds the complete stage summaries, areas, task-level point effects and
figure mappings. It checks existing interval values against the accepted endpoint
readout; it does not rerun bootstrap sampling. Producer-machine paths are only
provenance and must not be executed or opened as recovery destinations.

Source code, model tensors, embeddings, training examples and aggregate bundles
containing embedded Python are not included. The source/guide publication boundary
is unchanged. Same-host anonymous recovery is not a second-host model experiment,
and this data snapshot is not functional-analysis, crossed-continuation, manuscript
or full scientific completion.
