# Complete corrected DenseOn third-stage BEIR data snapshot

This immutable data-only snapshot contains all twelve declared optimizer/rate
configurations at checkpoint step 2345 (the third of five retained stages).
It does not contain a new model run, a complete five-stage retrieval trajectory,
a mechanism result or a source-code release.

## Exact coverage

- AdamW: 1e-6, 3e-6, 1e-5, 3e-5.
- Muon and NorMuon, each: 1e-4, 3e-4, 1e-3, 3e-3.
- All fourteen original decontaminated BEIR tasks per state: 168 task scores,
  twelve exact macro means, and 168 original successful worker executions.
- 216 native score/metadata/completion files and 336 original worker records.
- Two derived CSV tables, one acceptance JSON, this README and the manifest:
  557 files in total.

Six pool-B states / 84 task values overlap the earlier partial third-stage
snapshot, which is preserved unchanged. Only the six pool-A states / 84 task
values are newly backed up. Overlap is not an additional training run, score or
independent experimental replication.

The accepted source readback contains 48 checkpoint outcomes (all twelve at
steps 782, 1563, 2345 and 3907); this snapshot selects only the full twelve-state
step-2345 cohort using the declared configuration set, never scores.

## Numerical interpretation and recovery

Task values are nDCG@10 in [0, 1]. Macro means give each original task weight
1/14. The CSV column macro_score_0_to_100 multiplies the exact mean by 100.
Each mean is independently reconstructed from the native task JSON values;
no missing score is imputed and no task or learning rate is selected away.

The artifact_manifest.json supplies relative paths, sizes, SHA-256 and Git blob
identities for every other payload. Pin the immutable dataset revision and
externally recorded manifest SHA-256 when downloading. Historical absolute
paths in native provenance are recorded identifiers, not paths to read on a
new machine; a recovery reader must resolve only the supplied snapshot.

This snapshot includes neither executable source nor query-level ranking traces.
Byte verification and raw-score reconstruction do not establish model replay,
cross-host training equivalence, seed robustness or scientific completion.
The primary comparison still requires twelve configurations x five stages x
fourteen tasks (840 cells). Functional dimension analysis, crossed continuation
and complete paper/source release remain separate requirements.

