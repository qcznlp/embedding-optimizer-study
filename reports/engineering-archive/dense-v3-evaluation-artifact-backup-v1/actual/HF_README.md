# Corrected DenseOn: complete final BEIR and validation scores

This snapshot preserves all twelve final checkpoints' fourteen-task BEIR scores
and all twelve full 4,096-query validation outputs. It is separate from model,
training-dynamics and weight-geometry backups. It does not contain intermediate
checkpoint evaluations, a finished paper or a released program implementation.

All three optimizers retain their four declared learning rates. Each final BEIR
row is an equally weighted mean over the same fourteen pinned decontaminated
full-corpus tasks. The validation rule chooses the minimum mean contrastive loss
over all 4,096 rows within each optimizer, resolving exact ties by lower learning
rate. BEIR scores are not selection inputs.

## Files

- `native/beir-final/`: all 216 original endpoint score/metadata files, including
  all 168 task JSON results and twelve complete native receipts.
- `native/validation/`: all 48 original validation files, including 49,152 scored
  query records with eight candidate cosine scores and six recorded metrics each.
- `native/validation-selection/`: original complete selection and completion.
- `provenance/beir-workers/`: all 336 original started/exited worker records.
- `provenance/validation-workers/`: all 48 original admission/started/exited/scored records.
- `provenance/validation-native-readback.json`: the original source-bound native
  all-row metric/selection replay, with file identities and worker evidence.
- `provenance/primary-v3-checkpoints.json`: the exact immutable 60-checkpoint index.
- `tables/final-checkpoints/` and `tables/validation/`: all-rate score tables and
  the fixed validation-selected endpoint comparison.
- `artifact_manifest.json`: exact inventory and byte/content identities.

There are 659 payload files plus the manifest. No original file bytes are rewritten.
Raw query/document text, tokenized examples, model or optimizer tensors, executable
program bodies and credentials are outside this snapshot. Query/document IDs and
row-content digests remain as provenance. Source paths and historical worker PIDs
are records, not instructions to execute or signal them on another machine.

## Interpretation and recovery

The selected final recipes are AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4. See the
generated tables for the actual values. These are descriptive endpoint results,
not significance tests, multiple training seeds, complete five-stage dynamics,
functional-dimension evidence or a causal mechanism. Old experiments cannot fill
the missing v3 cells. All original source/scientific-completion flags remain as recorded.

Validation metrics can be recomputed from the original eight candidate scores;
their recorded FP32 values remain the original selection inputs. BEIR task JSON
contains aggregate metrics, not per-query rankings. This snapshot alone cannot
recompute query-level BEIR metrics from embeddings. Full reproduction also needs
the authenticated program source, model and data versions, and remaining analyses.

Use the immutable commit and exact snapshot prefix in the project's restoration
guide. Verify the transport manifest against its externally recorded SHA-256,
then verify every listed file's size and SHA-256 before analysis. Preserve incomplete
downloads for inspection; do not replace a recorded digest or overwrite another snapshot.

The addition does not delete historical artifacts, modify old snapshot payloads or
replace the dataset's repository card. This is numerical-artifact durability, not
source publication or proof that the entire NAACL project is complete.
