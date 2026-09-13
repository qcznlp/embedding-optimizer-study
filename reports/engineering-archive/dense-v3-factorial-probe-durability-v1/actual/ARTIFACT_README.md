# Complete DenseOn crossed-continuation probe measurements

This data-only snapshot preserves all twelve fixed state-by-operator continuations
under three data-order seeds and their five retained checkpoints (steps 79, 157,
235, 313 and 391). Every state has the same 224-query, fourteen-task probe with one
relevant document and seven fixed competing documents, and 768-dimensional vectors.

For each of sixty states, `raw/vectors.npz` contains the original normalized FP32
query/document vectors and ordered sample IDs/task groups. `vectors-fp16.npz`
preserves the exact FP16 metric inputs; `scores.npz` preserves the original FP32
scores. `metrics.json` contains the six frozen overall/per-task measures. The
native vector manifest and worker completion are retained alongside the data.
`pretrained/` provides the same reference in both original FP32 and scoring FP16
form; this reference is reused, not another encoding.

`provenance/` preserves original worker requests, actual exits, complete pool and
native per-state readbacks. `checkpoints/` gives every branch's existing immutable
Hugging Face model revision and full checkpoint inventory. Models are separately
available from `qcz/embedding-optimizer-study-checkpoints`; they are not duplicated
in this dataset snapshot.

These are eight-candidate diagnostic measurements, not full-corpus BEIR results.
The original full-corpus evaluation and three factorial estimands are separate.
No headline, effect sign or scientific conclusion follows from backup completion.
This snapshot contains no raw example text, executable source or pickle payloads.
Absolute producer paths inside receipts are provenance, not required paths on a
new host. The artifact manifest binds every included file by size and SHA-256.
