# Accepted pretrained DenseOn functional-probe vectors

This immutable data-only snapshot preserves the **one already accepted pretrained
state** from the corrected DenseOn functional-dimension campaign: 224 queries,
eight positive-first candidates per query, and 768-dimensional FP32 stored vectors.
The original inference used BF16 model parameters. Query/document arrays, sample
IDs and groups, the complete original vector manifest, five native worker receipts
and the original independent payload-readback record are preserved byte-for-byte.
The manifest includes candidate/query identifiers and hashes, not raw example text.

This is **1 / 61 vector states and 0 / 61 feature states**. It is not a completed
functional analysis, new inference, retrieval result, model checkpoint or source
release. The original campaign coordinator failed after this state; that historical
status is retained in the readback record. This backup neither restarts nor repairs
that campaign, and does not manufacture new worker or success receipts.

`artifact_manifest.json` records the byte count, SHA-256 and Git-blob SHA-1 of all
nine payload files. Obtain its SHA-256 from the separately supplied recovery guide,
not from an untrusted copy of the snapshot itself. Download all ten files at the
immutable commit, verify their hashes, then load `vectors/vectors.npz` with NumPy's
`allow_pickle=False`. Array layouts are query `[224,768]`, document `[224,8,768]`,
sample IDs `[224]` and sample groups `[224]`. Preserve their original row ordering;
the first document candidate is the positive. The vector manifest binds every row
to its query/candidate identifiers and the original encoding plan.

Absolute paths and process identifiers in original receipts are historical
provenance, not paths to read or commands to execute on a new machine. The separate
recovery reader uses only this recovered snapshot and NumPy; no model or GPU is
needed. Reader source and the recovery guide are separate local repository
artifacts, not included in this data-only snapshot. Save them before retiring the
original machine. No claim of physical second-host execution is made.
