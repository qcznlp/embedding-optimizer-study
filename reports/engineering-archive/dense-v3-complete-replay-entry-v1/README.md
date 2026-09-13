# Stable local entry for the completed full numerical/PDF replay

2026-09-13. The earlier guide still described the combined replay as waiting,
although it had completed. This task corrects that handoff and preserves the
actual original closed bundle in a stable repository-relative location.
It does not rerun any scientific computation or rewrite historical receipts.

Use [the complete replay command](../../../docs/paper-results-reproduction.md#complete-paper-replay).
The original `closed/replay_complete.py` must run with `closed/primary` as its
working directory, with a new separate output path, CUDA hidden and empty
PYTHONPATH. Its original external manifest and source digests remain unchanged:

- Manifest: `746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7`.
- Entry: `1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566`.
- Original accepted completion: `6a0a8a3657d88a20529db8d45d32c31c5b74be783c0abb9d57994b0009d3c275`.

Actual integration `d98a80` exits zero. The original inventory checker authenticates
the source and copied directories; all **189 inputs / 117,532,494 bytes**, plus
their original manifest, are copied exactly. The entry source is unchanged.
The four generated includes, results file and three figures (eight files) also
match `paper/current` byte-for-byte. Original successful completion and both
zero-refusal I/O receipts are copied under `original-completed/`.

The native complete-paper replay itself was completed earlier in session
96872 / acf598, exit zero. Its initial outer-working-directory attempt in
56449 / ef4678 exited one and remains preserved in the
[closeout](../dense-v3-final-evaluation-closeout-v1/README.md). No code, guard,
allowlist, source hash or tolerance changed in the successful working-directory
correction. The present copy/inventory check is not another execution of that
numerical graph, physical second-host test or newly completed experiment.

The combined replay reconstructs the **original complete-result prose**, not
the later reviewed prose/reference revision. The reviewed manuscript's
authenticated `paper/current` entry separately reproduces that version;
their generated scientific inputs agree. Both are local, not published source.

`before/` retains the stale guide and pre-change handoff; `after/` retains the
corrected versions. `integration.json` records inventory and generated-file
equality. Final documentation tests/build/full unchanged audit outcomes are in
`actual/`; each stage has its own observed exit receipt. `manifest.json` binds
the archive, excluding itself. The closed bundle is intentionally a historical
source/data closure, not a wheel payload or a claim that every current numerical
module has been admitted for fresh execution.

No models were encoded, checkpoints decoded, experiments dispatched, GPU/helper
processes touched, or external systems written. Unified fresh-source admission,
exact GPU-resume mismatch diagnosis and reviewed source publication remain open.
