# Combined main handoff repair — prepared, not deployed

This engineering-only package combines the corrected evaluation source-topology repair with the
previously tested three-header publication syntax repair. It was prepared from live main at
2026-09-05 00:56 UTC, before any primary BEIR result existed. It does not contain a scientific
result, runtime transition or grant of publication/deletion authority.

The exact [six-file patch](main-handoff-repair.patch) has SHA-256
`a9c642582fe431d28c0ae28303f2cbeff9e34ab16bc52e91fa0c6adefd2d0b15`.
It changes only:

- `evaluation_source_provenance.py`: recognize the corrected evaluator's exact ten-source mapping,
  retaining byte/hash authentication against reachable Git history.
- `corrected_publication.py`: replace the three malformed header separators, with no numerical or
  scientific text change.
- The outcome protocol: bind the previously omitted source-provenance helper before aggregation.
- The bridge, sensitivity and publication protocols: update their exact parent identities and
  record the non-scientific amendment. Every other scientific field remains unchanged.

The [preparation receipt](preparation.json), SHA-256
`253a0f3f7809b81b2e24c0678090ccf693d3963ff4d1cbb4e9f04939a9ff838e`, binds all original and
candidate sources. Its `before/` directory preserves the 24,577-byte observed main ledger under
SHA-256 `314b66a065d60e7bfedf9d4e9f669418d509de1a1844597e794b6257be1b21bf`.
That ledger has zero post-training steps and eight complete-run backup records. `after/` contains
only the six candidate files; it is not a complete checkout and must not be used as an experiment.

## Rehearsed transition

The [migration draft](migration-draft.json) changes main contract
`4152531e354bdf325411c7f4d6de044ea68b6d5f26a59fbf5914f8b55f3c9789` to
`af646ecf89bfe87cf79a20eaf3495d134eeb912899e9c219a7917f96dfdc976c`.
All 17 commands, every argument, the controller/backup implementation, matrix, execution protocol,
evaluation protocol and geometry analysis protocol remain unchanged. The four allowed source-list
changes are exactly the outcome, bridge, sensitivity and publication protocol identities.

[Forty focused checks](../../experiment-integrity/main-handoff-repair-tests.xml) pass with zero
failures/errors/skips, including ten new combined-repair cases. They reconstruct every candidate
byte and the entire patch, compare every scientific field with its original, exercise the actual
four protocol loaders, and reject source-verifier drift before outcome aggregation. The actual
migration function, applied only to a temporary ledger copy, preserves the byte-identical original
archive, all eight backups and prior migration records; a second invocation is idempotent.
The existing syntax fixture also compiles the exact same repaired renderer bytes. A read-only
`git apply --check` against live main passes; no patch was applied there.

The migration rehearsal substitutes the projected contract for the live-path contract recomputation
inside the temporary copy. It therefore verifies migration semantics, not a real controller lease,
deployed interpreter, complete live publication layout or executed runtime handoff. This selection
supplements the previous 1,052-test full suite; it is not a newly executed 1,062-test full run.

## Required deployment boundary

The previous header-only target `6d991052...` is not this combined target. The existing isolated
successor still requires that older target and correctly rejects `af646ecf...`. Its exact parent
amendment and the later narrative/dimension protocol bindings must be refreshed consistently before
deployment, preserving all primary completion gates and original factorial commands. A synthetic
fully complete main ledger passes only after changing the exact parent identity; unfinished steps
remain rejected. No successor configuration or live ledger was edited by this preparation.

Do not merge the whole story branch into the active main controller, silently replace the verifier,
rewrite existing backup receipts, or start a second controller. Obtain verified sole-controller
ownership before applying the exact repair and migration. Early BEIR workers and the main
controller's all-eight-GPU validation also require the single-owner coordination described in the
runbook. No training or controller process was inspected or signalled here.

The reproducible preparation entry point is `scripts/prepare_main_handoff_repair.py`; it requires a
new output directory outside the live experiment checkout, refuses a different predecessor or
existing primary BEIR results, and cannot deploy, signal a process, upload or delete anything.
The HF withdrawal and isolated WIP-publication approval questions remain separate and unanswered.
