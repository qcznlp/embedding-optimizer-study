# Owner-directed Hugging Face withdrawal: awaiting exact-scope approval

The owner requested removal of results from the erroneous historical implementation.
No Hugging Face file, card, branch, tag, commit or LFS object has been changed by
this cleanup task. Execution was rejected by the tool's safety reviewer before the
command started. Do not circumvent that rejection with smaller batches or another API.

The review requires explicit owner confirmation of the exact large deletion scope.
A separate owner question about current-tree deletion versus history removal is
also unanswered. The script implements current-tree deletion only and has no
history-purge operation.

## Reviewed scope

| Repository | Selected files | File bytes |
| --- | ---: | ---: |
| `qcz/embedding-optimizer-study-checkpoints` | 3,790 | 305,217,006,121 |
| `qcz/embedding-optimizer-study-analysis-artifacts` | 5,839 | 61,035,906,080 |
| Total | 9,629 | 366,252,912,201 |

These are logical file sizes, not a promise of reclaimed physical storage. HF
deduplicates large objects, and an ordinary deletion commit does not erase history.

`before.json` pins both read-only repository inventories. Its SHA-256 is
`16f416c0eb6ce2d4912097bd22b214b0e70fcab4f74b1316dd364a589259de94`.

`plan-v2.json` contains every selected path, byte count, Git blob identity and LFS
digest, plus the complete inventory of retained files. Its SHA-256 is
`1193dbc2a1a483cba79b3532b4ef47f66eb9198fefc7492ea4521a9d00f1d026`.
`plan.json` is a superseded initial plan; do not execute it.

Selected artifacts comprise the historical Dense discovery, routing-control,
confirmation and shared-start runs, affected Dense engineering outputs, quarantined
runs, and their evaluation/representation/weight-space/intervention derivatives.
Mixed historical summaries containing invalidated Dense results and identifiable
historical Dense/mixed result-bearing logs are included.

Protected artifacts include:

- All current `corrected-dense-no-packing-v1/` checkpoints and analysis files.
- Shared `project/data/`, configuration snapshots, and the confirmatory-data receipt.
- Separately identified LateOn weights, raw results and logs, without certifying
  their correctness or promoting them into the current paper.
- Independently encoded, untrained Dense BEIR baseline exports and metrics.
- Unknown/new namespaces, `.gitattributes`, and future primary-dimension/factorial paths.

The two `*-README.md` files here are **unpublished replacement drafts** for the
same atomic deletion commits. Their completed-deletion wording is not a statement
about current HF state. They must not be uploaded separately before deletion.

## Safety implementation and checks

`scripts/hf_obsolete_cleanup.py` compiles conservative selectors into a complete
explicit-file plan. Execution requires its reviewed SHA-256, re-inventories the
current revision, rejects changed targets or newly appearing invalidated artifacts,
uses an atomic parent-commit precondition, and verifies every preserved file's
byte count and digest after each repository commit. It does not delete local files.

`tests-v2.xml` records 23 passing focused safety tests. Ruff and `git diff --check`
passed. The tool-level rejection means the remote deletion and post-deletion
verification paths have **not** been exercised on HF.

History deletion requires an additional coordinated plan: the active corrected
backup receipts and controllers pin original immutable HF commits. Preserve and
independently re-audit all good payloads, retain old local receipts as provenance,
and explicitly migrate those dependencies before erasing their public revisions.
Do not silently invalidate backup proofs or claim that HEAD deletion is complete
historical erasure. No local historical records or live training/control source
bytes were deleted or changed by this cleanup task.

Do not re-upload withdrawn artifacts from a broad project snapshot after an
approved cleanup. Keep the withdrawal manifest available to later agents.

## Verified history dependencies — 2026-09-04 23:45 UTC

The read-only [dependency audit](history-dependencies.json), SHA-256
`d2f0a14d3f67087bbf55a111f9aa25f712811993499b24a0904df8a2b6469c1d`, checks fresh immutable
trees, the reviewed deletion targets, named remote refs and primary backup receipts.
All 9,629 deletion targets retain their reviewed identities. The two repositories
have only their `main` branches in this observation; tags, conversion refs and PR
refs are empty. This is not a scan of every historical or unreferenced object.

- The model repository has **52 shared LFS objects**, totaling 191,756 unique bytes,
  referenced by both withdrawn and retained paths. Their retained uses include
  **225 current primary checkpoint paths**, including RNG and scheduler state.
- The dataset has **one shared LFS object**, 3,111,978 bytes: the untrained Dense
  BEIR vector baseline appears under both a withdrawn historical namespace and its
  retained baseline namespace. Its bytes must remain available.
- **49 backup receipts pin 39 distinct commits**: eight complete-run receipts,
  39 intermediate-checkpoint receipts and two partial geometry receipts. All those
  pinned prefixes are readable, and every referenced file's byte count and Git/LFS
  digest agrees with the observed current tree. This comparison does not download
  or locally rehash every payload, nor does it retroactively strengthen old receipts.

Deleting an LFS object affects every commit referencing it and is irreversible,
as documented by the [HF API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.permanently_delete_lfs_files).
Therefore a filename-prefix match is insufficient for permanent object deletion.
The dependency report deliberately records `safe_to_purge=false`; it is not an
executable history-deletion plan or a grant of authorization. Preserve shared
payloads and establish independently verified replacement references before any
approved rewrite. The existing live backup consumers must be migrated explicitly,
never by silently changing the original upload identity in historical receipts.

The second NorMuon checkpoint-2345 receipt appeared at 23:46 UTC, after this
snapshot. It is separately verified and retained under `reports/dense-no-packing/`;
new uploads mean the dependency audit must be refreshed before any approved cleanup.
Do not reuse the 49-receipt count as a timeless description of the live archive.

The dependency and original cleanup checks pass 38 focused tests. The broader
backup/supervisor/header regression selection passes 65 tests with zero failures,
errors or skips; see `dependency-backup-regression-tests.xml`. These are focused
checks, not a newly executed whole-project test suite. The read-only command is:

```bash
PYTHONPATH=/root/embedding-optimizer-story-refactor/src \
  /usr/bin/python3 -m scripts.audit_hf_withdrawal_dependencies \
  --plan reports/hf-obsolete-cleanup/plan-v2.json \
  --plan-sha256 1193dbc2a1a483cba79b3532b4ef47f66eb9198fefc7492ea4521a9d00f1d026 \
  --repository /root/embedding-optimizer-study \
  --output /tmp/new-withdrawal-dependency-audit.json
```

Run from the isolated worktree and use a new output path. This command cannot
delete files, rewrite history, upload content or migrate live receipts.
