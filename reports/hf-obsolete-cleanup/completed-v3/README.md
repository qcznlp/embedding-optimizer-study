# Completed owner-approved current-tree withdrawal

The owner explicitly confirmed the exact 9,629-file scope after being asked
whether to remove those files from the two current HF trees while preserving
valid files, Git history and shared objects. This authorizes the executed
current-tree withdrawal, not permanent history or LFS erasure.

The original `scripts/hf_obsolete_cleanup.py` ran unchanged against
[`../plan-v2.json`](../plan-v2.json), SHA-256
`1193dbc2a1a483cba79b3532b4ef47f66eb9198fefc7492ea4521a9d00f1d026`.
Its 23 focused safety tests passed before execution. The actual execution and
independent anonymous verification processes both exited zero.

| Repository | Deletion commit | Removed files | Other payloads verified unchanged |
| --- | --- | ---: | ---: |
| Model | [7cdfd2be](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints/commit/7cdfd2be93fa0eb3272eb1cece7582962def1e65) | 3,790 | 5,573 |
| Dataset | [6e2750e6](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/commit/6e2750e62bcba0249bb31810cdb7512e0b98aa8b) | 5,839 | 8,248 |

Both operations used an exact-file allowlist, fresh inventory, unchanged target
identities and a parent-commit precondition. Each atomically removed its selected
paths and replaced only its own README. There were no surviving targets,
unexpected additions or mismatched retained files.

The independent verifier anonymously re-read both complete new trees and both
cards. It compared every retained file's size, Git blob and LFS SHA-256 identity
against the execution's actual pre-commit inventory. It also downloaded and
SHA-256 verified every one of the **64 model + 1 dataset shared LFS objects**,
totaling **3,488,882 bytes**. This does not claim a new full download of every
model payload. Both original parent revisions remain anonymously readable.

## Original receipts

- [Model execution](model-execution.json) and [dataset execution](dataset-execution.json).
- [Independent anonymous combined verification](anonymous-verification.json).
- [Model shared-object details](model-anonymous-verification.json) and
  [dataset shared-object details](dataset-anonymous-verification.json).
- [Pre-execution 23-test result](cleanup-tests.xml).
- [Actual model card](model-README.md) and [actual dataset card](dataset-README.md),
  downloaded byte-identically after their respective commits.
- [Unchanged-source full hosted-CI acceptance](hosted-ci-verification.json),
  run 34772464866 at scientific source `b09b334973e69d6990e2811d5de5c1d0f471f170`.

Full execution inventories and the independent read-only verifier are retained
locally at `/tmp/dense-v3-approved-hf-cleanup.gpSR9l9J`. Exact parent/new HF
revisions in the published receipts allow independent inventory reconstruction.
The executed targets remain explicitly available in the original reviewed plan.

## Preservation boundary

No Git-history rewrite, LFS-object purge, local artifact deletion or change to
scientific source/results occurred. Deleted paths can still be read at old Git
revisions. Current v3 primary and continuation checkpoints, other retained
namespaces, shared inputs, baselines and separately identified LateOn artifacts
are untouched. Retention is not a new correctness certification of old runs.

The two old replacement-card drafts in the parent directory are preserved as
historical drafts, not the cards used for this operation. Never upload them or
re-upload withdrawn paths from a broad old snapshot. Do not execute this plan again.
