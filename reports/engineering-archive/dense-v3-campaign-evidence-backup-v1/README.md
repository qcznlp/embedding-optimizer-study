# Original training-completion evidence: immutable data backup

The 39 original completion/worker/anchor records prepared by the
[local-role reader](../dense-v3-campaign-evidence-candidate-v1/README.md) now have a
public data-only backup. **All 41 snapshot files / 342,406 bytes were anonymously
downloaded and verified**, including the description and content manifest.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `9b2925acd9339b5ce25fd332ab8840ac50319407`.
- Parent: `37702ccb55820dbf5956a57257c5582d56733da9`.
- Manifest SHA-256: `5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95`.
- Prefix: `corrected-dense-correctness-v3/training-completion-evidence/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95`.

The sole upload returned exit 0 (`f15832`, session 76334). Its commit was observed
at **2026-09-12 04:09:41 UTC**. All 41 paths are new regular Git blobs, with no
ignored files, LFS rules, replacement paths or deletion. The original 20 other
root entries, all ten existing corrected subtrees and exact root attributes are
unchanged. Actual pre-upload modes, parent inventories, commit and post-upload
checks remain under [actual](actual/).

Complete anonymous recovery returned exit 0 (`002f1e`, session 20201), finishing
at **04:10:23 UTC**. The standalone reader also returned exit 0 (`11a64f`). A new
39-record copy selected from that verified download was combined with the 133
original admission/native inputs from the **previously recovered training HF
snapshot**. The unchanged candidate assembled all 172 files, then its isolated
file-audited reader returned exit 0 (`d67991`). The returned original-row digest
remains `58a01f3cbfe05755a739bda4503cb0518fc2115ffe783efbf6ca6154f5d57f44`.
This is real recovered-data composition on the same host, not a second-host
experiment, another training run or fresh tensor/data verification.

## Preserved local preparation failure

The first local preparation returned exit 1 (`f786a8`), before any preflight or
upload attempt. The new standalone reader initially expected only bytes/SHA-256,
whereas the existing transport also records Git blob SHA-1. The corrected reader
explicitly validates all three identity fields; it does not drop the Git digest.
The original failed code and staged manifest remain under [failed-first](failed-first/)
and in `/tmp/dense-v3-campaign-evidence-backup.qS8N6o3U`. No original failure was
rewritten as success, and the transport/helper were not changed.

A new work directory `/tmp/dense-v3-campaign-evidence-backup.QBUi2g7N` retains the
successful preparation, staged inputs, actual upload and full anonymous download.
All **18 bounded tests pass** (`3f7197`), including each of 41 payload corruptions,
wrong external anchors, missing/wrong Git identities, extra paths, symlinks,
unexpected upload modes, overwrites and preservation of all ten previous subtrees.
These are transport/reader tests, not model experiments. All owned backup/test
calls are terminal; do not rerun the one-shot uploader or either preparation.

## Use and limits

Follow [the recovery guide](../../../docs/training-completion-evidence-restoration.md).
The full original download stays untouched. Only the 39 required original records
are copied into an absent local role directory for composition; the README and
transport manifest remain in the complete 41-file download.

Reader/program source and the guide are **local only**, not uploaded to HF or
published to GitHub by this operation. Copy those files off-host separately. The
preceding candidate archive is immutable and still describes its earlier local
preparation; this new backup closes only its data-record durability gap.

Ten original exit-zero proofs and two original unobserved/null OS exits remain
distinct. Original fingerprints and false formal/scientific/release flags remain
unchanged. No model, evaluation score, functional feature, publication authority,
manuscript result, prior snapshot or failed runtime attempt was changed.
The [native observation](observations.json) is 684/840 primary task cells with
eight exact live original workers; functional recovery remains unapproved.
The complete scientific pipeline, portable source release and NAACL paper are
still required. Engineering failures belong nowhere in the manuscript.
