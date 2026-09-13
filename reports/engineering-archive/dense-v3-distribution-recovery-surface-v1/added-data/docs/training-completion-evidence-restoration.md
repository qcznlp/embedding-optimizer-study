# Restore the original corrected training-completion evidence

This public **data-only** snapshot preserves the 39 original evidence records
needed by the versioned local-role training reader. It does not publish source
code or grant scientific/execution authority.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Immutable revision: `9b2925acd9339b5ce25fd332ab8840ac50319407`.
- Manifest SHA-256: `5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95`.
- Files: **41 / 342,406 bytes**, including README and manifest.

```text
corrected-dense-correctness-v3/training-completion-evidence/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95
```

[Browse the immutable snapshot](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/9b2925acd9339b5ce25fd332ab8840ac50319407/corrected-dense-correctness-v3/training-completion-evidence/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95).
Download the entire pinned prefix, not the moving branch. Anonymous per-file
recovery actually completed at **2026-09-12 04:10:23 UTC**. All payload bytes and
remote Git identities matched. No file in a previous snapshot was changed.

## Required local code

Copy these local directories off-host separately; this backup did **not** publish
them to HF or GitHub:

- [Snapshot verification source](../reports/engineering-archive/dense-v3-campaign-evidence-backup-v1/source/verify_recovered.py).
- [Completion-reader sources](../reports/engineering-archive/dense-v3-campaign-evidence-candidate-v1/source/), especially `campaign_evidence.py` and `assemble_snapshot.py`.

The independent reader uses the Python standard library only; no GPU, Torch,
NumPy, project package or W&B credential is required. Its external manifest
anchor and embedded original-record digests are mandatory, not caller-selectable
replacement success records. Run on the downloaded **prefix directory**:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -I -B /path/to/verify_recovered.py \
  /absolute/path/to/downloaded-prefix \
  5cc7e43868457775f2335fb19c38ff868549fdc5f9013cd846540e5baa86db95
```

## Compose the 172-file evidence input

Also restore the unchanged [training-observation snapshot](training-analysis-restoration.md)
at commit `3c95da08a4c817d5bcd58b5c84df777716a02ca9`. It supplies the admission
and 132 native metadata files. The new snapshot supplies the other 39 records.

After the full 41-file verification passes, copy only its `anchors/` and
`receipts/` directories into a **new, empty local additional-records root**.
Keep the complete download, including README/manifest, intact. For example,
with explicit absolute paths and an absent destination:

```python
from pathlib import Path
import shutil

downloaded = Path('/absolute/path/to/verified-new-prefix')
additional = Path('/absolute/path/to/new-additional-records')
if not additional.is_absolute() or any(p.is_symlink() for p in (additional, *additional.parents)):
    raise ValueError('Use an absolute non-symlink destination')
additional.mkdir(exist_ok=False)
for role in ('anchors', 'receipts'):
    shutil.copytree(downloaded / role, additional / role)
```

Then run the existing assembler and isolated reader:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B /path/to/assemble_snapshot.py \
  /absolute/path/to/recovered-old-training-prefix \
  /absolute/path/to/new-additional-records \
  /absolute/path/to/new-combined-evidence
CUDA_VISIBLE_DEVICES='' /usr/bin/python -I -B /path/to/campaign_evidence.py \
  /absolute/path/to/new-combined-evidence
```

The actual recovered-data composition and isolated read both passed on this host:
**172 files / 12 runs / 60 checkpoints**, with ten observed exit-zero and two
unobserved/null OS exits. It authenticated original metadata, not model/optimizer
tensor bytes or raw training data. This is not physical second-host replication.
Preserve partial output on failure; never overwrite an existing evidence root.

The adapters are not installed in the formal scientific/manuscript consumer.
Historical failure flags and data-history fingerprints remain unchanged. Full
BEIR trajectories, functional analysis, crossed continuation, reviewed source
transition and final paper/repository publication remain separate work.

See [the backup record](../reports/engineering-archive/dense-v3-campaign-evidence-backup-v1/README.md)
for actual commands, the preserved first local preparation failure, eighteen
passing tests, upload/recovery receipts and old-root preservation checks.
