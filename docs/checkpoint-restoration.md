# Restore the current corrected primary checkpoints

The primary matrix has **60 verified checkpoints from 12 complete v3 runs**.
Use [primary-v3-checkpoints.json](primary-v3-checkpoints.json), which records each
checkpoint's immutable HF revision, exact prefix and all twenty payload hashes.
It covers 1,200 files / **89,889,820,336 logical bytes** (about 89.9 GB, 83.7 GiB).
Each stage has its own recorded revision; do not substitute one run's commit for another.

There are also **60 additional checkpoints from twelve complete crossed
continuations**. Their immutable locations and original manifests are indexed
by the separate [continuation recovery guide](continuation-probe-restoration.md).
The commands below cover only the primary sixty, not all 120 project checkpoints.

The repository is `qcz/embedding-optimizer-study-checkpoints`. The current prefix
starts with `corrected-dense-correctness-v3/dense/4400f1ce…/`; run IDs start with
`verified-v3-`. The older `corrected-dense-no-packing-v1` namespace is **not**
the current primary experiment. Do not download the moving default branch or the
whole mixed-history repository.

## Authenticate the index

The trusted SHA-256 of this exact 507,809-byte index is:

```text
76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89
```

Obtain this digest from the verified project handoff, not by trusting a digest
supplied alongside an arbitrary replacement index. The index was derived from
the complete training/durability archive and all **180 original upload, native
checkpoint and remote-audit receipts**. Its provenance and original digests are
retained in the index; generation itself was offline.

A separate fresh anonymous check subsequently verified all **60 immutable
revisions and 1,200 indexed remote paths** against those original digests. Its
[evidence archive](../reports/engineering-archive/dense-v3-recovery-entry-v1/README.md)
also retains the real complete download and independent offline file/seal check.
Remote metadata verification is not a new download of all 89.9 GB.

The standalone [recovery script](../scripts/restore_primary_v3.py) needs only the
Python standard library to list or verify. Downloading additionally needs
`huggingface_hub` (tested here with 1.28.0). It imports no model/training code,
executes no pickle, uses no GPU and downloads anonymously from fixed HF commits.
This local WIP source/index has not yet been published as a complete GitHub
release; obtain these exact files or wait for the authorized source release.
Public checkpoints being available is not a claim that a clean clone is complete.

## Inspect the download scope first

Run from the checkout containing the current script and index:

```bash
python -B scripts/restore_primary_v3.py list \
  --index docs/primary-v3-checkpoints.json \
  --index-sha256 76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89 \
  --all
```

The command prints only checkpoint identities and sizes. To select one complete
five-stage run, replace `--all` with `--run-id verified-v3-muon-3e-4`.
Add `--step 2345` to select only its 60% checkpoint. No selection is implicit.

## Download and verify

The following exact source was actually downloaded anonymously into a new local
directory: Muon 3e-4 at step 2345, immutable revision
`3b20ad40ac7cea5029b3058468381051a91ec7af`,
**20 files / 1,350,896,259 bytes**. Replace the destination with an absolute,
new or empty directory on the destination machine:

```bash
python -B scripts/restore_primary_v3.py download \
  --index docs/primary-v3-checkpoints.json \
  --index-sha256 76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89 \
  --run-id verified-v3-muon-3e-4 --step 2345 \
  --destination /absolute/path/to/new-checkpoint-download \
  --workers 2
```

For **all sixty checkpoints**, replace the run/stage options with `--all`.
For all five stages of one run, omit only `--step`. The default is always the
complete twenty-file checkpoint, not a weights-only substitute.

Every downloaded file is checked against its SHA-256 and original remote
LFS SHA-256 or Git-blob SHA-1. Model, tokenizer, pooling, optimizer, scheduler,
Trainer state, training arguments, numerical/run contracts, checkpoint seal
and all four rank RNG files are retained. The client also creates its own
`.cache/huggingface` metadata outside the checkpoint directories.

For an interrupted download, inspect the destination and repeat with `--resume`.
The script first checks every existing selected file; valid files are reused,
while corrupt files, symlinks or unexpected checkpoint files are **refused and
preserved**, not overwritten or deleted. Client partial-download files remain
available for resumption. Do not use a current training-output directory.

Offline verification is independent of a download command:

```bash
HF_HUB_OFFLINE=1 python -B scripts/restore_primary_v3.py verify \
  --index docs/primary-v3-checkpoints.json \
  --index-sha256 76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89 \
  --run-id verified-v3-muon-3e-4 --step 2345 \
  --destination /absolute/path/to/new-checkpoint-download
```

The directory layout preserves the exact HF prefix shown by `list`. Each actual
model directory ends in `verified-v3-muon-3e-4/checkpoint-2345`. Use that complete
directory when loading the model; do not flatten files from different stages.

## What recovery does—and does not—establish

The actual download and file checks were performed in a fresh directory **on this
same physical host**, not on a second machine. The commands are designed to be
host-independent, but an actual second-host end-to-end experiment is not claimed.

These are byte-integrity and durability checks, not model execution or training
resume tests. The index does not reconstruct root-level training logs or create
a `completed.json`. Do not relabel a downloaded checkpoint as a newly completed
run, bypass source/data/runtime admission, or claim that default DDP restart will
reproduce every subsequent floating-point update bitwise.

To analyze weights or recompute representations, also retain the exact model
configuration, runtime and data/probe identities. To reconstruct the whole paper,
the complete v3 retrieval, functional and factorial outcomes are additionally
required; some have not yet been produced. Historical analysis archives cannot
fill those gaps. Current state and next actions are in
[CURRENT_EXPERIMENT.md](../CURRENT_EXPERIMENT.md).

The **already completed current weight-analysis artifacts** now have a separate
public, immutable backup. Use [weight-analysis-restoration.md](weight-analysis-restoration.md)
for all original spectra/bases, full tables, update-map arrays and figures. This
does not substitute for the unfinished retrieval/functional/factorial outcomes.

The [previous recovery guide](../reports/engineering-archive/dense-v3-recovery-entry-v1/before/checkpoint-restoration.md)
is preserved for historical provenance only. Its old run IDs, commits, numerical
diagnostic and analysis namespace are not the current restoration procedure.
