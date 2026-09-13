# Restore current primary checkpoints

The current primary model namespace is `corrected-dense-no-packing-v1/dense/` in
[`qcz/embedding-optimizer-study-checkpoints`](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints).
Do not download the whole repository at its moving default revision: it also contains historical
experiments excluded from the paper, with withdrawal awaiting exact-scope approval.

The latest verified coverage is in [PROJECT_STATUS.md](../PROJECT_STATUS.md). At the ten-run
milestone, 50 checkpoints from ten complete runs are independently content-verified. This is not
12-run completion, and future checkpoint uploads must receive their own verification.

## Use the content audit as the download index

[The ten-run HF audit](../reports/experiment-integrity/huggingface-digest-audit-ten-run.json)
contains a `records` entry per complete run. Each entry names the repository, exact immutable
`revision`, `prefix`, and every local SHA-256/remote LFS or Git-blob digest. Use that run's recorded
revision, not `main` and not another run's upload commit.

The trusted SHA-256 for this exact ten-run audit is:

```text
9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6
```

Check the audit against a separately obtained trusted digest before following its locations. Merely
recomputing a digest from an untrusted replacement file does not authenticate it. Do not reuse this
digest for a future twelve-run audit.

The new restoration diagnostic and detailed audit index are currently prepared in the isolated
checkout; their GitHub source publication is still awaiting the WIP exception described in
`PROJECT_STATUS.md`. The checkpoints themselves are public. A clean clone of the current public
main branch does not yet contain this new diagnostic; obtain the verified files first or wait for
the approved source release. This documentation is not a claim that the final project release exists.

## Restore one complete run

For example, NorMuon 3e-4 has all five scheduled checkpoints at the following immutable revision.
Run this from the cloned checkout, with a **new** destination directory. The command downloads
about 7.36 GB; choose another run by substituting its exact audited prefix and revision together.

```bash
hf download qcz/embedding-optimizer-study-checkpoints \
  --revision e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae \
  --include 'corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/**' \
  --local-dir /path/to/new-primary-download
```

The directory layout remains the HF namespace. The model at stage 80% is then
`/path/to/new-primary-download/corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/checkpoint-3126`.
Each checkpoint includes weights, tokenizer/pooling settings, optimizer, scheduler, trainer state,
training arguments and four rank RNG payloads. Root-level run metadata is also preserved for a
complete-run download. Do not copy an isolated stage into the live matrix or manufacture a
`completed.json` marker for a partial restore.

## Download and verify a smaller restoration example

The following exact immutable source was actually downloaded and tested. A stage-only download
plus the run configuration is 18 files / 1,351,464,827 bytes. The client creates its own local
metadata cache; it does not need a simultaneous `--cache-dir` option.

```bash
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download qcz/embedding-optimizer-study-checkpoints \
  --revision e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae \
  --include 'corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/checkpoint-3126/**' \
  --include 'corrected-dense-no-packing-v1/dense/padded-normuon-3e-4/run_config.json' \
  --local-dir /path/to/new-checkpoint-download \
  --force-download --max-workers 2
```

After the diagnostic source is available, run from that checkout. Replace `/path/to/checkout` and
the new download/receipt locations with actual absolute paths. Use the pinned formal environment
described in the [main README](../README.md#installation).

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/path/to/checkout/src \
python -m scripts.audit_checkpoint_download \
  --repository /path/to/checkout \
  --audit /path/to/checkout/reports/experiment-integrity/huggingface-digest-audit-ten-run.json \
  --audit-sha256 9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6 \
  --run-id padded-normuon-3e-4 --step 3126 \
  --download-root /path/to/new-checkpoint-download \
  --output /path/to/new-restoration-receipt.json
```

The diagnostic first verifies exact paths, sizes and both file digests against the trusted audit;
only then does it open model or optimizer payloads. It rejects unsafe paths, symlinks, extra or
missing checkpoint files and changed content, and never repairs or overwrites a file. It requires
a new receipt outside the downloaded tree and performs no network or GPU work.

It compares the downloaded run configuration with the frozen matrix, applies the original Dense
reload control, checks two short synthetic query/document encodings, and reuses the existing CPU
optimizer/scheduler restoration test. Two seeded synthetic-gradient updates must give identical
parameters and states with or without an intervening serialization. The learning-rate schedule
must agree. All downloaded payloads are rehashed after the check. The continuation diagnostic
accepts only nonterminal scheduled stages; a final checkpoint's zero learning rate is not changed
to make a synthetic update test pass.

The [actual NorMuon receipt](../reports/experiment-integrity/checkpoint-download-normuon-3126.json)
records two finite 768-dimensional unit vectors and exact continuation of all 134 parameters and
optimizer states. This was a fresh forced HF download and an offline fresh-directory test **on the
same physical host**, not a second-host or four-GPU resume experiment. It does not test data-loader
order, rank RNG replay, GPU mixed precision, full-length input execution or retrieval quality.
The source and downloaded evidence remain unchanged.

## Restore analysis and reproduce findings

Raw geometry for these ten completed runs is at immutable analysis commit
`489c606076b0f8a0ef82adc7b85f1fb29da794bc`, under
`corrected-dense-no-packing-v1/weight-space/` in the
[`analysis-artifact dataset`](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts).
Its [archive receipt](../reports/experiment-integrity/primary-geometry-ten-run-archive.json) identifies
all 60 files, their digests and the original numerical sources. Use a new destination and that exact
prefix/revision; do not substitute historical geometry or assume the raw files are a complete
retrieval/dimension explanation.

Complete analysis also needs the frozen code/runtime, base-model revision, data/probe identities and
the eventual primary BEIR/dimension/factorial outputs. Checkpoints alone cannot reconstruct results
that have not yet been produced. The future source-bound dimension archive/replay procedure is
documented in [dimension utilization](dimension-utilization.md); its real primary archive is pending.
