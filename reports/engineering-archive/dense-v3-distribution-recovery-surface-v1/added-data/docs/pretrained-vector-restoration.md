# Restore the accepted pretrained functional vectors

The already accepted pretrained DenseOn state is now preserved in a separate
public, immutable data-only HF snapshot. **All ten files / 6,439,310 bytes were
anonymously recovered and checked on 2026-09-12 at 02:07 UTC.** The standalone
reader was also executed from outside the project against the recovered files.
This is same-host recovery, not a physical second-host model experiment.

## Immutable recovery coordinates

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`
- Revision: `37702ccb55820dbf5956a57257c5582d56733da9`
- Prefix: `corrected-dense-correctness-v3/pretrained-functional-vectors/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/009ae4903fa0cd39bcafa14ef9da8d7c0d862ab369933a15182cfdd26dda8933`
- `artifact_manifest.json`: 2,259 bytes; SHA-256 `009ae4903fa0cd39bcafa14ef9da8d7c0d862ab369933a15182cfdd26dda8933`.

[Open this exact snapshot on HF](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/37702ccb55820dbf5956a57257c5582d56733da9/corrected-dense-correctness-v3/pretrained-functional-vectors/4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b/009ae4903fa0cd39bcafa14ef9da8d7c0d862ab369933a15182cfdd26dda8933).
Do not substitute repository `main` for the immutable revision, or use a hash
read only from the downloaded manifest as its own trust anchor.

## Recover on another machine

Save this guide and [verify_recovered.py](../reports/engineering-archive/dense-v3-pretrained-vector-backup-v1/source/verify_recovered.py)
before retiring the current machine. The reader SHA-256 is
`ce355193ae530ccf4042f497eb594d8156ee4426e8dcdfd2bf9685959d6d8a75`.
**The HF snapshot contains data only. These local source files are not claimed
to have been published to GitHub or included in the HF snapshot.**

The reader needs Python and NumPy; it does not import this project, PyTorch or a
model. The actual recovery used NumPy 2.5.2. Download the ten files below from the
exact revision, preserving paths relative to the prefix:

```text
artifact_manifest.json
README.md
actual-payload-readback.json
vectors/manifest.json
vectors/vectors.npz
native/pretrained.admission.json
native/pretrained.started.json
native/pretrained.exited.json
native/pretrained.encoded.json
native/pretrained.verified.json
```

For programmatic downloads, `huggingface_hub.hf_hub_download` accepts the dataset
ID, `repo_type="dataset"`, the exact `revision`, `filename=prefix + "/" + name`,
`token=False`, and a fresh `local_dir`. This is the same public download interface
used in the recorded complete recovery, not a requirement for credentials.
Keep HF's local download metadata outside the prefix directory supplied below.

Run the copied reader against the recovered **prefix directory**:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -I -B /absolute/path/to/verify_recovered.py \
  --root /absolute/path/to/recovered/prefix-directory \
  --manifest-sha256 009ae4903fa0cd39bcafa14ef9da8d7c0d862ab369933a15182cfdd26dda8933
```

It checks all ten file identities, the eight original accepted payload anchors,
the native worker's exact exit-zero provenance, the encoding-plan and tensor
fingerprint bindings, the NPZ's four arrays, shapes, dtypes, finite/nonzero vectors,
and all 224 sample IDs/groups in their original ordering. Historical absolute paths
and commands are compared only as provenance; they are never executed or read.
The NPZ is loaded with `allow_pickle=False`.

## Scope and limits

The arrays are query `[224,768]`, positive-first candidate document `[224,8,768]`,
sample IDs `[224]`, and sample groups `[224]`. Embeddings are stored as FP32 after
the original BF16 inference. Fourteen task groups each contribute sixteen queries.
The original full vector manifest retains query/candidate IDs and hashes without
raw query/document text. This backup does not include the probe text dataset,
model weights, functional feature results, or the other sixty vector states.

This remains **1 / 61 accepted vector states and 0 / 61 feature states**. The
original coordinator failure and recovery-approval boundary are unchanged. No
encoding, dimension deletion, feature extraction or statistical inference was
performed by this transport. This backup does not prove a retrieval mechanism.

The sole commit added only this subtree. All twenty other root entries and nine
previous corrected-analysis subtrees were unchanged. Exactly one NPZ used the
existing LFS rule; no root attributes, old files or source release were changed.
Sixteen bounded CPU transport tests passed, including corrupted files, symlinks,
wrong upload modes and overwrite refusal. They are engineering tests, not runs.

The [local execution archive](../reports/engineering-archive/dense-v3-pretrained-vector-backup-v1/README.md)
contains the source, actual test/process results, upload-mode preflight, immutable
remote audit, anonymous recovery receipt and separately dated primary observations.
