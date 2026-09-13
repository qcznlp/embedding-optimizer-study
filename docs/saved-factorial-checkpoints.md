# Inspect a restored continuation checkpoint

This CPU-only entry checks an existing DenseOn crossed-continuation checkpoint
without requiring the original machine's source directories. It preserves the
original run and component identities. It does **not** start training, admit a
new source version, or establish bitwise GPU-resume equivalence.

## Restore and establish trust first

Use the [continuation restoration guide](continuation-probe-restoration.md) to
verify the immutable probe snapshot and list its sixty model-checkpoint links.
For the selected link, download the complete checkpoint from its exact model
repository revision and prefix, retaining every file. The verified backup
manifest binds the checkpoint's `factorial_trainer_component.json`; obtain its
SHA-256 from that authenticated external manifest.

**Do not use an untrusted checkpoint's self-reported hash as authentication.**
Saved training arguments and rank RNG states use pickle-backed serialization.
Hash checking detects changes relative to a trusted reference; it is not a
signature or a sandbox for malicious checkpoints. Do not modify checkpoint
files while reading them.

## Run the reader

Use the study's pinned runtime in a separate environment. Keep the full runtime
asset layout: `configs/formal_runtime.json`, its adjacent constraints file,
and both referenced lock files in their recorded relative locations. Copying
only the JSON correctly fails reconstruction validation. In an installed wheel,
these assets live below `share/embedding-optimizer-study`; in a checkout they
are at the repository root. Supply the **local** runtime path explicitly.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  embed-optim-inspect-factorial-checkpoint \
  --checkpoint /absolute/path/to/checkpoint-313 \
  --expected-receipt-sha256 TRUSTED_EXTERNAL_COMPONENT_SHA256 \
  --runtime-spec /absolute/path/to/configs/formal_runtime.json
```

From a checkout, set `PYTHONPATH` to its absolute `src` directory and use
`python -m embed_optim.saved_factorial_checkpoint` with the same arguments.
All supplied paths must be absolute, without symlinks or `..` components.
No network access, model encoding or GPU execution is needed. The command
prints JSON to stdout and does not overwrite any experiment artifact.

The reader authenticates the complete payload before decoding, validates the
actual runtime, and retains the original checks of all 134 model tensors,
named optimizer state, scheduler, training arguments and four rank RNG states.
The native readback retains the historical identity digest. Runtime and
checkpoint hashes are rechecked; identity paths remain historical provenance,
not permission to read an old producer directory.

## Demonstrated scope and limits

The [actual verification](../reports/engineering-archive/dense-v3-portable-factorial-checkpoint-v1/README.md)
used two genuinely restored step-313 checkpoints: AdamW-source/AdamW and
AdamW-source/Muon, both order seed 314159. Their native readbacks exactly match
the previously accepted reader. All 22 loaded study modules came from an
extracted wheel, and explicit old-directory refusal controls passed. This is
same-host relocation, not a physical second-host test or a read of all sixty
continuation checkpoints. Python audit hooks are not an OS sandbox.

The historical fresh-run source guard remains unchanged and currently rejects
the changed factorial input-location module. This inspection entry does not
bypass that guard. Both previously tested GPU resumes still fail exact endpoint
comparison; see the [recovery diagnosis](../reports/engineering-archive/dense-v3-resume-endpoint-diagnosis-v1/README.md).
Original scientific checkpoints and results are unchanged.
