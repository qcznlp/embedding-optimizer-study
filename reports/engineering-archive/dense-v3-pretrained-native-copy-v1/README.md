# Genuine pretrained-vector copy: native checkpoint-backed compatibility

The accepted pretrained vector pair now passes the **unchanged native reader in
both its original location and a fresh temporary directory**. This is an actual
CPU read of genuine model/vector data, not the preceding mocked recovery-flow
test. It did not encode a model, create an execution authority, take a GPU lease
or restart the failed functional coordinator.

## Verified result

The [actual readout](actual/readout.json), SHA-256
`773c184b474c0508ff22a549db5303abefe922c95272bd539c32ed5c0b503195`,
finished **2026-09-12 13:35:21 UTC**. Tool session **38638** returned zero in
terminal chunk `47df0b`.

- Both native reads reconstruct the pretrained checkpoint's **134 saved tensor
  fingerprints** under the original BF16 conversion and compare them to the
  recorded immutable model-loading observation.
- The four arrays have identical shape, dtype, values and raw bytes:
  queries **224 × 768 FP32**, documents **224 × 8 × 768 FP32**, and both 224-row
  sample identity arrays.
- The original manifest and vector files remain byte-identical in the copy.
  The original native return matches the accepted worker receipt. The copied
  return differs only in its explicitly relocated manifest's absolute path;
  every other field is equal.
- Incorrect external manifest SHA and a different state plan are both refused.
- All **66 original bound files** and **11 pretrained checkpoint files** are
  checked before and after. CUDA remains uninitialized.

The loaded-state aggregate fingerprint is
`23b590011b87358cbd51f17619bf8cef06974f378268fdaf713a9f9b0e570ecb`.
The exact native reader, SHA
`ea33869f8cf677ad0c15e8747b6d91045ba55a39301a8bf2c52f09bf642ed858`,
is copied under [source-original/](source-original/) without modifications.
The calling adapter and plan remain in [source/](source/) and [plan.md](plan.md).
The compatibility call is one-shot and host-bound, not a portable launch command.

## Boundaries and next action

This completes the real copied-pretrained native-inspection item left by the
[offline full-flow candidate](../dense-v3-functional-flow-candidate-v1/README.md).
It establishes that existing accepted vectors can be reused through their native
reader after a path change, without re-encoding or changing their provenance.
It does **not** deploy that candidate or approve its production execution.

Actual functional progress remains **one accepted pretrained vector state and
zero feature states**. These are two reads of that same state, not two new states.
No coordinate ablations, rotations, retrieval, new model forward/backward,
scientific inference, manuscript or remote publication occurred. The copy was
read on the same physical host, not a second machine. Checkpoint fingerprint
verification is model-file inspection, not training resumption.

Still required: scoped owner recovery approval; reviewed production source-bound
coordinator/observer authentication in a new namespace; exact new handle and
original dual-lease checks; sixty remaining encodings and all real functional
features. The original stopped/failed controllers remain untouched.

The [archive verification](verification.json) independently reopens the copied
arrays without importing the project or Torch, checks their recorded identities,
and preserves all original data/source inputs. The actual tool command is in
[commands.json](commands.json). Engineering details stay out of the manuscript.
