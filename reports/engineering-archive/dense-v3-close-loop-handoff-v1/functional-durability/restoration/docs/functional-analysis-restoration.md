# Restore functional measurements and continuation calibration

All **624 files / 7,496,655,065 bytes** are uploaded and were independently
downloaded anonymously with every checksum matching on 2026-09-12. The snapshot
is public and immutable. No older HF file or root attribute was changed.

- Repository: [qcz/embedding-optimizer-study-analysis-artifacts](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/9b182d77b31c93277457a578c16389935bc3fdb9/corrected-dense-correctness-v3/functional-and-calibration-v1/0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38).
- Revision: `9b182d77b31c93277457a578c16389935bc3fdb9`.
- Prefix: `corrected-dense-correctness-v3/functional-and-calibration-v1/0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38`.
- Manifest SHA-256: `0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38`.

## Contents

| Directory | Preserved actual data |
| --- | --- |
| `vectors/` | Pretrained + all 60 primary states; complete FP32 probe vectors and original provenance |
| `features/` | 61 FP64 coordinate-attribution arrays, every native per-state and aggregate table |
| `calibration/` | Both genuine source states, 16 FP32 gradient shards, parameter mappings, norms and calibrated rates |
| `results/functional-inference/` | Nine primary contrasts, all rotations, 240 predictions, complete input tables and independent verification |
| `results/functional-sensitivity/` | All 24 exploratory comparator tests, 1,440 predictions and independent verification |
| `provenance/` | Original feature/calibration completion and readback evidence |

The snapshot contains no raw example text or executable source. All primary
model checkpoints are separately restored through [the checkpoint guide](checkpoint-restoration.md).
The feature arrays are measurements on the fixed eight-candidate probe, not a
full-corpus compressed-retrieval benchmark. Calibration gradients are not
continuation optimizer-state initialization.

## Download and verify

Use the standalone [restoration script](../scripts/restore_functional_analysis.py).
Downloading requires `huggingface_hub`; verification needs only standard Python.
No credentials, GPU, model imports or pickle deserialization are used.

```bash
HF_HUB_DISABLE_PROGRESS_BARS=1 python scripts/restore_functional_analysis.py \
  download --path /data/dense-functional-NEW
```

The destination must not exist; failed attempts are preserved. The script prints
the resulting snapshot directory. Verify a previously downloaded copy with:

```bash
python scripts/restore_functional_analysis.py verify --path /path/to/snapshot
```

This script is currently available in the local worktree; a remote GitHub source
release is still pending. The HF data link above is already public. Original
absolute paths in receipts are provenance, not assumed paths on the new host.
Byte verification and the successful same-host download are not a claim of
cross-host native scientific admission or GPU resume equivalence.
