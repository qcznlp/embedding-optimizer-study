# Initial input-audit wrapper refusal

The actual CPU command below exited 1 (owned session 22566) before any upstream task completed:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
  OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -B scripts/audit_dimension_probe_inputs.py \
  --repository /root/embedding-optimizer-story-refactor \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --output /tmp/dense-v3-dimension-probe.2d7A6Y/result.json
```

`upstream_files -> binding -> primary_contract.file_identity` raised
`ValueError: Require an ordinary source/payload file`. An HF snapshot entry is a
symlink to a cache blob, and the wrapper passed the unresolved entry despite
displaying its resolved path. No `result.json` was produced; no upstream text
verification, model computation, data edit or scientific finding is claimed.
The exact executed source is retained in `source/scripts/`.

The retry resolves only the declared HF-cache source role with `strict=True`,
then applies the unchanged ordinary-file guard and compares the actual blob
size/SHA-256 with independent metadata at the immutable revision. Local probe
directories still reject symlinks. No data, scientific rule or tolerance changes.
The retry uses a new output directory, not this failed path.
