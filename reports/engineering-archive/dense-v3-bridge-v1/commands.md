# CPU-only reproduction and retained executions

Run from the isolated checkout with absolute source paths. Existing namespaces and receipts below
are evidence: use new empty namespaces for a repeat, never overwrite them. No command here resumes
the original controller or launches a training/GPU worker.

```bash
cd /root/embedding-optimizer-story-refactor
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -B scripts/prepare_dense_v3_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --output configs/dense_primary_v3_bridge_protocol.json

python -B -m pytest -q -p no:cacheprovider \
  tests/test_bridge_numerics.py tests/test_primary_v3_bridge.py \
  --junitxml=/tmp/dense-bridge-numerics.DE6nMV/focused-final.xml
python -B -m pytest -q -p no:cacheprovider \
  --junitxml=/tmp/dense-bridge-numerics.DE6nMV/full-tests.xml

python -B scripts/audit_dense_v3_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-bridge-retry.kWkgar \
  --expected-protocol-sha256 540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803

python -B scripts/audit_dense_v3_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-bridge-replay.osgmcN \
  --expected-protocol-sha256 540c3c3041ea13d9029fb5534283187b4f4141dfc66eba668938a7dc7d8ae803 \
  --replay /tmp/dense-v3-bridge-retry.kWkgar/result.json \
  --replay-sha256 3fbfd13a2af746cc6ee73caba639599de90400e67106aecef4809d323cef4a22

python -B scripts/validate_dense_v3_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-bridge-retry.kWkgar/result.json \
  --audit-sha256 3fbfd13a2af746cc6ee73caba639599de90400e67106aecef4809d323cef4a22 \
  --replay /tmp/dense-v3-bridge-replay.osgmcN/result.json \
  --replay-sha256 c4316000384ea2e0e6cf2182e55a2719d88526f0596099a90e8a9c0720a59da2 \
  --output reports/engineering-archive/dense-v3-bridge-v1/validation.json
```

The first audit used identical protocol/arguments except workdir
`/tmp/dense-v3-bridge-audit.6xH3YX` and the preserved reference-wrapper source in `attempt-1/source/`.
It exits 1 for unsupported SymPy Boolean-to-int conversion. The retry changes only reference rank
counting and adds three exact reference controls; no production analysis source or tolerance changes.
All old attempts and test receipts remain preserved, including the initial 28 and intermediate
52 passing focused checks. The final focused/full counts are 54 and 2,105.

SymPy is the independent full-system reference; SciPy's pivoted-QR least squares supplies additional
well-conditioned numerical controls in tests. Production exact OLS uses only Python rational/decimal
arithmetic plus NumPy rank diagnostics. These checks do not imply a cross-platform bitwise guarantee
for NumPy's numerical-resolution boundary. The protocol remains draft, not execution authority.

The artifact-only progress refresh runs from the live checkout using the same isolated imports:

```bash
cd /root/embedding-optimizer-study
python -B -m embed_optim.corrected_progress \
  --output /root/embedding-optimizer-story-refactor/CURRENT_PROGRESS.json
```
