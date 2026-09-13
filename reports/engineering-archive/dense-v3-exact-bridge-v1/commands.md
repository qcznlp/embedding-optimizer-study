# CPU-only reproduction commands

The named namespaces are retained evidence. A repeat must use new empty directories and new
receipt paths; do not overwrite or regenerate the frozen protocol. No command below deploys a
source tree, resumes a controller or launches a model/GPU worker.

```bash
cd /root/embedding-optimizer-story-refactor
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

python -B scripts/prepare_dense_v3_exact_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --output configs/dense_primary_v3_exact_bridge_protocol.json

python -B -m pytest -q -p no:cacheprovider \
  tests/test_exact_bridge_measurements.py tests/test_primary_v3_exact_bridge.py \
  --junitxml=/tmp/dense-exact-bridge-preparation.VGv9YL/focused-final.xml
python -B -m pytest -q -p no:cacheprovider \
  --junitxml=/tmp/dense-exact-bridge-preparation.VGv9YL/full-tests.xml

python -B scripts/audit_dense_v3_exact_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-exact-bridge-audit.XoMIzZ \
  --expected-protocol-sha256 270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc

python -B scripts/audit_dense_v3_exact_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir /tmp/dense-v3-exact-bridge-replay.gJuW7Z \
  --expected-protocol-sha256 270a908d4242a831ce239f6d6258a19b3884329646f076c0af43ab968e5c9fdc \
  --replay /tmp/dense-v3-exact-bridge-audit.XoMIzZ/result.json \
  --replay-sha256 a80544e0cd5fc3355fbc92b9a09a20e96b03feecaffb012a79d9f460e1c3371d

python -B scripts/validate_dense_v3_exact_bridge.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-exact-bridge-audit.XoMIzZ/result.json \
  --audit-sha256 a80544e0cd5fc3355fbc92b9a09a20e96b03feecaffb012a79d9f460e1c3371d \
  --replay /tmp/dense-v3-exact-bridge-replay.gJuW7Z/result.json \
  --replay-sha256 8096eea0b558094d04262c1fc0fdeebd7f1d179969f8af3ca75aab8541dfb99d \
  --output reports/engineering-archive/dense-v3-exact-bridge-v1/validation.json
```

The first focused run used only `tests/test_exact_bridge_measurements.py` and retained
`first-focused.xml` (23 passing cases). The complete selection passes 48; the full suite passes
2,153 with no failures/errors/skips. The audit and independent-process replay both exit zero.
SymPy full augmented systems provide the independent reference, not another call to the same
FWL kernel. The accepted prior reference-helper failure remains in its prior archive unchanged.

The real diagnostic mapping reads only the already accepted checkpoint exact-geometry CSV at
`/tmp/dense-v3-exact-geometry.CBVst9/checkpoint_exact_geometry.csv`, authenticating it against
the immutable prior acceptance. It does not read new checkpoint tensors or add retrieval outcomes.

Artifact-only status refresh, using the same absolute isolated imports:

```bash
cd /root/embedding-optimizer-study
python -B -m embed_optim.corrected_progress \
  --output /root/embedding-optimizer-story-refactor/CURRENT_PROGRESS.json
```
