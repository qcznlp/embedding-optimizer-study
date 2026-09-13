# Exact source-preparation and training-check commands

These are preparation/CPU checks, not permission to execute a DRAFT protocol.
Use new output locations for a deliberate independent replay; do not repeat
completed preparation as missing work.

The new `prepare.py` copied unchanged audited source into
`/tmp/dense-v3-primary-launch.pztjAW/assembled-first`, then the durable working
snapshot `/root/embedding-optimizer-primary-v3`, using:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /usr/bin/python -B \
  reports/engineering-archive/dense-v3-primary-launch-v1/prepare.py \
  --repository /root/embedding-optimizer-story-refactor \
  --candidate /tmp/dense-identity-source.8rUgLF \
  --output /tmp/dense-v3-primary-launch.pztjAW/assembled-first
```

The relevant tests ran from the first assembled root with CUDA hidden, Python
bytecode disabled, OMP/OPENBLAS/MKL thread counts one, HF_HUB/HF_DATASETS/TRANSFORMERS
offline flags one, W&B disabled, and PYTHONPATH explicitly its `src` and root:

```bash
/usr/bin/python -B -m pytest -q \
  tests/test_dense_numerical_contract.py tests/test_dense_run_contract.py \
  tests/test_losses.py tests/test_optimizers.py tests/test_collators.py \
  tests/test_runtime.py --tb=short \
  --junitxml=/tmp/dense-v3-primary-launch.pztjAW/training-tests-first.xml
```

Under the same environment and working directory, the real-input preflight was:

```bash
/usr/bin/python -B \
  /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-primary-launch-v1/preflight.py \
  --root /tmp/dense-v3-primary-launch.pztjAW/assembled-first \
  --output /tmp/dense-v3-primary-launch.pztjAW/preflight-first.json
```

The natural-data directories were copied with `cp -a --reflink=auto --update=none`
from `/tmp/dense-partition-candidate.kHXoGW/training` and `validation` into the new
experiment's declared `data/denseon-sft-500k-v3` and `data/validation-4096-v3`.
The actual PrimaryV3Contract.dataset method then verified both copies under the
real `/root/embedding-optimizer-primary-v3` source. Full inventories are retained
in the unchanged input/amendment files and the preflight result.

No formal launch command was executed. Do not run the draft entrypoint by editing
its status or disabling committed-source checks. Follow the outstanding local
commit direction and separate training-only release preparation in README.md.
