# CPU-only preparation commands

Use the pinned `/usr/bin/python`, not the live `.venv`. All commands below are from the isolated
development checkout. They do not train, change controllers, encode primary models or write paper
includes. Existing outputs are evidence; reproduction requires newly allocated directories.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

The prepared draft was generated into a new file, authenticated, and copied byte-for-byte into
`configs/dense_primary_v3_dimension_inference_protocol.json`. Its SHA-256 is
`d66274878d90e463b74a03cf6e09518ae01dcddfe76584542e3bece435713545`.

```bash
/usr/bin/python scripts/prepare_dense_v3_dimension_inference.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --output /tmp/dense-v3-dimension-inference-tests.uzVGkM/protocol.json
```

The first failed oracle run remains `/tmp/dense-v3-dimension-inference-audit.TgRDmM` with no
result receipt. Its fixture SHA-256 is
`dddd7d07bb463b950a32f1bef80054d20d2f9753d6872d60cb6816b30c0ab371`.
The successful audit used a different empty directory:

```bash
/usr/bin/python scripts/audit_dense_v3_dimension_inference.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --protocol /tmp/dense-v3-dimension-inference-tests.uzVGkM/protocol.json \
  --expected-protocol-sha256 d66274878d90e463b74a03cf6e09518ae01dcddfe76584542e3bece435713545 \
  --workdir /tmp/dense-v3-dimension-inference-retry.cMGAjM
```

Fresh-process replay used the same arguments except `--workdir` and the two added parent flags:

```bash
--workdir /tmp/dense-v3-dimension-inference-replay.CoE2it \
--replay /tmp/dense-v3-dimension-inference-retry.cMGAjM/result.json \
--replay-sha256 5b3f2e3db1ec345239e98cbd0bee6a21e78e9d85cf6dcb73994b9177c2862268
```

The focused test command covers 64 cases:

```bash
/usr/bin/python -m pytest -q \
  tests/test_dimension_inference.py \
  tests/test_primary_v3_dimension_inference.py \
  tests/test_dimension_inference_reference.py \
  --junitxml=/tmp/dense-v3-dimension-inference-tests.uzVGkM/focused-retry.xml
```

The isolated full suite was run with the same explicit imports/interpreter and a new receipt:

```bash
/usr/bin/python -m pytest -q \
  --junitxml=/tmp/dense-v3-dimension-inference-tests.uzVGkM/full-initial.xml
```

The audit itself records both actual CLI refusal commands and the standalone LaTeX compilation
command. No command here releases the draft or overrides the outstanding owner decisions.
