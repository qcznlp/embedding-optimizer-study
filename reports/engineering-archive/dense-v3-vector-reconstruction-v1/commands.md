# CPU-only numerical reconstruction commands

Use `/usr/bin/python`, with the isolated checkout's absolute imports. Do not use the live `.venv`,
old controllers or a previously populated output directory.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Completed test commands

```bash
/usr/bin/python -B -m pytest -q tests/test_primary_v3_vector_reconstruction.py \
  --junitxml /tmp/dense-v3-vector-reconstruction-tests.zaQFHV/focused-retry.xml
/usr/bin/python -B -m pytest -q \
  tests/test_main_resume_repair.py tests/test_primary_v3_vector_reconstruction.py \
  --junitxml /tmp/dense-v3-vector-reconstruction-tests.zaQFHV/order-reproducer-retry.xml
/usr/bin/python -B -m pytest -q \
  --junitxml /tmp/dense-v3-vector-reconstruction-tests.zaQFHV/full-retry.xml
```

These exact output paths are now preserved evidence: choose fresh paths for another run.
The observed counts are 38 focused, 49 ordered-reproducer and 2,419 full cases. The first failed
full attempt remains `full-initial.xml` and the complete source/failure copy is in `attempt-1/`.
No frozen historical test or production source gate was weakened to make the retry pass.

## Full-width fixture and cold audit

The independently labeled fixture generator is `scripts/vector_reconstruction_fixture.py::make`.
Its default kernel is the unchanged actual 768D `compute_state`; `reduced_kernel` is only a unit-
test wiring control. The observed full fixture used every 61 state with seeds `2026090700 + index`,
224 × 768 FP32 queries and 224 × 8 × 768 FP32 documents. All checkpoint authoring is simulated.
The generation receipt binds its source before and after completion; do not regenerate to silently
replace an existing fixture or confuse it with encoded model results.

The current first audit uses `/tmp/dense-v3-vector-reconstruction-cold.osaUb4`. A reproduction
must instead use a new root:

```bash
vector_cold_dir=$(mktemp -d /tmp/dense-v3-vector-reconstruction-cold.XXXXXX)
/usr/bin/python -B scripts/audit_dense_v3_vector_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --fixture /tmp/dense-v3-vector-reconstruction-audit.wM4h97/fixture.json \
  --fixture-sha256 e7850b6ee6fe8acb97d8a49c21dc4e261511efbdce13bda82ccae72e13c09ffe \
  --workdir "$vector_cold_dir"
```

The script launches the actual reconstruction CLI in a fresh process using **only** archived
package sources. It actively blocks network and original-path access, recomputes all states,
and runs separate cold one-unit numerical corruption controls. The first successful receipt
must exist before a replay is launched; provide its externally recorded SHA-256 using `--replay`
and `--replay-sha256` together, with another new empty work directory.

The first audit is terminal/pass. The independent replay was launched at 02:59 UTC using the
following exact invocation (owned session 7506, terminal exit 0 at 03:23 UTC):

```bash
/usr/bin/python -B scripts/audit_dense_v3_vector_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --fixture /tmp/dense-v3-vector-reconstruction-audit.wM4h97/fixture.json \
  --fixture-sha256 e7850b6ee6fe8acb97d8a49c21dc4e261511efbdce13bda82ccae72e13c09ffe \
  --workdir /tmp/dense-v3-vector-reconstruction-replay.9D4FHM \
  --replay /tmp/dense-v3-vector-reconstruction-cold.osaUb4/result.json \
  --replay-sha256 3a5ff804562f772aa29039beee9d03106141f55c98abea12c7c100b7c657212e
```

Do not reuse these populated paths for another run. The observed replay result has SHA-256
`dc74d5e584967c4585138a7eb40d6d9751f9b59b2499f4e8f3efa4a233fa21d3` and 922,134 bytes.
The independent acceptance check, after all source/documentation checks, is:

```bash
/usr/bin/python -B scripts/validate_dense_v3_vector_reconstruction.py \
  --repository /root/embedding-optimizer-story-refactor \
  --fixture /tmp/dense-v3-vector-reconstruction-audit.wM4h97/fixture.json \
  --fixture-sha256 e7850b6ee6fe8acb97d8a49c21dc4e261511efbdce13bda82ccae72e13c09ffe \
  --audit /tmp/dense-v3-vector-reconstruction-cold.osaUb4/result.json \
  --audit-sha256 3a5ff804562f772aa29039beee9d03106141f55c98abea12c7c100b7c657212e \
  --replay /tmp/dense-v3-vector-reconstruction-replay.9D4FHM/result.json \
  --replay-sha256 dc74d5e584967c4585138a7eb40d6d9751f9b59b2499f4e8f3efa4a233fa21d3 \
  --output /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-v3-vector-reconstruction-v1/validation.json
```

An existing acceptance receipt must never be overwritten. Its scope remains synthetic raw-vector
reconstruction, not original-outcome/inference closure, actual primary admission or publication.

## Reconstruction entry point

For a verified future archive, the actual numerical CLI is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/path/to/verified-archive/source/src \
  /usr/bin/python -B -m embed_optim.primary_v3_vector_reconstruction \
  --archive-root /path/to/verified-archive \
  --expected-manifest-sha256 EXTERNALLY_TRUSTED_SHA256 \
  --output /path/to/new-reconstruction
```

An archive hash inferred from untrusted local content does not establish checkpoint-backed
authoring. This command verifies the raw-vector-to-feature portion only; its receipt always
keeps primary scientific admission, checkpoint revalidation, inference and publication false.
No command here releases draft parents, trains/encodes a model or authorizes a remote write.
