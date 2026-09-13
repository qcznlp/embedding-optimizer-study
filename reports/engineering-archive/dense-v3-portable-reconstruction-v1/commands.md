# CPU-only reproduction record

Run only from `/root/embedding-optimizer-story-refactor` with `/usr/bin/python`. The live `.venv`
is not the pinned interpreter. Use new output directories and retain failed attempts. These
commands do not train, re-encode, deploy, publish, alter controllers or complete the paper.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
```

## Source/transport audit and independent replay

The actual first output root is `/tmp/dense-v3-portable-audit.NG58fO`. For a new audit:

```bash
portable_audit_dir=$(mktemp -d /tmp/dense-v3-portable-audit.XXXXXX)
/usr/bin/python -B scripts/audit_dense_v3_portable_sources.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir "$portable_audit_dir"
```

The original receipt is 366,379 bytes, SHA-256
`cd76f91b9b3470eaa5f062cdcbd00fa7e6bd8a9739049e7951cb440d72cc4d8a`.
Its observed audit passes. It distinguishes actual source-only contract loading from a separately
marked positive fixture with simulated upstream admission; no primary vectors or outcomes are admitted.

The actual independent replay root is `/tmp/dense-v3-portable-replay.pl7lYH`. Replaying that frozen audit:

```bash
portable_replay_dir=$(mktemp -d /tmp/dense-v3-portable-replay.XXXXXX)
/usr/bin/python -B scripts/audit_dense_v3_portable_sources.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /tmp/dense-identity-source.8rUgLF \
  --workdir "$portable_replay_dir" \
  --replay /tmp/dense-v3-portable-audit.NG58fO/result.json \
  --replay-sha256 cd76f91b9b3470eaa5f062cdcbd00fa7e6bd8a9739049e7951cb440d72cc4d8a
```

The retained replay receipt is 365,083 bytes, SHA-256
`6b82460148d777223d67ee60bbb0218c481a3c5957909cb949995e00f9f83992`.
It verifies the first audit's exact source and artifact bindings before replaying. Both fresh child
results record exit code 0 and identical 12 expected training identities / 69 archived imports.
Neither replay reconstructs model vectors, dimension statistics or paper evidence.

## Regression checks

```bash
portable_tests_dir=$(mktemp -d /tmp/dense-v3-portable-tests.XXXXXX)
/usr/bin/python -B -m pytest -q \
  tests/test_reconstruction_files.py \
  tests/test_primary_v3_reconstruction_sources.py \
  --junitxml "$portable_tests_dir/focused-final.xml"
/usr/bin/python -B -m pytest -q \
  --junitxml "$portable_tests_dir/full-final.xml"
```

Observed focused receipt `/tmp/dense-v3-portable-tests.KwcrUJ/focused-final.xml`: 47 tests,
22.470 seconds, no failures/errors/skips. Full receipt
`/tmp/dense-v3-portable-tests.KwcrUJ/full-initial.xml`: 2,381 tests, 313.660 seconds,
no failures/errors/skips; copied without editing as this archive's `full-final.xml`.
Earlier green receipts `focused-initial.xml` (46 cases) and `builder-initial.xml` (one added
case) remain in the same temporary directory. No failed test/audit is omitted from this milestone.

After the final handoff documentation edits, run and retain these existing regression cases:

```bash
/usr/bin/python -B -m pytest -q \
  tests/test_config.py::test_public_docs_record_compute_and_acceleration_contract \
  tests/test_wandb_dense_provenance_audit.py::test_publication_contract_requires_wandb_before_distribution_and_tracks_receipt \
  tests/test_paper_audit.py::test_current_dense_paper_constants_match_strict_sources \
  tests/test_dense_trainer_normalization_candidate.py::test_stack_guard_verifies_actual_pinned_source_bytes \
  --junitxml "$portable_tests_dir/final-document-regressions.xml"
```

## Acceptance

Once the exact receipts, source copies and final documentation exist, the validator writes a new
receipt only; it must not overwrite a previous acceptance:

```bash
/usr/bin/python -B scripts/validate_dense_v3_portable_sources.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-portable-audit.NG58fO/result.json \
  --audit-sha256 cd76f91b9b3470eaa5f062cdcbd00fa7e6bd8a9739049e7951cb440d72cc4d8a \
  --replay /tmp/dense-v3-portable-replay.pl7lYH/result.json \
  --replay-sha256 6b82460148d777223d67ee60bbb0218c481a3c5957909cb949995e00f9f83992 \
  --output reports/engineering-archive/dense-v3-portable-reconstruction-v1/validation.json
```

The audit includes only the existing narrow reader of the exact stopped study handles. Never
replace it with broad process/GPU inspection. This engineering acceptance intentionally leaves
raw-vector numerical reconstruction, complete primary publication, formal replication and
scientific completion false. No parent protocol is released by changing a status or hash.
