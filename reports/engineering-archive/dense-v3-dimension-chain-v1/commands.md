# Exact retained commands and reproduction scope

Working directory: `/root/embedding-optimizer-story-refactor`.
Use the pinned `/usr/bin/python`, not the old live `.venv`; that mistake and its failed receipt
are preserved in [attempt-1](attempt-1/README.md). No packages were replaced.

```bash
export PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
```

The following are actual retained paths. A reproduction must allocate new empty directories with
`mktemp -d` and use new receipt files. No command below trains a model, performs a model forward,
changes a controller or lease, or mutates a remote repository.

```bash
/usr/bin/python -B scripts/prepare_dense_v3_dimensions.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --output /tmp/dense-v3-dimension-chain-tests.MSJXgc/protocol-retry.json

/usr/bin/python -B scripts/audit_dense_v3_dimension_chain.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --experiment-root /root/embedding-optimizer-study \
  --reference /tmp/dense-weight-entry-reference.Um8TxW \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --protocol /tmp/dense-v3-dimension-chain-tests.MSJXgc/protocol-retry.json \
  --output-root /tmp/dense-v3-dimension-chain-retry.W9CZOw

/usr/bin/python -B scripts/audit_dense_v3_dimension_chain.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --experiment-root /root/embedding-optimizer-study \
  --reference /tmp/dense-weight-entry-reference.Um8TxW \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --protocol /tmp/dense-v3-dimension-chain-tests.MSJXgc/protocol-retry.json \
  --output-root /tmp/dense-v3-dimension-chain-replay.yWiGG6 \
  --replay /tmp/dense-v3-dimension-chain-retry.W9CZOw/result.json \
  --replay-sha256 77dad2a11be7c10ada0223c206bf1e69d66e9b8643c0fdabac7a12c1415210fd

/usr/bin/python -B -m pytest -q tests/test_primary_v3_dimension_vectors.py \
  tests/test_primary_v3_dimensions.py \
  --junitxml=/tmp/dense-v3-dimension-chain-tests.MSJXgc/focused-final.xml

/usr/bin/python -B -m pytest -q \
  --junitxml=/tmp/dense-v3-dimension-chain-tests.MSJXgc/full-final.xml
```

The four targeted compatibility regressions use:

```bash
/usr/bin/python -B -m pytest -q \
  tests/test_config.py::test_public_docs_record_compute_and_acceleration_contract \
  tests/test_wandb_dense_provenance_audit.py::test_publication_contract_requires_wandb_before_distribution_and_tracks_receipt \
  tests/test_paper_audit.py::test_current_dense_paper_constants_match_strict_sources \
  tests/test_dense_trainer_normalization_candidate.py::test_stack_guard_verifies_actual_pinned_source_bytes \
  --junitxml=/tmp/dense-v3-dimension-chain-tests.MSJXgc/final-failure-regressions.xml
```

The new-only retry protocol was copied byte-for-byte to
`configs/dense_primary_v3_dimensions_protocol.json`; the original failed draft remains immutable.
The real primary `plan`/`inspect`/feature paths are tested to refuse missing runs before output;
formal `export` refuses the draft before loading a model or acquiring any GPU lease.
Do not invoke the internal synthetic rehearsal patching as a primary-run admission method.

The five real fresh-process CLI refusal controls are retained separately:

```bash
/usr/bin/python -B scripts/audit_dense_v3_dimension_cli.py \
  --repository /root/embedding-optimizer-story-refactor \
  --training-root /root/embedding-optimizer-story-refactor/reports/engineering-archive/dense-full-identity-v1/candidate-source \
  --experiment-root /root/embedding-optimizer-study \
  --reference /tmp/dense-weight-entry-reference.Um8TxW \
  --probe-root /root/embedding-optimizer-study/data/probes/decontaminated-beir-224-seed4242 \
  --protocol /root/embedding-optimizer-story-refactor/configs/dense_primary_v3_dimensions_protocol.json \
  --output-root /tmp/dense-v3-dimension-cli.ueXSs3
```

```bash
/usr/bin/python -B scripts/validate_dense_v3_dimension_chain.py \
  --repository /root/embedding-optimizer-story-refactor \
  --audit /tmp/dense-v3-dimension-chain-retry.W9CZOw/result.json \
  --audit-sha256 77dad2a11be7c10ada0223c206bf1e69d66e9b8643c0fdabac7a12c1415210fd \
  --replay /tmp/dense-v3-dimension-chain-replay.yWiGG6/result.json \
  --replay-sha256 1156600930bfae330d2dacdb18c670da9ed304a1c115010a0c449a901ad51c05 \
  --focused-count 55 --full-count 2270 \
  --output reports/engineering-archive/dense-v3-dimension-chain-v1/validation.json
```

Final acceptance also rehashes all retained parent evidence, source copies, new raw/feature files,
the unchanged six live numerical cores, manuscript and original stopped-controller ledger. It is
still engineering acceptance, not scientific completion or release authorization.
