# Contributing

Contributions that improve correctness, reproducibility, or throughput are welcome.

1. Open an issue describing the proposed protocol or implementation change.
2. Create a focused branch and include tests for deterministic or numerical behavior.
3. Recreate the locked contributor environment and run the release checks below.
4. Report the exact hardware, package versions, seed, configuration diff, and before/after throughput
   for performance changes.

From the repository root:

```bash
uv sync --extra dev --extra eval --extra analysis
uv run cffconvert --validate --infile CITATION.cff
uv pip install --python .venv/bin/python --no-config --require-hashes --torch-backend cu129 --overrides requirements-formal.lock -r requirements-formal.lock
uv pip install --python .venv/bin/python --no-config --no-deps --require-hashes --no-build-isolation-package flash-attn -r requirements-formal-flash.txt
.venv/bin/python -m embed_optim.runtime --spec configs/formal_runtime.json
uv build
uv run --no-sync embed-optim-audit-distribution
uv run --no-sync python scripts/portable_evidence.py --audit-only
uv run --no-sync python scripts/test_source_roles.py --output /tmp/dense-source-tests-new
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
```

Use a separate environment with the CUDA 12.9 compiler available for the genuine
FlashAttention build; no GPU is needed for these CPU regressions. CI installs
the compiler packages explicitly. Do not install placeholder package metadata or
skip the CUDA extension build. After the formal pins are installed,
`uv run --no-sync` prevents the broader development lock from replacing them.
Never change the live study environment to prepare contributor checks.
The same hashed lock supplies the original formal version overrides, including
Torch 2.9.1+cu129 despite fast-plaid's 2.9.0 dependency declaration. An unhashed
constraints override is not sufficient in require-hashes mode. This reproduces
the existing recorded environment, not a change to its package pins.

The test output directory must be new and outside the checkout. The runner executes
every test module exactly once across the current source and two authenticated
historical source roles. It does not skip failed cases or replace frozen hashes.
See [source-version testing](docs/source-version-testing.md). A bare `uv run pytest`
mixes intentionally different source versions and is not the full-suite entry.

Changes to the manuscript, result renderers, or claim logic must additionally preserve the
active Dense-only scope and pass the complete numerical-to-reviewed-paper build.
Install `pdflatex`, `bibtex`, `pdfinfo`, `pdftotext` and `pdffonts`, then run:

```bash
make -C paper release PYTHON="$PWD/.venv/bin/python" \
  NUMERICAL_BUNDLE="$PWD/reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed" \
  RELEASE_OUTPUT=/tmp/dense-complete-paper-new
```

Do not hand-edit generated result blocks, tables, figures, manifests, or receipts. Change their
producer and regenerate them so content hashes and source bindings remain auditable. A strict paper
audit is required for publication. The complete numerical graph and reviewed manuscript
are now available; [the versioned entry](docs/versioned-paper-reproduction.md) preserves
their distinct source identities. The historical renderer/audit and `make legacy-release`
remain historical regression interfaces, not substitutes for this current gate.

Changes to data sampling, negatives, loss logits, checkpoint fractions, task definitions, or score
aggregation alter the experimental contract. Such pull requests must explain whether prior runs remain
comparable and must update the paper's limitations section. Historical LateOn artifacts remain
available for provenance, but they cannot be promoted into the active DenseOn confirmatory claims.

Do not commit API keys, local datasets, model checkpoints, W&B run files, MTEB caches, or PLAID
indexes. The only tracked files under `results/` are the deterministic minimal paper-evidence set
listed in `configs/portable_paper_evidence.json`; update it through its builder and include a reason
in the pull request. The repository `.gitignore` covers the standard locations, but contributors
remain responsible for reviewing their diff.
