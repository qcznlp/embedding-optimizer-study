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
uv build
uv run embed-optim-audit-distribution
uv run python scripts/portable_evidence.py --audit-only
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
```

Changes to the manuscript, result renderers, or claim logic must additionally preserve the
active Dense-only scope and pass the result-safe paper build and audit:

```bash
uv run embed-optim-render-paper-results \
  --if-ready \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
make -C paper all PYTHON="$PWD/.venv/bin/python"
uv run embed-optim-audit-paper \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
```

Do not hand-edit generated result blocks, tables, figures, manifests, or receipts. Change their
producer and regenerate them so content hashes and source bindings remain auditable. A strict paper
audit is required for publication, but it is expected to remain pending while formal experiments
are incomplete.

Changes to data sampling, negatives, loss logits, checkpoint fractions, task definitions, or score
aggregation alter the experimental contract. Such pull requests must explain whether prior runs remain
comparable and must update the paper's limitations section. Historical LateOn artifacts remain
available for provenance, but they cannot be promoted into the active DenseOn confirmatory claims.

Do not commit API keys, local datasets, model checkpoints, W&B run files, MTEB caches, or PLAID
indexes. The only tracked files under `results/` are the deterministic minimal paper-evidence set
listed in `configs/portable_paper_evidence.json`; update it through its builder and include a reason
in the pull request. The repository `.gitignore` covers the standard locations, but contributors
remain responsible for reviewing their diff.
