# Contributing

Contributions that improve correctness, reproducibility, or throughput are welcome.

1. Open an issue describing the proposed protocol or implementation change.
2. Create a focused branch and include tests for deterministic or numerical behavior.
3. Run `pytest`, `ruff check src tests scripts/eval`, and `ruff format --check src tests scripts/eval`.
4. Report the exact hardware, package versions, seed, configuration diff, and before/after throughput
   for performance changes.

Changes to data sampling, negatives, loss logits, checkpoint fractions, task definitions, or score
aggregation alter the experimental contract. Such pull requests must explain whether prior runs remain
comparable and must update the blog's limitations section.

Do not commit API keys, local datasets, model checkpoints, W&B run files, MTEB caches, or PLAID
indexes. The repository `.gitignore` covers the standard locations, but contributors remain
responsible for reviewing their diff.
