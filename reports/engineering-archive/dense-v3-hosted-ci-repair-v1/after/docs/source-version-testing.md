# Complete source-version regression tests

The verified matrix contains **3,800 passing cases**: 2,873 current, 733 original
analysis and 194 original factorial cases, with zero failures/errors/skips.
The first current-role attempt's five legacy-entry routing failures are preserved;
all current-role cases were then rerun successfully. Historical roles were not restarted.

From a complete Git checkout with the [formal contributor runtime](../CONTRIBUTING.md)
installed, including the genuine pinned FlashAttention extension:

```bash
uv run --no-sync python scripts/test_source_roles.py --output /tmp/dense-source-tests-new
```

The output must not exist and must be outside the checkout. No GPU is required.
The full suite includes actual native-runtime guards: the broad development lock
alone is not sufficient. Missing FlashAttention or substituted package versions
must fail, not be mocked into successful native-runtime admission.
Document integration tests need `pdflatex`, `bibtex` and Poppler's PDF tools.
The command retains input hashes, source-role assignments, complete logs, JUnit
case inventories and actual child exits. Any failure, error, skipped case,
duplicate case or altered test source prevents success. Failed outputs are never
overwritten. The runner does not launch training, contact W&B or publish files.

Why three source roles? The current numerical training implementation and the
original analysis/continuation consumers have different immutable source locks.
Combining them in one Python process correctly fails those locks. Rewriting the
locks would destroy provenance. Instead, the runner uses separate processes:

| Role | Source | Tests |
| --- | --- | --- |
| Current | The current checkout, with current training and reviewed-paper entries | Every test module not explicitly assigned below, including new tests |
| Original analysis | A private copy with six SHA-256-verified original source/document dependencies | The 27 original analysis/publication modules |
| Original factorial | A private copy with the original executed input module | The three original worker/factory/contract modules |

[The role declaration](../configs/source_test_roles.json) gives every exact source
origin, digest and test assignment. Both historical copies contain byte-identical
current test assertions and unmodified scientific contracts. The current checkout
is never reverted. All test modules are assigned once; new modules default to the
current role. Existing Git-history tests run there with actual Git objects.

One historical 47-command test expresses the newly assembled commands in its
recorded path coordinate system using the existing four-slot `project_steps`
function. All 47 full argument vectors still compare exactly; no interpreter,
scientific input, command, tolerance or expected historical hash is changed.

This verifies the declared implementations, not admission of a mixed-source
fresh training worker, another GPU recovery case or a physical second machine.
The current release entry separately executes the original complete numerical
graph and builds the reviewed document; see [paper reproduction](versioned-paper-reproduction.md).
