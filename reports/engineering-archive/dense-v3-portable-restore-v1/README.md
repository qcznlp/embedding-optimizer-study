# Scoped restoration add-on

2026-09-13. The restoration procedure is now available as the separately named
`embed_optim_restore` package. See [usage](../../../docs/training-restoration.md).
This closes the local packaging task, not full source publication or a general
new-host training launcher. No new GPU experiment was run for this integration.

## Implemented behavior

The adapter authenticates one of two genuine step-313 checkpoint identities,
requires the original four-rank Trainer/source/resource admission, verifies
native collated inputs and loaded model/optimizer/scheduler state, places named
Adam counters with their parameters, and repeats the original first four
microbatches before clipping. Exact local-gradient, post-DDP-gradient and RNG
checks must pass against immutable recorded references before native updates.
Failure stops the call; no automatic retry or tolerance is provided. Callbacks
and clipping hooks are removed on normal or exceptional exit.

The numerical helpers retain the computation ASTs of the original actually
executed helpers. The trace import is package-relative; only three independent
stdlib imports and docstring indentation require explicit narrow normalization.
The recursive bitwise comparison function is unchanged. Sixteen JSON references
are byte-identical copies of the original two-case/four-rank cold/warm records.
There are no large gradient arrays, producer-path fallbacks or network requests.

The public `resume` wrapper requires final step 391, but reports
`endpoint_compared=false`: independent endpoint comparison is a separate check.
The original [two real GPU continuations](../dense-v3-warm-reducer-recovery-v1/README.md)
already pass full bitwise endpoint equality. The new outer wrapper has CPU
integration coverage; it was not separately rerun on GPUs. Coverage is not
extended to NorMuon, arbitrary checkpoints, a physical second host or all drivers.

## Actual tests and retained failures

| Execution | Observed terminal result |
| --- | --- |
| Initial 54 tests | 78877 / 282bc3 / exit 0 |
| Expanded 83 tests, first run | 47481 / 45215f / exit 1: two source-fidelity controls |
| Final 83 tests | 33836 / e18651 / exit 0 |
| Package build | 15576 / 4b3056 / exit 0 |
| Original full distribution audit | 25a535 / exit 0, complete, no findings |
| Additional exact-member and isolated-wheel execution | 44123 / 1fe9ef / exit 0; child exit 0 |

The two expanded-test failures were formatter-only AST comparisons: independent
stdlib import ordering and indentation of an original function docstring. Their
preimages and failed JUnit file remain in this archive. The final controls verify
exact original imports/docstring words and normalize only those two forms; they
do not ignore arbitrary statements or change numerical behavior. Ruff's initial
failure (3e22f4 / exit 1) and later successful format/check (149af5 / exit 0,
eb963b / exit 0) are retained as observed terminal history, not full captured logs.
Whitespace check 296b69 exits zero.

All 83 cases cover original trace, native Trainer normalization, exact reference
gates, counter placement and new orchestration/refusal/cleanup behavior. Five
orchestration tests use actual CPU autograd through pinned native
`Trainer.training_step`, including stochastic inputs and unchanged-state/RNG
checks. Their synthetic CPU model and patched GPU preflight are not DDP proof.

The actual wheel execution verifies all **25 new namespace files** (nine Python,
sixteen references) against wheel and sdist bytes, authenticates all eight
case/rank reference pairs, and runs the five CPU integration tests. All eight
loaded add-on modules come from the extracted wheel. Seven original producer
roots are explicitly denied and each denial is exercised as a negative control;
there are zero subsequent forbidden reads. The original `embed_optim` package
is not imported, CUDA is not initialized, and network access is refused by the
Python audit hook. This is same-host isolated Python execution, not an OS sandbox
or physical second-host test. See `actual/cold-result.json` and `actual/cold.log`.

The original full distribution audit was not weakened or replaced by the new
namespace-specific check. Its source SHA is
`b802f276e30887d4698195ef4b6cc3b6e576a669224cb7f7c588cdbb1017f383`.
It accepted 210 original modules, 217 data files, 104 scripts, 64 runtime configs
and 50 transitive execution configs. Full build console was tool-truncated and
is not claimed archived. Exact build artifacts and their hashes are retained.

## Remaining release boundary

The add-on does not shadow or rewrite original numerical `embed_optim` modules.
The old 56/60-file analytical source contracts still bind different `config.py`
and `optimizers.py` versions than the validated current primary-training source.
The factorial's old 66-file parent likewise retains its input-module binding.
These need an explicit versioned runtime/consumer composition, not new hashes
pretending to be old identities. Complete scientific replay and all 24 scientific
runs remain finished. Do not rerun them as missing work.

`before/` preserves the prior handoff/package configuration. `source/`, `tests/`
and `actual/` preserve the implemented package, tests, actual artifacts and
executed wheel-check scripts. `manifest.json` binds every archived file except
itself. Historical archives, original protocols, scientific states and paper
content are unchanged. No protected-helper access, process inspection, runtime
upgrade, remote write or deletion occurred.
