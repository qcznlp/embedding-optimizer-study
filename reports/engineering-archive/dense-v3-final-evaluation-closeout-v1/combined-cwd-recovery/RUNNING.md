# Source-identical combined replay with explicit component working directory

Owner authority: the active DenseOn paper/reproducibility goal and the direct
instruction authorizing all task work to finish promptly. This is a reviewed
execution-context correction, not an automatic retry or a new experiment.

The original combined session 56449 failed (ef4678, exit one) at its unchanged
zero-refused-I/O check. The nested primary replay itself exited zero and
reconstructed the primary graph, but recorded 48 refused probes of two
nonexistent editable-import path-hook names resolved against the coordinator
working directory, outside the component bundle/output. Both names occur 24
times; neither path exists. The failure and all diagnostics remain intact.

Run the original immutable closed replay from its own primary component
directory. The only intentional caller changes are working directory and new
output destination. The original manifest, numerical functions, source,
I/O audit hooks, zero-refusal requirement, no-network requirement, input
coverage, exact comparisons and strict complete-document checks are unchanged.
No denied path is allowlisted or observation filtered. Both old producer
directories and network remain forbidden. No GPU or external write is used.

Bundle: `/tmp/dense-v3-combined-paper-replay.URO4uk9L/closed`.
Manifest SHA: `746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7`.
Working directory: that bundle's `primary` subdirectory.
Output: this new directory's `actual` subdirectory (must not already exist).

This attempt is incomplete until the original process and all native complete
numerical/PDF and I/O checks succeed. Keep failed outputs if any; no automatic
retry, new tolerance or source change is authorized by this local record.
