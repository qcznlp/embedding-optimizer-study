# Complete third-stage evaluation artifact backup

All twelve declared DenseOn optimizer/rate configurations at step 2345 now have
an immutable, data-only public snapshot: **557 files / 1,959,113 bytes**.
Full anonymous recovery and independent reconstruction of **168 raw task scores,
168 original successful workers, twelve exact means and 180 CSV rows** pass.
Only six pool-A states / 84 values are newly backed up; six pool-B states overlap
the unchanged earlier snapshot. Overlap is not another experiment.

- Dataset: `qcz/embedding-optimizer-study-analysis-artifacts`.
- Revision: `fdad53c239d9ed59141fcafbd92a0438bb1a657f`.
- Parent revision: `dbd16adcc835c2c6b7537a61b8d511f027a2c351`.
- Manifest SHA-256: `4014de3241535c339d29e7d08d3bc2901597db546440218f78b44ac4fc6dce34`.

Use the [recovery guide](../../../docs/third-stage-evaluation-restoration.md) and
[standalone stdlib reader](source/verify_recovered.py). The reader is local code,
not part of the public data snapshot or an updated GitHub release.

## Actual execution

1. The [new adapter](source/backup.py) selects all twelve step-2345 states from the
   accepted [48-state readback](../dense-v3-complete-third-stage-evaluations-v1/README.md).
   It authenticates the actual fourteen-payload archive schema, both original queues,
   the complete four-stage population and the independent 672-score readback.
   Source/checkpoint membership is fixed, never selected using scores.
2. **46 adapter tests and 18 reader tests pass**, including wrong archive population,
   bad bindings, semantic corruption with recalculated hashes, and old-file
   preservation controls. See [actual tests](actual/tests.json). These are bounded
   backup checks, not GPU tests or independent optimizer experiments.
3. The sole upload completed at **2026-09-12 00:52:30 UTC**.
   Its [remote audit](actual/remote-audit.json) confirms all twenty other root entries
   and eight earlier corrected subtrees unchanged, including root attributes and
   the repository card. Every operation was a regular new-file addition.
4. Complete anonymous recovery passed at **2026-09-12 00:59:24 UTC**.
   The byte-identical archived reader ran with `python -I -B`, from the temporary
   working directory outside the project, and passed at **2026-09-12 00:59:53 UTC**.
   It reads only the supplied recovered files and imports only the standard library.

The download encountered HTTP 429 responses. The original client continued with
its built-in backoff and finished with exit zero; no second upload, alternate
identity or manual download restart occurred. [commands.json](commands.json)
retains the actual returned chunks, including one explicitly truncated chunk.
It is not described as a complete HTTP transcript. The harmless anonymous-client
warning is also preserved. Original upload and download receipts are not rewritten.

## Scope and remaining work

The [manifest](actual/artifact_manifest.json) binds all native files, worker records
and data tables. This is same-physical-host data recovery and score reconstruction,
not model re-execution, a physical second-host experiment, new statistical inference,
source publication or a mechanism finding. Per-query ranking traces are not included.

All **48 currently complete checkpoint outcomes** now have off-host score backups:
all twelve configurations at steps 782, 1563, 2345 and 3907. The twelve step-3126
checkpoints still need complete fourteen-task outcomes. At the separately timestamped
**2026-09-12 00:59 UTC** snapshot, primary evaluation is **676 / 840**, with eight
exact original workers live. No original evaluator or its numerical source changed.
See [observations.json](observations.json).

Functional analysis remains one accepted pretrained vector state and zero feature
states; the original failure is preserved, and recovery is neither authorized nor
launched. Formal crossed continuation remains 0 / 12. No manuscript or GitHub
publication changed. Historical HF deletion and protected-helper limits are unchanged.

The final local evidence check is [verification.json](verification.json).
The [previous handoff](before/CURRENT_EXPERIMENT.md) is preserved byte-for-byte.
See [CURRENT_EXPERIMENT.md](../../../CURRENT_EXPERIMENT.md) for whole-goal work.

