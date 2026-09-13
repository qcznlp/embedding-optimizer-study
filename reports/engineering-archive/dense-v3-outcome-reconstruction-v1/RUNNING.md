# Original-outcome reconstruction work in progress

Updated 2026-09-07 04:36 UTC. All numerical/test sessions below are terminal. Full goal remains active.
The bounded [acceptance validator](source-current/scripts/validate_dense_v3_outcome_reconstruction.py)
must finish and produce its own [receipt](validation.json); green tests alone are not that receipt.

Production readers now reconstruct all 49,152 raw validation records, the independent validation
selection, all 840 pinned BEIR task files and all ten original outcome tables. Every upstream
checkpoint/score/timing in the integration fixture is explicitly synthetic. No model is encoded,
no retrieval is executed, no primary run is admitted and no manuscript result is installed.

The first 41-case focused suite passed. The 62-case semantic-control attempt has one test-message
assertion failure; the missing task file was correctly refused. The first strict cold attempt
also correctly refused an old editable-install search path. Both exact attempts are preserved in
[attempt-1](attempt-1/README.md). Only the new test and audit launcher changed; all production
readers, frozen numerical kernels and scientific rules remain byte-identical.

Owned sessions and immutable/new output locations:

- 15997: terminal exit 0, initial 41 tests, `/tmp/dense-v3-outcome-reconstruction-tests.QX7X3H/focused-initial.xml`.
- 2155: terminal exit 1, 62 cases / one assertion failure, `focused-controls.xml` in that directory.
- 14580: terminal exit 1 before numerical work, `/tmp/dense-v3-outcome-reconstruction-cold.EDmj0q`.
- 30230: terminal exit 1; 2,481 cases and the same one pre-fix assertion failure. XML/source/input
  are retained; no other full-suite failure was observed.
- 45289: terminal exit 0; corrected 62-case focused retry, `focused-retry.xml` in the test directory.
- 68209: terminal exit 0; complete cold audit, `/tmp/dense-v3-outcome-reconstruction-cold.GsLOPE`.
  All raw-outcome reconstruction and all 21 semantic refusals pass.
- 40528: terminal exit 0; independent replay, `/tmp/dense-v3-outcome-reconstruction-replay.2PrtN4`.
  Exact child result equality and all 21 refusals pass again. Receipt SHA-256
  `ae0d73c8079be783077a660dedfc562a1e16eaede0e2139a4e00ea7ebf858295`.
- 48708: terminal exit 0; corrected full retry has 2,481 cases, no failures/errors/skips,
  `full-retry.xml` in the test directory.
- 80248: terminal exit 0; both cold receipt schemas and their exact equality checked separately.

Do not count a printed summary or existing partial file as terminal completion. Both complete
cold audits and the full regression are finished; only source-bound acceptance is a separate final
step. Its presence and actual verified flags, not a running/finished assumption, define acceptance. Preserve the
original fixture and every failed output. Do not repeat the already
accepted 61-state raw-vector audits, change old hashes, or use these simulations for paper claims.

The old accepted vector parent and four front documents are preserved in `before/`. The initial
production smoke sources remain in `source-initial/`, initial unit tests in `initial-tests/`,
the failed audit/test sources in `attempt-1/source/`, and corrected sources in `source-current/`.
No formal training, deployment, controller transition, commit/push, HF deletion or protected-helper
action is authorized or performed by this work. Follow the root [agent instructions](../../../AGENTS.md).
