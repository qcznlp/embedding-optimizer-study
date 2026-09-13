# Exact owned calls

Observed 2026-09-07 11:36 UTC. Every raw, oracle, test and cold-replay handle below
is terminal. Preserve all six files in `source-current/` and every retained attempt.

- **79178 is terminal/pass**, observed 11:13 UTC, started 10:50 UTC. Complete full-shape synthetic
  exact reconstruction via scripts/audit_dense_v3_exact_reconstruction.py.
  Work directory: `/tmp/dense-v3-exact-reconstruction-cold.LVS9GV`.
  All twelve exact fixture runs/five stages are written, and its local-role payload
  has anchor `ceefc39c128d541032b73ed6bfbd2773fd5f39f6a81e6be6be68da1079695660`.
  All 503 outputs and 15 controls pass; 86 archived package imports. The result is
  1,613,988 bytes, SHA-256 80d0727dc4422b90255027dba4d2c28e433d40667561d438154898bc6a21717a,
  preserved as `complete-first.json`; fixture metadata is `complete-fixture.json`.
- **82844 is terminal/pass**, observed 11:13 UTC, started 10:50 UTC. Whole-development pytest;
  `/tmp/dense-v3-exact-reconstruction-tests.SR979t/full-first.xml` has 2,713 cases,
  zero failures/errors/skips, 1,031.301 seconds. Its preserved `full-final.xml` is
  381,521 bytes, SHA-256 4128879e4e2ba351ef2ef348bd4d0ee87d5a87cc1648a74afe8cf361b750b4a0.
- **70022 is terminal/pass**, observed 11:36 UTC, started 11:13 UTC. Independent complete
  replay in `/tmp/dense-v3-exact-reconstruction-replay.9Gx3RL`, using the externally
  authenticated first result. All 503 outputs match exactly and all 15 controls pass
  again. `complete-replay.json` is 954,060 bytes, SHA-256
  893368c2182c01bd189637df0c02ab1c1c9a489f9e953352e14185cd11cbbc05.
- **84726 is terminal/pass**, observed 10:59 UTC. Independent complete-fixture
  scalar/set and SymPy oracle. Receipt `full-oracle-first.json`, 5,619 bytes,
  SHA-256 dc1eb081419a91db467674fbd7330066472f3c4600057fe37a2d8062e8aaed8b.
- **99774 is terminal/pass**, observed 10:54 UTC. Retained real diagnostic spectrum
  readback; 1,584 records, no tensor loading/SVD. Receipt `real-spectrum-readback.json`,
  8,408 bytes, SHA-256 1568596c508cdff5a2c1ed5156741adf1403349989dd808ec9eed0de1b6e9372.
- **98810 is terminal/pass**: 50 focused cases, zero failures/errors/skips.
  XML is preserved as `focused-final.xml`.
- **98247 is terminal/exit 1**: 50 focused cases, two test-message assertion
  failures. Correct synthetic-admission refusal occurred in both. Exact sources
  are retained under `source-integration-first/`, XML `focused-integration-first.xml`.
  The retry changes only that new assertion and one import-order formatting error.
- **17618 is terminal/pass**: the first 44 primitive tests. Original sources are
  preserved under `source-first/`, XML `focused-first.xml`.
- **91703 is terminal/pass**, observed 11:03 UTC: read-only preservation check.
  All six new source-current files, both six-file original numerical cores, the real
  manuscript and main ledger match. All three exact owned dispatchers remain T.

The preceding publication archive is accepted at its bounded synthetic scope:
64182 and 30075 are terminal/pass. Its acceptance is
1b9044c4506b483d1dddadb54819c0a7ae5809f15103b743ae6d98b5e0ad2e62.
No previous publication/joint/standalone test handle remains live. Do not poll or
restart them. No raw/test/replay handle remains live in this archive.

The final `validate.py` command in commands.md verifies both completed receipts,
independent oracles, all current/archive source bindings, the old numerical cores,
manuscript, ledger and exact stopped dispatchers. Its generated validation.json is
the authority for combined bounded acceptance; it is not a primary-science receipt.
Follow the root handoff for any live validation call. No formal training, controller
transition or remote action is authorized.
