# Exact owned publication calls

Observed 2026-09-07 12:16 UTC. No handle in this archive remains live.

- **74668 is terminal/pass**: all 57 initial focused cases, zero failures/errors/skips.
  XML `/tmp/dense-v3-exact-publication-tests.5ncZ0V/focused-first.xml` is preserved here.
  The five formatted package/prepare/test sources are retained under `source-first/`.
- **25120 is terminal/pass**: first full generation from the completed exact reconstruction's single
  synthetic population, in `/tmp/dense-v3-exact-publication-full.ZN9rJB`. Its script
  source is preserved under source-first. Result SHA-256
  6a7366d7647f1fa155a7966b79d35b6dc10e5db5f900d8ec50fb419b3ad49fb5 (52,158 bytes).
- **24933 is terminal/exit 1**: two source-timing counterexamples, both correctly
  exposing missing rejection in the first new adapter. Sources/XML are retained
  under source-race-first and race-first.xml. Only the new adapter's snapshot timing changed.
- **97653 is terminal/pass**: the expanded 65-case retry; focused-retry.xml.
- **3463 is terminal/pass**: all 95 final focused cases; focused-final.xml.
- **41684 is terminal/pass**: complete generation after the timing fix, in
  `/tmp/dense-v3-exact-publication-retry.lioRRV`. All ten output files equal the first
  generation. Result SHA-256 2363e68ee559dfa7235eb395a45f7c78d2883beaa5a174c979b0703fbb962e63
  (52,280 bytes), preserved as complete-retry.json.
- **94532 is terminal/exit 1**: PDF generated, then the old layout reader rejected
  `tab:exact-geometry-sensitivity` as an unexpected float. Its workdir is
  `/tmp/dense-v3-exact-publication-layout.OwZUxX`; exact PDF/logs are retained in layout-first/.
- **33605 is terminal/pass**: complete v3 layout retry, in
  `/tmp/dense-v3-exact-publication-layout-retry.E2mqpj`. Layout SHA-256
  33a957d993fb8da1c658c6265875b8171a3d2d0f9db5579b54d0f940fb416819 (16,973 bytes).
  Main end page 7, conservative abstract 178/200, eight required labels, no overfull/Type 3.
  Pages 1/5/9 were visually inspected; this is explicitly synthetic, not the real paper.
- **75196 is terminal/pass**: generated the preparation-only source-bound protocol,
  e8409f534931b85ad855fbdd3804e3ef8d771b0ee22ee99d9b81bf07d605f83e (12,363 bytes).
- **7103 is terminal/exit 0**, observed 12:15 UTC. The complete XML is preserved as
  `full-first.xml`, 397,128 bytes, SHA-256
  c2bd40e788c9c440f6936010c936a28c76c6d78bfc9d8b6f4774d380400bbc34.
  JUnit reports 2,808 tests, zero failures/errors/skips, 1,054.795 seconds. Its actual
  testcase-element count is 2,799; the preceding full milestone also has a nine-element
  discrepancy (2,713 reported / 2,704 elements). The count difference is retained,
  not silently treated as 2,808 distinct serialized cases. No rerun is required merely
  because this completed handle was previously recorded as live. Preserve source-current.
- **55237 is terminal/pass**, observed 12:04 UTC: all ten source-current files
  (nine code/test/helpers plus the protocol) match the running checkout; both original
  six-file numerical cores, the actual manuscript and original ledger are unchanged.
  The new contract loads, and all three exact owned dispatchers remain stopped (T).
  All nine current code/test/helper files also pass lint and format checks.

No real checkpoint-backed positive authoring or new raw reconstruction is claimed. The preceding
exact raw reconstruction archive is accepted at SHA-256
d9888b172648555b54895c8a0b0acc751a3b8a345b7872ebf929ebf7038f5c59.
79178, 82844, 70022 and 78568 are all terminal/pass; never restart them as missing work.
Formal runs, the strict complete active-v3 paper consumer, routed factorial, reviewed
source/runtime/release transition and remote publication remain separate requirements.
