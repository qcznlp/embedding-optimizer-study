# Exact owned document-component calls

- **77334 terminal/exit 0**, observed 12:34 UTC: 93 focused cases.
  `focused-first.xml`, 14,775 bytes,
  SHA-256 5014a8417fb60cfa1cb2f50057f09c500e900160cc0288ae4fe0bb5b036a9046.
- **65097 terminal/exit 0**, observed 12:42 UTC: 154 focused cases.
  `focused-build-first.xml`, 23,767 bytes,
  SHA-256 8a3d0b87d0aadfc70a1e1cbe226c5b1e8edbdaeb06bc9bf50605b04ac62dab7c.
  The subsequently removed unused `os` test import changed no test semantics.
- **3676 terminal/exit 0**, observed 12:43 UTC: complete synthetic document.
  `/tmp/dense-v3-complete-document.I0a0Bj/result.json`, 72,828 bytes,
  SHA-256 da5c7d950e1e2c5daef1343c2c8794609ff0db853f1087f8108210d5870d8202.
  The document component passes actual expansion, full fresh compile and source/
  caption/font checks. No raw numerical replay or primary/factorial admission is
  supplied by this call. The real manuscript and three owned dispatchers are unchanged.
- **5462 terminal/exit 0**, observed 13:05 UTC: whole-development regression.
  XML `/tmp/dense-v3-manuscript-tests.gfHZZN/full-first.xml` is preserved here as
  `full-first.xml`, 420,615 bytes, SHA-256
  a42fbb8140151b2b27176eba18160c61e1d957858e78a7f46d50a45d1d3c7109.
  JUnit reports 2,962 tests, zero failures/errors/skips, elapsed 1,051.058 seconds.
  It serializes 2,953 testcase elements: the previous nine-element difference
  persists and is not a claim of 2,962 unique serialized cases.
  Preserve both new package modules, both new tests and complete_document.py exactly
  as copied under `source-current/`.
  The 12:52 UTC observation confirms the same handle is live, at 65% without a
  reported failure; this percentage is not a completion verdict.

The 12:53 UTC read-only preservation check exits zero: all five source-current
copies match, both trees' original six-file numerical cores are unchanged, and
the actual main.tex, old paper audit/layout and Makefile keep their prior hashes.
The original ledger passes its fixed hash and all three exact owned dispatchers
remain stopped (T). All five new source/test/helper files pass lint/format checks,
and git diff --check reports no whitespace errors. No final scientific acceptance
is inferred from these preservation checks.

No call in this archive is live. Do not poll or restart completed 7103,
77334, 65097, 3676 or 5462. Terminal process completion and XML were both observed.
No aggregate scientific/strict-release acceptance exists for this archive.
