# Exact owned calls

Observed 2026-09-07 10:32 UTC.

- **40446 is terminal, exit 0**, observed 10:10 UTC: complete synthetic source-isolated
  publication reconstruction, work directory `/tmp/dense-v3-publication-cold.PvOmLl`.
  The complete fixture was built; its manifest anchor is
  485969aa42e0afaa3f46c8ff4ac777b034dc94f9603de2c33f5b6ee6cbf75d21.
  All 61 states, all joint branches and seven publication outputs pass; the complete
  receipt covers 487 outputs and 83 archived package imports. All seven rehashed
  publication-output controls are refused. Result: 1,401,082 bytes, SHA-256
  abf6d68f5eab6d457edec92d25448e7ed4d0a7137db0402ec4eea0a6d36e601c.
  Copies are complete-first.json and complete-fixture.json. Do not poll this handle.
- **64182 is terminal, exit 0**, started 10:12 UTC: independent full replay in
  `/tmp/dense-v3-publication-replay.kVna8I`, anchored to the completed first result.
  All 61 states and 487 outputs match the first full reader exactly; all seven
  rehashed output controls are refused again. Result: 838,882 bytes, SHA-256
  2e51be2d5623bf78e90661a3fa0064aad8429912e943458dc7b479914a56ed0a.
  Preserved as complete-replay.json. Do not poll or restart this completed handle.
- **28989 is terminal, exit 0**: whole-development pytest, XML destination
  `/tmp/dense-v3-publication-reconstruction-full-tests.3kdRv5/full.xml`.
  All **2,663** cases pass, zero failures/errors/skips, duration 1,037.716 seconds.
  The exact 373,752-byte XML is preserved as full-final.xml, SHA-256
  3a16cdc1be0d5d7956a689d17af3974afb4c2d33ae2788d0624037d0d83306cd.
  Do not poll or restart this completed call. Preserve the five bound sources;
  the separate training candidate is not accepted.
- **98303 is terminal, exit 0**: 27 focused cases, zero failures/errors/skips.
  XML `/tmp/dense-v3-publication-transport-tests.cPEgNC/focused.xml` is preserved as
  focused-first.xml. This is not the new whole-development regression.
- Prior whole-development **44536 is terminal, exit 0**: 2,636 cases.
- Prior preparation acceptance **65642 is terminal, exit 0**, receipt
  34a28795f0a6b1682971e8e0424137224a18b4ffafa80f4a5f3bae38e7d65e8c.

No publication, joint, generation, layout or test call in this archive remains live. Do not poll or
restart them. No formal training/controller/remote action is authorized.

Read-only preservation check e14c99 exits zero after the first cold audit: both
six-file original numerical cores and the real manuscript are unchanged; all six
non-provenance publication outputs match the accepted prior complete generation;
all three exact owned dispatchers remain T. This is not combined replay acceptance.

This first new fixture explicitly reuses the accepted complete joint raw bytes;
the new reader must recompute every raw branch and all seven publication outputs.
Its seven added output alterations test exact publication inspection using only
the same cold run's fresh expected bytes; they are not seven independent full
reconstructions. Strict manuscript-consumer integration remains unimplemented.
