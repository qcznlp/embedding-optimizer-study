# Exact owned calls

All times are UTC on 2026-09-07. An observation timeout is not completion.

| Handle | State | Scope |
| --- | --- | --- |
| 18216 | terminal / exit 1 | 77 cases, 71 new config-adapter failures |
| 5717 | terminal / exit 1 | 77 cases, 2 new BF16 reference failures |
| 89516 | terminal / exit 0 | 77 optimizer cases after exact reference correction |
| 82357 | terminal / exit 0 | 78 cases with exact corrected candidate sources |
| 9140 | terminal / exit 0 | first actual full-model reset/routing check |
| 42926 | terminal / exit 0 | read-only actual scheduler-state shape |
| 18189 | terminal / exit 1 | 4 new extra-field restore counterexamples |
| 63620 | terminal / exit 0 | final 82 optimizer cases in development |
| 19056 | terminal / exit 0 | final 82 optimizer cases with corrected sources |
| 75915 | terminal / exit 0 | current-source full-model retry |
| 42923 | terminal / exit 0 | actual upstream sampler tail counterexample |
| 13666 | terminal / exit 0 | 19 new sampler cases |
| 40320 | terminal / exit 0 | 103 combined optimizer/sampler/InfoNCE cases |
| 90041 | terminal / exit 0 | tail observation after import-order-only cleanup |
| 73858 | terminal / exit 0 | full 3,044-case population, before new sampler tests |
| 64963 | terminal / exit 0 | 103 combined cases with corrected sources; observed 14:18 UTC |
| 21015 | terminal / exit 0 | full regression including new batches; observed 14:18 UTC |

All calls are finished. Do not poll or restart them as missing work. Neither
21015 nor 64963 was a training process.

The 73858 XML reports 3,044 tests, zero failures/errors/skips, 1,065.772 seconds,
and 3,035 serialized testcase elements. The nine-element discrepancy recurs from
earlier milestones. It collected tests before the 21 sampler cases were authored;
its success is not full-suite coverage of the later code. Preserved full-first.xml:
433,601 bytes, SHA-256 5516560ebfc6526d11bbe7efee1eb6068a2db9955b479b1ac1ead7b04f2fa2ca.

21015 completed in 1,062.477 seconds. Its preserved `full-with-batches.xml`
reports **3,065 tests, zero failures/errors/skips, and 3,056 serialized testcase
elements**. The recurring nine-element difference remains explicit. The XML is
436,739 bytes, SHA-256
`9ca88411e1e55b2a967f99d491bfe8ea4aea199c8918d988ff69c86215f0e954`.
The complete 103-case corrected-source XML is `candidate-batches-final.xml`,
16,404 bytes, SHA-256
`31a938ae4303f3d74f7ed5a7079ea2a65e8ba802fab726d7baacf4d89895fe35`.
Both XMLs were observed directly after their owned handles returned exit 0.
Preserve all six source-with-batches files; no combined whole-experiment
acceptance exists.

## Current bounded evidence hashes

- focused-final.xml: 13,264 bytes,
  6149fc3b591142498b0925664288b0b0b8db100727c978d93d70170d5701382b (82 cases).
- candidate-final.xml: 13,264 bytes,
  4315b4c41f4854ca6f57b51fdfdfcba03988b461051f379f44bf6da49cc443d0 (82 cases).
- focused-batches-final.xml: 16,404 bytes,
  2af78105c525a4f83aa7629501fd99e4b169b18630ed51065b94d9afa067bf7d (103 cases).
- full-model-first.json: 29,055 bytes,
  d23ab822079e6aaaf56c837037e6c162ddec6cd6cd80d019ce0e39acd5e6a0df.
- full-model-retry.json: 29,079 bytes,
  3eee249cbf189cac920733b4c9fa505cd68158458c091222acf1f5794ccab1ff.
- batch-tail-first.json: 11,797 bytes,
  0540db14da710f4649268a70e88a64eba314844d0e1a52b09b5f65713fd85596.
- batch-tail-final.json: 11,797 bytes,
  fc3f617cd1434dfb07ddb996e62abed091994ad1e411a561aa0eb9a882a532ac.

All model checks are CPU construction/loading observations, not DenseOn updates.
The tail observations use actual sampler objects with synthetic identity labels,
not a repeat of the full natural-data reconstruction. The old primary/legacy
scientific hold and formal authority gates are unchanged.
