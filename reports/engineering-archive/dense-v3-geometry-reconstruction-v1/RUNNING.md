# Exact execution record

Observed 2026-09-07 06:03 UTC. Every numerical/test handle below is terminal.
Do not poll or restart these completed calls. Final documentation and acceptance are separate.

| Handle | Work | Last verified state |
| --- | --- | --- |
| 15605 | Full native-shape same-process smoke | terminal, exit 0 |
| 41271 | Independent set/rational oracle | terminal, exit 0 |
| 3807 | Ten zero-statistic counterexamples on initial reader | terminal, exit 1; ten expected failures preserved |
| 93840 | First 63-case focused run | terminal, exit 1; one test-helper failure preserved |
| 58258 | Nine real diagnostic records after consistency check | terminal, exit 0 |
| 74266 | Preserve failed fixture and current source snapshots | terminal, exit 0 |
| 99933 | Full regression, 2,548 collected cases | terminal, exit 0; no failures/errors/skips |
| 42912 | 67-case focused retry | terminal, exit 0; no failures/errors/skips |
| 13684 | Native-shape cold audit and 28 rehashed controls | terminal, exit 0; all 28 refusals pass |
| 20415 | Independent relocated replay, same full scope | terminal, exit 0; all eight outputs and all 28 controls match/pass |
| 7251 | First cold acceptance precheck | terminal, exit 0 |
| 57422 | Prior bindings/core/paper/ledger preservation precheck | terminal, exit 0 |
| 41318 | Final public-doc/strict-paper/source-stack checks | terminal, exit 0; four cases pass |

Source trees and runtime authority remain as stated in the root AGENTS.md. All calls are
CPU-only with the pinned `/usr/bin/python` and absolute development PYTHONPATH. Nothing
here changes the stopped old dispatcher chain or touches the protected helper.

- First smoke: `/tmp/dense-v3-geometry-reader-smoke.V1KzY2`.
- Tests and XML: `/tmp/dense-v3-geometry-reconstruction-tests.aGzFAJ`.
- Failed first focused fixture preserved outside pytest cleanup:
  `/tmp/dense-v3-geometry-failed-tests.myLpUE/focused-fixture/fixture/archive`, external test anchor
  `7785d0bdba2fe406066facad11ba1a9100e004c41cb721ab1ee8d92bbb3ec9db`.
- Real-record retry: `/tmp/dense-v3-geometry-real-records-retry.GDZfZl`.
- Cold audit: `/tmp/dense-v3-geometry-reconstruction-cold.6zffw0`.
- Independent replay: `/tmp/dense-v3-geometry-reconstruction-replay.LHk0mo`.

The first cold `result.json` is 3,181,986 bytes, SHA-256
`ea35d5406d135ea0b2ce117dad1004ca51cb4782312a0a70fd58794241cf2730`.
Its `fixture.json` is 2,521 bytes, SHA-256
`df5eb5d5510dbb38f41f8fa54fb99325fd613b2b0dc30943cfe21ba18e1bc6f3`.
Exact copies are `complete-audit.json` and `complete-fixture.json` in this archive.
The independent replay `result.json` is 3,060,111 bytes, SHA-256
`f13f6567772f956045de896c93a3d301d14b3847475966b09eee28c13b5f95f3`;
its exact archive copy is `complete-replay.json`. The shared external payload manifest
anchor is `26eba51063170e376c033ec3c7be359dfb0392efa39ff149563d656c6e5eb88e`.
Both cold processes import only 72 archived package modules. The first audit binds 11,788
retained artifacts and all 28 semantic refusals; no network/original-path fallback occurred.

Original logs/source/inputs are preserved; these are evidence locations, never destinations
for another producer. An independent replay requires a new empty work directory and the
externally supplied SHA-256 of the completed first cold audit receipt.
