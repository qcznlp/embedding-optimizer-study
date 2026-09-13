# Stable current-paper source and packaged execution

2026-09-13. The reviewed complete-result manuscript is now in
[`paper/current`](../../../paper/current/README.md), with a real package entry
and `make -C paper current` target. This is document/source integration, not
the complete scientific source release or retirement of historical gates.

```bash
make -C paper current PYTHON=/usr/bin/python \
  CURRENT_OUTPUT=/absolute/path/to/new-paper-build
```

The Python entry is `python -m embed_optim.current_paper --paper-dir paper/current
--output /absolute/path/to/new-paper-build`, with `CUDA_VISIBLE_DEVICES=''` and
the checkout's `PYTHONPATH=src`. All twelve inputs must match the reviewed
snapshot. Existing outputs, symlinks, noncanonical output paths, modified
results and source substitution are refused; failed attempts are preserved.
The original complete-document checker was copied **byte-for-byte**, not
reimplemented with weaker checks. The legacy default `all` and `release`
targets and their original audit remain unchanged.

## Actual evidence

| Check | Observed result |
| --- | --- |
| Initial new-entry plus legacy-layout controls | 65 tests pass; 69537 / 65b595, exit 0 |
| Actual Make target | 4778 / 50428f, exit 0; new complete PDF, 158-word abstract, main page 8, 13 pages, 25 embedded non-Type-3 fonts |
| First real wheel/sdist | 12056 / 5593d7, exit 0; original audit f06da2 remains exit 1 with ten producer-path findings |
| Expanded integration checks before guide fix | 49464 / 583863, exit 1; 85 pass and one missing newly created continuation-outcomes restoration guide |
| Final integration checks after adding the omitted guide | 30859 / 66d481, exit 0; all 86 tests, no failures/errors/skips |
| Final real build and extracted-wheel execution | 81331 / 59d2ab, exit 0; build exit 0, original full audit exit 1, actual packaged child exit 0 |
| Final focused Ruff/format/diff checks | 77d86f, exit 0; only the new authored wrapper/tests formatted |

The final wheel contains **426 members**, with **213 declared data files**
across the distribution. It includes the complete current paper, vendor style,
manifest, package entry and the previously omitted outcome-restoration guide.
It was built from the sdist. The actual child runs from a separate directory
with only the extracted wheel on `PYTHONPATH`: **all 28 loaded study modules**
come from that wheel. The rebuilt PDF's complete extracted text is identical
to the already visually reviewed manuscript. No repository source fallback,
new model/data download or scientific recomputation occurs. This is same-host
relocation, not a fresh-environment or physical-second-host test.

Final wheel SHA-256:
`f37f33b7df40a5b261d92aa7148c24c1429db549cb9d0d7502c9fee154105c84`.
Final sdist SHA-256:
`cb12b0be77af3bdd3df5e2def2a95d7b1796bd2d412e0e6fc8a2b911f7c5a72f`.
Reviewed document snapshot SHA-256:
`6ef28bcf1f2cd873bcb0a663c2baea57ec6dc98a4e441d0850608a61f46ae6b9`.

The original full distribution audit still fails the same **ten producer-path
findings**; they are recorded, not waived. The first missing-guide test and
initial new-file formatting failure are retained. Passing document execution
does not establish training recovery or a complete source release. No numerical
source, scientific result, old manuscript, frozen protocol, kernel or original
release gate changed, and no external write was made. Both exact-resume failures
remain unresolved. All sessions in this report are terminal.

Next address the remaining actual source/distribution transition: separate
historical producer namespaces from portable runtime roles without concealing
them or dropping negative controls, and expose the complete scientific consumer
through stable source. Do not repeat these completed builds/tests as missing work.
