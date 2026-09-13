# Passive original functional diagnostics — no acceptance override

CI34767862594 failed in the same complete numerical replay (watch33375/4d14ca/1).
The full retained failure output now shows the original require_same failure at
primary/replay.py:151, comparing functional decisions. All earlier outcomes,
original/exact weight tables and the functional result table passed the unchanged
exact comparisons. No failed numerical output is classified as accepted.

Original GitHub artifact download36525/0d9f9f/0 matches GitHub's SHA:
c5c8f37b7c0feee0477b89fd2cc6df1784b2c8f795e541536b964f2fff775730.
The raw complete job log download31356/35026c/0 is retained. The error alone did
not identify the actual differing values, so a read-only diagnostic reconstructs
the original functional calculation from authenticated original inputs/source.
It exports actual decisions/differences and runtime details, never changes a
reference or turns a failed original comparison into a passing result.

In a genuine isolated environment reconstructed from the current CI locks, the
whole original numerical-to-reviewed-paper replay passes locally (80075/6d5c9b/0).
The local diagnostic with default SkylakeX OpenBLAS also matches every result and
decision exactly (67005/f72ec2/0). A controlled local Haswell OpenBLAS invocation
(46570/03fdb7/0) leaves the functional tables exactly unchanged but produces 136
floating-point diagnostic differences, including singular values at roughly 1e-15.
This is a backend hypothesis for the hosted discrepancy, not yet identification
of the hosted CPU/runtime or a scientific difference. Both complete diagnostic
outputs are retained. No native source, tolerance, expected record or algorithm
was changed. OpenBLAS documents the explicit kernel selection at
https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/.

The next CI runs this passive diagnostic only after the full numerical gate fails,
retaining CPU/runtime and actual decision records with the original failed output.
The diagnostic's own exit zero means observation completed, not replay acceptance.
All scientific and full CI gates retain their original behavior.
