# Functional-dimension integration: prospective preparation

Parent: exact-feature bridge acceptance `f43a9049a4cc2022ef6a6262c18fa80f4f2088d0ee465da8a3d708bbd848a077`.
The old scientific dimension protocol and every existing result remain unchanged. This work is
CPU-only preparation; it cannot release training, admit historical states as primary, export on a
GPU, change controllers, publish a paper or upload artifacts.

Before adopting the existing coordinate-deletion kernel, check two fixed binary-exact examples:
an exactly one-coordinate vector (deletion leaves an undefined cosine), and a dominant-coordinate
vector with nonzero `2**-30` residual coordinates (subtraction may erase their norm/dot product).
Use explicit deletion and scalar compensated dot/norm calculations as independent references.
Do not pick examples, thresholds or seeds using optimizer outcomes.

If those examples confirm a numerical defect, prepare a separately named literal-deletion kernel;
do not modify or silently rehash the old implementation. Cosine uses only the remaining coordinates,
with FP64 normalization, strict-greater-than positive ranks, one positive at index zero and seven
fixed negatives. A zero residual norm must fail the complete computation rather than be clamped or
silently excluded. Keep original attribution signs, task-first averaging, all shared masks/seeds,
all three endpoint rotations, original spectrum definitions and the explicit shortlist boundary.
The replacement is a numerical implementation amendment, not an optimizer mechanism finding.

Verify small algebraic controls, non-degenerate random controls, all coordinates at 768 dimensions,
and one already retained historical pretrained export as an engineering-only real-vector input.
Record independent rank equality and numeric error separately; no output-dependent tolerance.
Use `rtol=1e-9, atol=1e-12` for floating numeric comparisons, as in the existing reconstruction
protocol; require exact rank/nDCG equality. Preserve failures and never relax these checks.

Then connect fixed-probe row identities and the complete 61-state v3 export/feature interfaces.
No legacy run-name parser, available-state subset, legacy parent receipt or filename rename may
establish a primary identity. All twelve complete run admissions must precede outputs and GPU work.
The exact named-feature bridge will retain the four true functional predictor names. Publication,
portable reconstruction and source/runtime release must remain explicit separate gates.
