# Joint raw-to-inference reconstruction — both cold audits complete

Updated 2026-09-07 08:21 UTC. This is explicitly synthetic engineering work,
not a primary experiment, optimizer finding or paper publication.

The new `primary_v3_joint_reconstruction` entrypoint runs the existing complete
raw-outcome, native-shape geometry and full-768D vector readers before recomputing
the original nine-feature bridge and the four-feature functional inference.
It preserves the original evidence/plan schemas, source paths as provenance only,
all statistical rules and exact rendered LaTeX. See [plan.md](plan.md).

## Current evidence

- The first complete fixture calculated all 61 vector states, then failed because
  its new fixture builder passed serialized learning-rate strings directly into
  inference. The actual frozen authoring path decodes checked CSVs first. Every
  row in the four decoded tables matches its declared state; only the new fixture
  builder was corrected. The numerical kernels and old accepted parents are unchanged.
- A separate mapping test found that the new reader/fixture used Python constant
  order instead of the actual frozen geometry protocol's file order. Both now
  follow the actual authoring order; no stored source order is silently rearranged.
- The first 12-case unit attempt has one failure and is preserved in
  [attempt-1/](attempt-1/). The current **25-case focused suite passes**. It also
  tests typed file sizes, disjoint provenance roots, CSV decoding and refusal-audit
  failures that must not be counted as successful semantic checks.
- The same first fixture's original bridge independently passes all 540 exact
  predictions, 36 fold errors and nine pooled decisions using full SymPy systems.
- The corrected complete fixture has now been assembled. Its independent joint
  identities/feature joins, original and functional exact systems, nine task
  intervals, rotation contrasts and 68 figure points have returned successfully.
  Complete first cold audit **18914** is now terminal/pass, including all 477
  numerical outputs and all 21 semantic controls, using only 75 archived package
  imports. Independent complete replay **74613** also exits zero: the entire cold
  result and independent oracles match, and all 21 controls pass again. The two
  original result records are preserved as complete-audit.json and complete-replay.json.
- Full regression 47895 is terminal/pass: **2,573 cases**, zero failures/errors/skips.
  This does not clear the separate training candidate's eleven source-contract failures.

The completed synthetic raw components were copied byte-for-byte into a new
externally anchored 1,774-file transport. The new fixture reassembles the correct
bridge/inference evidence from that single population; it does not merge the old
standalone vector/outcome/geometry populations. Reuse is explicit in its metadata.
The cold public entrypoint still recomputes every raw branch from the archived inputs.

Follow [RUNNING.md](RUNNING.md) for exact terminal handles and result hashes.
No owned audit or test is live; do not poll or restart completed calls. The
validate.py checker binds both completed audits, their payloads and preservation
evidence. Only its generated validation.json supplies bounded acceptance; the
development suite alone does not. Late-table negative controls reuse the same cold
run's freshly verified raw outputs at the affected consumer; an early refusal is
not counted as a completed raw replay.

Use [commands.md](commands.md) for bounded reproduction. The remaining old/v3
publication and manuscript-audit interface gaps are documented in
[next-publication-boundary.md](next-publication-boundary.md). That note is not a
released implementation, an executed plan or permission to bypass pending gates.

Formal training remains 0/12 accepted full-horizon v3 runs. No model encoding,
corpus retrieval, fresh weight spectrum/health measurement, physical cross-host
test, manuscript installation, deployment or remote publication occurs here.
The paper, six numerical cores, old checkpoints and stopped controller chain remain
protected. Never touch gpu.py or its processes.
