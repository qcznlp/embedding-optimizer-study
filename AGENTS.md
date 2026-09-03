# Agent handoff instructions

Start with `PROJECT_STATUS.md`, then read `README.md`,
`configs/dense_scope_amendment.json`, and the protocol relevant to your task. Keep
`PROJECT_STATUS.md` current whenever a meaningful run, failure, release gate, backup, or scientific
interpretation changes.

The active paper scope is DenseOn only. LateOn files are historical provenance and must not be
promoted into primary inference or used to justify new computation.

Never inspect, read, edit, signal, stop, replace, or otherwise touch `gpu.py` or its processes. It
is outside this repository and automatically yields to study jobs.

Preserve all existing checkpoints and evidence. Use new output namespaces for corrected reruns;
never overwrite the 34 completed historical Dense runs. Treat protocol thresholds and failed gates
as results, not knobs to relax. In particular, the candidate-breadth width-7 reproduction failure
and `reports/candidate-breadth/packing_invariance.json` must remain disclosed.

Before committing, run the tests and release/audit commands appropriate to the changed surface,
check `git diff --check`, and verify that the manuscript has no pending result macros or Type 3
fonts. Do not push generated evidence or change GitHub pull-request state until its source-bound
audits pass. Never print or commit credentials.

