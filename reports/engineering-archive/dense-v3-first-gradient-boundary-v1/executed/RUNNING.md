# First resumed backward boundary: bounded diagnostic

**Terminal:** A session 50164 / 84aa35 exits zero; B session 69733 / 4a2137
exits zero. All eight ranks exit zero after the requested backward boundary,
with zero optimizer updates. Do not poll or restart these terminal sessions.
Both original dual-lease sets are released by the completed owned processes.

Two coordinators were actually launched after CPU preparation completed.
Do not edit `probe.py`, `trace.py`, their tests or authorization while live.

- Scope: `dense-v3-four-rank-first-backward-boundary-v1`.
- Source: `17c76f5519b49f305be4c22353e52262fa25d2e5804ec73e34459f127f311d5a`.
- Authority: `6bbe4aea5db03a43ff8b01f1662673c495893d33b60cbe564528eb82c2868f49`.
- Trace tests: 10 pass, original session 26660 / ccdcac / exit 0.
- CPU preparation: session 76884 / bf1572 / exit 0.
- Pool A: session 50164, PID 1019345/start326378932, AdamW, GPUs 4–7.
- Pool B: session 69733, PID 1019372/start326379050, Muon, GPUs 0–3.

Both coordinators acquired all eight descriptors for their original dual GPU
lease namespaces. Children inherit those same descriptions. Only direct owned
children are supervised, with a 30-minute timeout and no automatic retry.
All prior scientific queues and failed GPU resumes remain terminal, unchanged.

Each worker restores the original genuine step-313 checkpoint with the original
bound Trainer and the already tested scalar-counter device adapter. It compares
device-loaded weights, all optimizer state and scheduler with the actual save.
It independently collates the expected next four rank-local batches and checks
every actual loss input's tensor shapes, dtypes and bytes against them. Public
read-only leaf hooks copy each raw backward contribution; no communication hook
or alternative mathematical kernel is installed.

At the original clipping call, an explicit diagnostic stop trap captures all
post-DDP/pre-clip gradients and stops **before clipping or any optimizer update**.
No training checkpoint, full continuation or scientific completion is produced.
CPU sums of observed leaf contributions are labelled as such, not claimed to be
the internal DDP bucket buffers. Instrumentation can change timing. This is not
an uninterrupted-run trace or a cause verdict about the earlier endpoint mismatch.

Preserve all failures and partial outputs. Poll the existing sessions or inspect
only the exact PIDs recorded in `run/pool-{a,b}/*.started.json`; never enumerate
processes/GPUs, touch the protected helper, restart on a timeout observation,
modify original source, relax an equality check or repeat an entire endpoint run.
