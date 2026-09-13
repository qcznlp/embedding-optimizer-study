# Corrected paired backward diagnostic

**Terminal:** A session 32954 / c06ef0 exits zero; B session 85232 / 6ee966
exits zero; all eight ranks exit zero. No optimizer updates were performed.
All local gradient contributions match across passes; 123 of 134 post-DDP
gradient tensors differ in both cases. Both passes agree across ranks separately.
The completed pair is not a repaired long endpoint resume. Do not restart it.

This is a new separately bound successor, not a relabelled retry of the first
failed pair. The original failure and all its cold-pass traces are unchanged.
Only the diagnostic precondition is corrected: the actual original
`get_batch_samples` must return four batches and `num_items_in_batch=None`, with
no custom compute_loss_func. The original installed `Trainer.training_step`
normalizes by four under either model_accepts_loss_kwargs flag when count is None.
Actual native-function CPU tests cover both flag values and pass.

- Source SHA: `52d44883225e27ee1b0db9cd862ff949b14ec060f58f291d13dad1f8320cc34a`.
- Authority SHA: `b43f2ba1a49e0e53b5049d2f27234fde818cecbe90ffc73e1950cd36aa9bab52`.
- Controls: 17 pass, session 11434 / 0de2c0 / exit 0.
- Preparation: session 30104 / f2d912 / exit 0.
- A: session 32954; PID 1023384/start326499728; GPUs 4–7.
- B: session 85232; PID 1023415/start326499847; GPUs 0–3.

Both pools acquired their original dual leases and inherited eight descriptors
to each of four direct rank children. No helper/process enumeration, W&B write,
old-controller transition or automatic retry is allowed. Each supervisor has a
30-minute limit and may reap only its exact direct children.

The paired passes retain weights, input tokens, first-input RNG and optimizer/
scheduler state, with eight backward microbatches and **zero optimizer updates**.
The second pass explicitly calls the unchanged native training_step with the
same no_sync schedule; it is not an uninterrupted Trainer-loop trace. It reuses
model caches and runtime state as well as DDP. Both local leaf contributions and
actual post-DDP gradients must be compared before assigning a location to any
difference. No old endpoint comparison is waived, no new tolerance is defined,
and no scientific checkpoint or completion is produced.

Do not edit bound sources/tests/authorization while live. Poll these sessions
or only their recorded exact process identities. Keep all partial/failing output.
