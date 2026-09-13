# Paired backward diagnostic — live source is frozen

**Terminal failure:** A session 17406 / 663091 exits 1; B session 39489 /
33677f exits 1. Actual rank exits are A `[1,1,-15,1]`, B `[1,-15,1,1]`.
The second pass was refused before backward by the diagnostic's incorrect
assumption that `model_accepts_loss_kwargs` must be false. The original Trainer
uses accumulation division when `num_items_in_batch is None` even with that
flag true. No original kernel or training recipe was changed. Keep all first
pass traces and failed sources; do not restart these entries or relabel them.

The previous first-backward diagnostic is terminal; it is not restarted here.
This new pair fixes the genuine step-313 weights and next four rank-local batches,
captures the first native backward pass, then restores its first-input Python,
NumPy, CPU Torch and current-rank CUDA RNG states for a second pass through the
same DDP object. Weight/optimizer/scheduler identities are rechecked afterward.

- Source SHA: `28d706e46c7a6ae549532fbcce7a951553e38a06d3e731d6eaa96149c7425774`.
- Authority SHA: `38d6add0e12786e722312fd40628ff94fe3c0408ed243c9bada6521da3473ef6`.
- Helper controls: 14 pass, session 73922 / 2e57d9 / exit 0.
- Preparation: session 95598 / 1cf94c / exit 0.
- A (AdamW, GPUs 4–7): session 17406; PID 1021800/start326464527.
- B (Muon, GPUs 0–3): session 39489; PID 1021825/start326464645.

Both original lease namespaces were acquired and inherited by the exact four
direct rank children per pool. The supervisor has a 30-minute bound and never
automatically retries. W&B is disabled. No protected helper or historical
controller is touched. Do not inspect arbitrary processes or enumerate GPUs.

Each pass has four actual microbatch backward calls; there are **zero optimizer
updates**, no clipping, no new training checkpoints and no scientific completion.
The second pass is explicitly driven through the unchanged inherited training_step
and original three no_sync/one sync schedule; it is not falsely labelled an
uninterrupted Trainer-loop iteration. It reuses process/model caches as well as
DDP state. Therefore only equal observed local leaf contributions and differing
post-DDP gradients would locate a tested difference downstream of differentiation.
That observation alone would not prove the old endpoint mismatch repaired.

Do not edit bound source/helpers/tests/authorization while these sessions run.
Use these original session handles or only the exact identities in the new
`run/pool-{a,b}/*.started.json`. Preserve failures and all prior evidence.
