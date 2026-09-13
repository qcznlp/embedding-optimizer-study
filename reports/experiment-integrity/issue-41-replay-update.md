## Scope

Run the corrective DenseOn-only replication after the historical packed-path non-invariance finding. LateOn is historical provenance only.

## Locked and completed setup

- [x] Add explicit, auditable `dense_can_flatten_inputs=false` execution mode
- [x] Run the frozen worst-case memory preflight and select common micro-batch 8
- [x] Commit the final 12-run execution contract before formal training
- [x] Freeze corrected validation and BEIR reload paths before corrected outputs existed
- [x] Freeze corrected weight-space and retrieval-bridge analysis before checkpoint 782 ([PR #44](https://github.com/qcznlp/embedding-optimizer-study/pull/44))
- [x] Freeze full-grid outcome inference before checkpoint 782 ([PR #45](https://github.com/qcznlp/embedding-optimizer-study/pull/45))
- [x] Add fail-closed, remotely verified Hugging Face checkpoint backup
- [x] Add durable human/machine agent handoff contracts

## Active execution

- [ ] Complete 12/12 formal training runs with five checkpoints each (8/12 complete; NorMuon 1e-4/3e-4 active; final two rates pending at 2026-09-04 22:18 UTC)
- [ ] Produce all 60 resumable checkpoints (44/60 observed and remotely covered; the eight complete runs account for 40 deeply audited stages)
- [ ] Upload and remotely verify every completed corrected run (8/8 currently complete runs uploaded; independent digest audit passes for all 808 files, 67,658,898,978 bytes)
- [ ] Evaluate all 60 corrected checkpoints on 14 decontaminated BEIR tasks (840 task units)
- [ ] Run the frozen primary/secondary statistics and corrected weight-space analysis
- [ ] Complete the paper-only publication, Hugging Face artifacts, and release audits (the separate blog is retired)

## Integrity audit and isolated follow-up (2026-09-04 22:18 UTC)

- All 500,000 materialized training-row identities agree with the sampling ledger; zero within-row repeated negative IDs or positive/negative ID overlaps.
- Eight completed primary runs / 40 checkpoints pass deep payload validation; 89 frozen source/parent/configuration bindings match. This audit did not numerically resume every checkpoint.
- A separate header check finds 134 FP32 model tensors in each of the 40 completed primary checkpoints, consistent with the declared FP32 parameters and BF16 autocast.
- Both active NorMuon runs have sealed checkpoint 1563: 1e-4 at HF commit `6c1f11fd22ae9f2199d5bfc9138adfecd77eb5ca`, 3e-4 at `7945be0bfaac13ef8bec88c6ca61e78a8e908f7b`. The two new stages total 34 files / 2,702,861,935 bytes; missing/extra/size/digest mismatches are all empty. They are durable partial runs, not scientific completion.
- The isolated `narrative/weight-space-spine` worktree passes **965 tests**, zero failures/errors/skips/exclusions. A subsequent documentation regression check passes all 51 tests.
- The read-only **47-step** plan inserts eleven primary-publication/dimension steps before factorial calibration and preserves the original 36 commands. The live main/factorial controllers remain on contracts `4152531e...` / `6605090d...`, with zero post-training/factorial steps executed.
- Portable publication recomputes statistics and exact LaTeX. The new independent raw-feature replay recomputes coordinate ablations, random masks, spectra and rotations from a trusted downloaded archive in a new output directory, never using the producer path. Tests include a renamed/network-disabled full synthetic 61-state panel without PyTorch or dataset imports.
- A separate full recomputation of all 61 real historical 768D states matches every original CSV cell and attribution array exactly. The frozen rotation control now checks both scores and ranks; all 39 historical endpoint/pretrained rotation pairs preserve ranks. These are implementation/equivalence checks, not primary scientific findings.
- The source-bound archive implementation uses an immutable content-addressed HF prefix and includes all 61 raw vector arrays plus the publication evidence. **No actual primary dimension export, analysis, upload or replay has executed.**
- **Not yet complete:** remaining primary training; all 840 primary BEIR units; primary dimension/bridge findings; the crossed intervention; remote primary analysis archival; final paper release. Historical results cannot satisfy these gates.
- **Deployment boundary:** implementation, narrative and audit receipts remain in the isolated local worktree, not committed/merged/pushed. Approval for a clearly labeled WIP branch was requested because the existing pre-commit rule requires no pending manuscript results; no approval has yet been received. This does not authorize a live merge or relaxed final-paper gate.
- Finish the exact main controller contract before the controlled still-zero-step factorial transition. Do not alter live source bindings, launch duplicate controllers, or touch `gpu.py`.

## Start here

1. [`README.md`](https://github.com/qcznlp/embedding-optimizer-study#60-second-handoff) — 60-second public entry point
2. [`AGENTS.md`](https://github.com/qcznlp/embedding-optimizer-study/blob/main/AGENTS.md) — non-negotiable operating and read-order rules
3. [`PROJECT_STATUS.md`](https://github.com/qcznlp/embedding-optimizer-study/blob/main/PROJECT_STATUS.md) — canonical human-readable status and next actions
4. [`CURRENT_PROGRESS.json`](https://github.com/qcznlp/embedding-optimizer-study/blob/main/CURRENT_PROGRESS.json) — latest durable machine-readable snapshot
5. [`docs/dense-no-packing-retrain.md`](https://github.com/qcznlp/embedding-optimizer-study/blob/main/docs/dense-no-packing-retrain.md) — exact run, resume, evaluation, analysis, and backup commands

Historical outputs must never be overwritten. Frozen protocols and failed gates are evidence, not tunable knobs. `gpu.py` and its processes are out of scope and must never be inspected or touched.
