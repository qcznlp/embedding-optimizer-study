# Final delivery status

This closeout supersedes the older CI-pending and HF-approval-pending snapshots
in the handoff documents. Those snapshots retain historical provenance; they are
not instructions to restart work.

## Delivered scope

The owner's final scope is **DenseOn only, paper only**: no new LateOn or blog.
The delivered study compares AdamW, Muon and NorMuon from the same pretrained
checkpoint using the same deterministic 500K queries, seven fixed distinct hard
negatives, no in-batch/cross-device negatives, 8192 context and four rates per
optimizer. All five primary training stages are retained and evaluated.

| Requirement | Completed evidence |
| --- | --- |
| Primary matrix | 12 complete runs, 60 checkpoints, 840 checkpoint/task BEIR results, baseline and validation selection; [results and evidence map](PROJECT_STATUS.md#completed-work) |
| Weight and functional analyses | All 60 trained states and the 61-state functional panel including the pretrained reference; [findings](PROJECT_STATUS.md#findings-supported-by-the-completed-evidence) |
| Crossed continuation | 12 complete runs, 60 checkpoints/probes and 168 final full-corpus task results; [restoration](docs/continuation-outcomes-restoration.md) |
| W&B tracking | All 24 finished runs and 5,172 ordered history rows checked against native records; [tracking audit](reports/engineering-archive/dense-v3-final-tracking-audit-v1/README.md) |
| Model/analysis durability | All 120 scientific checkpoints backed up; required analysis groups anonymously downloaded and verified; [primary](docs/checkpoint-restoration.md) and [continuation](docs/continuation-probe-restoration.md) restoration |
| Paper | Complete reviewed manuscript, 158-word abstract, eight-page main and 13 total pages; [PDF](reports/paper-review/dense-v3-complete-manuscript-revision-v1/actual/paper/build/main.pdf), [source/build](paper/current/README.md) |
| Reproduction and source | Public scientific source at `b09b334973e69d6990e2811d5de5c1d0f471f170`; complete original numerical-to-reviewed-paper replay, package/distribution checks and all 3,800 tests pass |
| Historical HF cleanup | Owner-approved current-tree deletion completed and independently verified; [receipts and precise scope](reports/hf-obsolete-cleanup/completed-v3/README.md) |

No scientific, recovery or CI verification job remains live in this closeout.
Do not repeat completed experiments, run old controllers or touch `gpu.py` or its
processes. A different scientific question or additional replication would be
new work, not an unfinished part of this declared matrix.

## Final verified scientific-source CI

[Run 34772464866](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34772464866)
completed successfully on `b09b334973e69d6990e2811d5de5c1d0f471f170` at
2026-09-13 18:18:54 UTC. The downloaded terminal receipts verify:

- all **3,800 cases / 211 modules**, with zero failures, errors or skipped tests;
- all **15,543 source input files** byte-identical to that published commit;
- the genuine original-source runtime build and subprocess-isolation controls;
- exact functional results, the full original numerical graph and the reviewed PDF.

The copied [terminal verification receipt](reports/hf-obsolete-cleanup/completed-v3/hosted-ci-verification.json)
identifies the exact run, commit and GitHub artifact digest. This is CPU
reproduction, not a new GPU-training run. The final cleanup/status publication
changes documentation and receipts only; it does not replace the tested source,
change any result or claim another full CI execution.

## HF withdrawal completed

The owner explicitly confirmed the previously reviewed 9,629-file current-tree
scope. The original digest-bound script executed one atomic commit per repository:

- [Model commit 7cdfd2be](https://huggingface.co/qcz/embedding-optimizer-study-checkpoints/commit/7cdfd2be93fa0eb3272eb1cece7582962def1e65): **3,790 files removed**.
- [Analysis commit 6e2750e6](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/commit/6e2750e62bcba0249bb31810cdb7512e0b98aa8b): **5,839 files removed**.

All **13,821 other payload files** retain their original size and Git/LFS content
identity. Both cards were replaced with current v3 restoration guidance and the
withdrawal notice. All **65 shared LFS objects** were independently downloaded
anonymously and SHA-256 verified. Both original parent revisions remain readable.

This is not permanent historical erasure: **Git history, shared objects, valid
files and local engineering records were preserved**. The old approval-pending
paragraphs below the newer notices in archived handoffs no longer describe the
current cleanup state. Never execute the same deletion plan again or re-upload
withdrawn paths from broad old snapshots.

## Scientific interpretation is unchanged

Completion is not proof of universal optimizer superiority or of a causal
weight-space mechanism. The all-rate contrasts remain inconclusive; the
validation-selected NorMuon advantage over AdamW is +0.437 nDCG@10 points, while
selected Muon's +0.317 interval crosses zero. No tested descriptor passes all
four exploratory recipe comparators, and native helpful participation is not
rotation-stable. The crossed continuation concerns two fixed source states and
three order seeds, not independent primary-seed replication or mediation.

All original failed attempts, unobserved process exits, source-role boundaries
and scientific limitations remain preserved. Engineering-error narratives stay
outside the manuscript, including its appendix.
