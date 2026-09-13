# DenseOn primary v3 training snapshot — not yet execution-released

This is the standalone training-source assembly for the ongoing AdamW/Muon/NorMuon
DenseOn NAACL study. The complete study and paper remain in
`/root/embedding-optimizer-story-refactor`; this directory is not a replacement
deliverable or a claim that the paper is finished.

All 56 selected source/config/parent/test files match `source-assembly.json`.
The initial byte-identical assembly passes 104 training tests, its actual v3
contract and pinned runtime load, and the complete real-input/twelve-recipe CPU
preflight. The exact numerical core has prior real four-rank diagnostic evidence.
These checks are not full training runs or optimizer-quality findings.

The owner has requested immediate primary training after correctness confirmation.
The current protocol remains its immutable DRAFT copy. A separately reviewed local
training-code commit/execution release is required before launching. Do not change
old failed source locks, use a status-only bypass, or import old checkpoints as v3.

The fixed experiment retains 500,000 groups, one positive/seven negatives, no
in-batch negatives, context 8192, batch 128 on four ranks, and all 12 rates/cells.
The new data/output root is `/root/embedding-optimizer-v3-experiment`. The first
planned pair is `verified-v3-adamw-3e-5` and `verified-v3-muon-3e-4`; all remaining
recipes must still run. Checkpoint steps: 782, 1563, 2345, 3126, 3907.

Use `/usr/bin/python3` and this source root explicitly on PYTHONPATH. Do not use the
live repository's differently versioned venv. Do not touch `gpu.py`, enumerate
unrelated processes, resume/kill old controllers, publish WIP, retry rejected HF
deletions or retry GitHub writes. Follow the canonical agent handoff for authority.
