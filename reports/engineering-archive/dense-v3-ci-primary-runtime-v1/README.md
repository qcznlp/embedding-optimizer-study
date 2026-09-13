# Actual primary Hub runtime reconstruction

Hosted run 34765504737 / commit 2651e1e44dad067a6ff05b8c4a13f902fdb5dacf
completed with failure; watcher 14170 exited 1 (18e8a1). Genuine runtime/binary
installation took 38 seconds and passed, as did the original distribution audit.
Actual complete tests: current 2873 with one failure, original analysis 733 passed,
original factorial 194 passed; no errors or skips. Total 3799/3800 passed. Numerical
and paper reconstruction were skipped after the failed test and are not accepted.

The retained original test assertion found only a package-identity mismatch:
the historical requirements-formal.lock supplies huggingface-hub 1.29.0, whereas
all twelve immutable actual primary source records and the study system use 1.28.0.
The narrower native runtime spec does not list Hub. Full source identity caught
what that spec could not; neither check was weakened or changed.

The original artifact ZIP was downloaded via the normal GitHub endpoint with exit
0 (56522 / 69afd1), and its SHA matches GitHub's digest:
f01ef820e5a450ee59e4169634ccb7ae643027788888e4a2cd7c8dbdf371224c.
The complete original job log is preserved too (98659 / ee070c / exit 0).

Actual isolated-venv control 81059 / 4b79c6 / exit 0 installed genuine hashed Hub
1.29.0, reproduced the unchanged test's exact failure (child exit 1), then installed
genuine hashed 1.28.0 and passed all three actual primary-source tests and the
unchanged runtime spec (child exits 0). No system package installation occurred.
The new requirements-primary-replay.txt makes this actual recorded runtime explicit;
the historical formal lock, all source receipts and scientific outputs are unchanged.
CI now checks all twelve source identities immediately after installation as well
as retaining the complete original 3800-case matrix and numerical/paper gates.

The 37 relevant workflow/distribution tests pass (fffbb2 / exit 0); style/diff
checks pass. Replacement hosted acceptance is pending, not implied by local checks.
