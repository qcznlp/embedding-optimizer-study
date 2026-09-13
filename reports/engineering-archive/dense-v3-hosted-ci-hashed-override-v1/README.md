# Formal hashed installer override — hosted verification pending

Actual hosted repair run [34758294285](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34758294285)
failed before tests: fast-plaid requests Torch 2.9.0, while the immutable formal
environment records Torch 2.9.1+cu129. The installer omitted the original formal
version override. CUDA compiler installation succeeded. No test verdict follows.

Using the same original requirements-formal.lock as both hashed requirements and
hashed overrides resolves successfully in an empty isolated Python 3.12 environment
(12a81c / exit 0, dry run only). The first dry run targeting the system interpreter
was refused as externally managed; no bypass or installation was performed. An
isolated unhashed-constraints attempt was then rejected by require-hashes. Both
failures are preserved. No lock, native guard, numerical source or result changed.

Current focused JUnit: 37 cases, zero failures/errors/skips. Ruff and diff-check
pass. The hosted log is retained unchanged, including its terminal failure.
This commit repairs CI and contributor commands only; full hosted execution is
still pending. No GPU or live study environment changes.
