# Hash-bound runtime binary for CPU CI

The scientific runtime specification and both original formal locks are unchanged.
`requirements-ci-flash-wheel.txt` supplies a genuine precompiled FlashAttention
2.7.4.post1 wheel to avoid compiling all 85 units on each small hosted runner.
The wheel was built from the unchanged official source, not a stub or skipped build.

| Binding | Value |
| --- | --- |
| Official PyPI source SHA-256 | `f03485c9a49a4d68d0733acdcad80ab0e72afa025a777fdc2966ceccf9d51765` |
| Wheel SHA-256 | `9feca56918f1603358f32ce0c7ab06d221e6123b8b1072566880479f3b1aa1ba` |
| Wheel size | 192,793,979 bytes |
| Immutable HF dataset revision | `18c1ab7abc636d2542be6831591f31c171c7dcf6` |
| Bundle manifest SHA-256 | `c097029fa6c92cdcb690eba43c2c5ae93cec52dd70caeafd834ae2aaa4a97a2d` |

The [36-file distribution](https://huggingface.co/datasets/qcz/embedding-optimizer-study-analysis-artifacts/tree/18c1ab7abc636d2542be6831591f31c171c7dcf6/runtime/flash-attention-2.7.4.post1-cu129-cp312-v1/c097029fa6c92cdcb690eba43c2c5ae93cec52dd70caeafd834ae2aaa4a97a2d)
contains original source and licenses, exact compiler command/log, source hashes,
original runtime locks and verification receipts. Build platform: CPython 3.12,
Ubuntu 24.04 x86_64, glibc 2.39, Torch 2.9.1+cu129, CXX11 ABI true, CUDA compiler
12.9.86, architectures 80 and 90; 32 CPU jobs. Actual build exit was zero,
elapsed 1,705.57 seconds. No GPU execution was requested.

Local verification checked all 77 wheel members, RECORD hashes, 58 Python files
against official source, a real ELF extension and five callable native exports
in a fresh process, and the original native runtime specification. A corrupted
control was refused and is not distributed. The study system runtime remained
unchanged. No computational source was patched.

These receipts establish producer-local binary integrity and CPU importability,
not second-host success, GPU numerical equality or training-resume compatibility.
Independent CI must still execute every source-role test, original distribution
audit, complete numerical reconstruction and reviewed paper build. None is skipped
or allowed to fail. The immutable URL and SHA are enforced by `--require-hashes`.

Other platforms or ABIs should build from `requirements-formal-flash.txt` using
the original compiler route in CONTRIBUTING.md. This wheel is not a blanket
new-host GPU launcher. Engineering artifacts do not change any scientific result
or add implementation discussion to the manuscript.

## Primary source runtime

Hosted run 34765504737 successfully installed the genuine binary, passed the
native runtime specification and distribution checks, and executed all 3,800
tests: 3,799 passed, one failed, none skipped. Both historical source roles passed
completely (733 analysis / 194 factorial). The sole current-role failure was the
unchanged twelve-primary-source equality assertion: the historical formal lock
installs `huggingface-hub==1.29.0`, but all twelve actual source receipts bind
`1.28.0`. The live study environment is also 1.28.0. The native runtime spec does
not cover this package; passing it was therefore not sufficient for source identity.

`requirements-primary-replay.txt` is an explicit, hash-locked replay overlay for
the genuine PyPI 1.28.0 artifacts. Apply it after the original formal base install.
It reconstructs the recorded primary environment without rewriting either original
lock, any source receipt, scientific code, checkpoint or test assertion. CI also
runs the unchanged actual-source tests immediately after installation to fail fast.
Full hosted acceptance remains pending until the replacement run completes.
