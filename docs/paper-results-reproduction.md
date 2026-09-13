# Reproduce the complete numerical results and paper

The complete **primary numerical result graph** has been reconstructed from a
closed source/data directory, with CUDA hidden and no fallback to the original
producer directories. This includes the original and exact publication outputs,
all exploratory comparator controls, and the two empirical sixty-state figures.
The subsequent **complete primary + continuation numerical/PDF replay also
passed on 2026-09-13**. It reconstructs all six continuation tables and the
original complete-result paper. The later reviewed prose/reference revision is
reproduced separately by [paper/current](../paper/current/README.md), using the
same generated findings. Neither replay is a completed public source release.

For the full combined command, go to [Complete-paper replay](#complete-paper-replay).
The primary-only entry below remains available for the narrower numerical graph.

## What to copy

Copy the entire `closed-primary/` directory from
[`dense-v3-closed-primary-paper-replay-v1`](../reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1/README.md)
to the destination machine. Keep its contents unchanged; put notes and outputs
outside that directory. It contains 129 source/data files, 112,631,759 bytes,
plus its input manifest. No checkpoint weights are needed for this numerical replay.

This bundle is currently a **local unpublished artifact**, not something a clean
GitHub clone is guaranteed to contain. Copy it explicitly from this workspace.
The separate [checkpoint](checkpoint-restoration.md),
[weight](weight-analysis-restoration.md),
[functional](functional-analysis-restoration.md) and
[continuation-probe](continuation-probe-restoration.md) guides cover the immutable
public data/model backups needed for deeper upstream analyses.

The externally recorded SHA-256 of `closed-primary/manifest.json` is:

```text
88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d
```

The manifest lists every allowed file and its byte count and SHA-256. The replay
rejects changed, missing or extra inputs, symlinks and nonlocal roles, and verifies
the input inventory again after computation. Do not regenerate the manifest to
make a modified bundle pass.

## Run

From this repository root, in a compatible separate CPU environment:

```bash
replay_bundle="$PWD/reports/engineering-archive/dense-v3-closed-primary-paper-replay-v1/closed-primary"
replay_parent="$(mktemp -d)"
CUDA_VISIBLE_DEVICES='' PYTHONPATH='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/usr/bin/python -B "$replay_bundle/run_isolated.py" \
  --manifest-sha256 88861e52218f56b860851489106cd30e35dd79bb038eb9082967e3376db6457d \
  --output "$replay_parent/replay"
```

For an explicitly copied bundle, change only `replay_bundle` to its absolute
ordinary directory. Use the Python executable of the destination environment
if it is not `/usr/bin/python`. The output must be a new absolute path; failed
attempts and preexisting outputs are preserved. No network, Hugging Face token,
W&B key or GPU is needed. Do not install or upgrade packages in the running
experiment environment.

The successful host used Python 3.12.3, NumPy 2.5.2, Torch 2.9.1+cu129 and
Matplotlib 3.11.1, with one numerical thread. The full read-only installed-package
snapshot is in the archive's `runtime-after-replay.json`; it is not a minimal
dependency specification or a claim that every listed distribution was imported.
Byte-exact figures can depend on numerical/font/runtime versions. A discrepancy
on another environment must be investigated, not hidden with a new tolerance.

## Outputs and evidence

The attempt produces `reconstructed/` and `io-boundary.json`. Successful exit zero
and `reconstructed/complete.json` establish the following bounded reconstruction:

| Output | Recomputed scope |
| --- | --- |
| Primary outcomes | All 840 task scores, validation-only reselection, ten outcome tables and original paired-task inference |
| Weight predictors | All nine original and five exact predictors, all sixty states and original held-dose folds |
| Functional inference | Nine table families, nine primary and 27 rotation contrasts, 240 predictions and all figure points |
| Exploratory controls | All 84 weight and 24 functional comparisons; 5,040 and 1,440 predictions |
| Publication | All seven original and ten exact output files, byte-for-byte equal to the genuine native assembly |
| Empirical maps | Both sixty-state PDF/PNG figures and the complete point data, all byte-exact |

The original functions are copied unchanged. Saved recipe metadata supplies only
the identifiers and declared settings needed by pure table functions; it is not
a replacement admission validator. Complete native model/run/analysis admission
is authenticated upstream evidence, not rerun by this offline command. This
replay does not recompute SVDs, re-encode text, repeat coordinate deletions, replay
training, or freshly reconstruct raw validation-query scores.

The actual accepted completion SHA is
`cc980ed44ed7a3681a35d8d40f84d8d14bf0c42a17996fae75c0e060a773e41a`.
Its timestamp is intentionally specific to the original attempt, so a later
completion receipt itself need not have the same digest. The numerical output
digests inside the receipt are the comparison targets.

The wrapper refuses Python-level network connections and file opens under the
old producer roots, except the copied bundle and new attempt directory. Its
`closed_bundle_files_opened` field records attempted open events, including
unsuccessful bytecode-cache lookups; it is not a list of only successful reads.
This is an observed fresh-process I/O boundary, **not** an OS sandbox or evidence
that execution has already succeeded on a physical second host.

The primary-only completion does not include continuation inference. The
separate complete replay below includes that inference and strict compilation;
neither command is a final source-publication verdict.

## Complete-paper replay

The completed closed bundle is now preserved at
`reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed` in this
local checkout. Copy that **entire directory**, unchanged, to the destination.
It contains 189 bound input files / 117,532,494 bytes plus its original manifest.
Its manifest SHA-256 is
`746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7`.
The original entry source SHA-256 is
`1a9c50dba2f58cc68b110749ccbc40d427445d53f0dc6c994b8f78c02096d566`.

From the repository root, using the recorded numerical runtime and TeX Live:

```bash
replay_bundle="$PWD/reports/engineering-archive/dense-v3-complete-replay-entry-v1/closed"
replay_parent="$(mktemp -d)"
(
  cd "$replay_bundle/primary" || exit 1
  CUDA_VISIBLE_DEVICES='' PYTHONPATH='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/python -B "$replay_bundle/replay_complete.py" replay \
    --bundle "$replay_bundle" \
    --manifest-sha256 746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7 \
    --output "$replay_parent/replay"
)
```

For a copied bundle, change `replay_bundle` to its absolute ordinary directory.
The **component working directory is required**: preserve the `cd` above.
The initial outer-directory attempt failed its I/O boundary; the source-identical
component-directory execution passed. Both are preserved in the
[actual closeout](../reports/engineering-archive/dense-v3-final-evaluation-closeout-v1/README.md).
No allowlist, original source, input hash or comparison tolerance was changed.

The command reruns the primary graph, the original six continuation tables,
independent rational effects and 100,000-draw bootstrap check, regenerates all
four result includes and empirical maps, and freshly compiles the complete
native document. It checks source/layout/fonts and full extracted PDF text;
PDF metadata need not be binary-identical. `replay/complete.json` and both
`replay/io-boundary.json` and `replay/primary/io-boundary.json` retain results.
The accepted original completion SHA is
`6a0a8a3657d88a20529db8d45d32c31c5b74be783c0abb9d57994b0009d3c275`;
new timestamps and paths mean a later completion receipt need not share its hash.

This reproduces the original complete-result manuscript, not the subsequent
158-word-abstract prose revision. After numerical reconstruction, build the
reviewed version using `make -C paper current CURRENT_OUTPUT=/absolute/new/path`.
The four generated includes, results file and three figures are byte-identical
between the combined bundle and the reviewed snapshot; prose and references
have their separate authenticated document snapshot.

The [local entry evidence](../reports/engineering-archive/dense-v3-complete-replay-entry-v1/README.md)
checks the copied inventory and unchanged sources without rerunning completed
science. The bundle remains unpublished local WIP, not a guaranteed remote
GitHub asset or an installed-wheel payload. No model weights or raw text
encoding are needed for numerical replay. Python I/O hooks and TeX input
recording are not an OS sandbox, a physical second-host test, a GPU-resume
proof or a source-release verdict.
