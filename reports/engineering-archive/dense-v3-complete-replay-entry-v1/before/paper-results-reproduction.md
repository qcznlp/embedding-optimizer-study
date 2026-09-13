# Reproduce the current primary paper results

The complete **primary numerical result graph** has been reconstructed from a
closed source/data directory, with CUDA hidden and no fallback to the original
producer directories. This includes the original and exact publication outputs,
all exploratory comparator controls, and the two empirical sixty-state figures.
It is not yet a reproduction of the complete paper: the continuation BEIR
outcomes, their factorial inference and the final combined PDF are still pending.

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

The final combined-paper reproduction will additionally require the complete
continuation inference, strict fresh compilation, PDF review and source-release
checks. Do not present this primary-only completion as final paper publication.

## Complete-paper handoff

A separate [combined-paper replay consumer](../reports/engineering-archive/dense-v3-combined-paper-replay-handoff-v1/README.md)
is now queued behind the existing actual complete-document author. It will copy
the genuine full factorial inputs and this unchanged primary bundle, rerun both
numerical branches, regenerate all four result includes and empirical figures,
and freshly compile the full paper under the same strict checks. It will compare
the extracted PDF text and source/layout evidence with the actual native paper;
compilation metadata need not make the PDF binary identical.

At the **2026-09-12 22:11 UTC** observation, the original complete document and
this combined replay were both still waiting. The final combined bundle does
not yet exist. This handoff removes a manual scheduling delay; it is not a claim
of complete-paper portability, final publication or physical second-host proof.
