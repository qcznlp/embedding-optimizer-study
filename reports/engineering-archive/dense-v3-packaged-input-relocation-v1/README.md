# Genuine input relocation with the unchanged wheel

Engineering evidence, 2026-09-13. This is not a scientific result or source release.

## Actual outcome

The original packaged `factorial_v3_inputs.load_inputs` completed against relocated
genuine inputs. Parent CPU session **44065**, terminal **3e9334**, returned actual
exit zero. Its isolated child ran from **01:59:50 to 01:59:55 UTC**, returned exit
zero, and the parent's final content rechecks completed at **02:00:05 UTC**.
The child was PID 905662; its original receipt records no start tick, so none is
inferred. Do not rerun this completed reader as missing work.

The inventory contains **101 bound files / 5,628,099,441 logical bytes**: one
primary-source input binding, six already-packaged repository assets, nine input
evidence files, 23 data-store files and 62 experiment files. The 95 non-repository
files were copied to new explicit role directories; the six repository assets
had to exist byte-identically in the wheel. They were not supplied from the
producer checkout to repair the wheel. Two additional packaged configurations,
`representation_probe.json` and `formal_runtime.json`, are checked directly by
the original reader and are outside that 101-file inventory. The complete wheel
has its own preserved content identity.

Inputs include the real 500k primary data, fixed 50k continuation branch, probe
data and both genuine source checkpoint-2345 states. Each source is authenticated
by its original 20-file native manifest, run identity, content seal and retained
durability receipts. The reader confirms the 50k branch and 32 calibration rows.
It hashes model/optimizer files; it does **not** deserialize their tensors or
repeat forward/backward, calibration, training or GPU-resume verification.

The unchanged wheel is the already built artifact from
[the distribution coverage archive](../dense-v3-distribution-recovery-surface-v1/README.md):

- Wheel: `51a65e250a75b5e54f459c12cd930b7bf8cd87dc2c0928889091928be4e37f6e`.
- Native input module: `372db4abf1e7136f118a42916e07146d20dc09d5c5de868d3c09f0951aa237f7`.
- Native checkpoint module: `e49f55d5d01408de001b8e1d56244fbcaf189de45c0209caffa0b4e34f828e19`.
- New diagnostic driver: `1d8eb006e8658d113cd6792cc390e152f3a571be1809d0ac2fa485a3e331c027`.
- Inventory: `d7f26f841cb23bf38db66267e794e5c180fcd6823e907a16e2334d6baac4a82c`.
- Native readback: `da57593de89f5405ce7c1d31f16c3d203f5285dbe619652abca705ee13f8d227`.

Original working directory: `/tmp/dense-v3-portable-input-read.OHnjIOc4`.
Only the driver, metadata, receipts, log and scoped source/ops snapshots are
archived here. The 5.63 GB data/model payload and wheel are **not** duplicated into
the repository or uploaded. Existing immutable artifact locations remain in the
original input inventory and native readback. This archive is not itself a
self-contained model/data restoration bundle.

## Isolation and acceptance boundary

The child used `/usr/bin/python -B -I`, a fresh working directory, hidden CUDA,
empty project `PYTHONPATH`, one-thread CPU settings and closed inherited file
descriptors. All three loaded project modules came from the extracted wheel.
Its original audit digests, checkpoint checks and `Locations` implementation
were retained; no admission was mocked or source binding replaced. The native
`scientific_admission=False` and `execution_authorized=False` scope is retained.

Python audit hooks reject old producer-root opens/directory enumerations,
networking, child spawning and input mutations. Two separate diagnostic controls
correctly refuse an original configuration read and a new payload write before
opening either file. The native call then records **zero denied operations**.
The log's `opened_local_files` records Python open events, including possible
unsuccessful import-cache probes, not a syscall-level list of successful reads.
The parent independently rechecks all 101 original and relocated bindings and
the original wheel after the child exits.

This is **not** an OS sandbox or syscall trace, a fresh dependency installation,
a physical second host, a complete application replay, GPU endpoint equivalence
or a complete distribution audit. No active scientific job, original source,
scanner, input evidence or manuscript was changed.

## What the ten original distribution findings mean

The original complete audit is still **failed**, with ten findings in seven
unique source paths. This table classifies inspected content; it does not waive
any finding or declare that every package entry point is portable.

| Unique source path | Artifact findings | Inspected role |
| --- | ---: | --- |
| `src/embed_optim/factorial_v3_inputs.py` | 2 | Recorded historical identifiers mapped lexically to explicit new roots; the genuine native relocation above succeeds. |
| `AGENTS.md` | 2 | Operational/historical producer-location provenance in the packaged snapshot. |
| `PROJECT_STATUS.md` | 2 | Dated historical producer-location evidence in the packaged snapshot. |
| `tests/test_dense_evaluation_pause.py` | 1 | Exact historical worker-command examples used to test pause guards. |
| `tests/test_factorial_v3_calibration.py` | 1 | Negative controls reject unbound, escaping and symlinked input roles. |
| `tests/test_factorial_v3_run_contract.py` | 1 | Missing-calibration refusal fixture, not a genuine training launch. |
| `tests/test_primary_v3_outcome_reconstruction.py` | 1 | Cold-reader negative-control guard explicitly forbids the old producer root. |

The unchanged scanner examines every regular payload, including documentation,
tests and JSON. Moving strings into JSON, encoding/splitting them, removing
negative controls or dropping evidence to evade the scanner is not a repair.
A separately reviewed runtime-versus-provenance packaging design is still
needed before source release. No such remediation is implemented here. The
original failed audit is copied unchanged, not rerun or replaced by this reader.

## Concurrent scientific work and next action

At **02:09:29 UTC**, the exact original evaluation observer reports **18/168**
tasks complete, eight FEVER workers live/R and no coordinator failure. At
**02:10:02 UTC**, all six downstream CPU waiters remain live/S with no actual
summary, complete paper/replay, outcome upload or GPU-resume result yet.
These are dated observations, not a heartbeat or throughput estimate.

Continue the existing queues. Collect their original complete statistical,
document, reconstruction, durability and resume outcomes before final paper
integration/review. Preserve the pending one-time GitHub access question and
all existing external-write boundaries. This turn is **PROGRESS** in a genuine
native input relocation, not additional training, new optimizer evidence or
overall completion. Do not add this engineering narrative to the manuscript.
