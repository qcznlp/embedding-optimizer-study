# CPU arithmetic environment for exact numerical replay

The original analytical closure compares canonical JSON exactly, including
floating-point SVD diagnostics. The primary functional result tables are computed
with the original exact-arithmetic predictions; the diagnostic spectrum and rank
cutoff still depend on the selected BLAS kernel. Package versions alone therefore
do not specify this byte-replay environment.

Hosted run [34768399864](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34768399864)
used an AMD EPYC 7763 with AVX2, no AVX-512, and NumPy's OpenBLAS Haswell backend.
The unchanged functional tables matched exactly, but 136 diagnostic values differed
from the original SkylakeX record by at most 4.440892098500626e-15. The complete
hosted decisions and differences exactly match a controlled local Haswell run.
The genuine local SkylakeX run matches both tables and diagnostics without changes.
This diagnoses the failed comparison; it does not accept the failed full replay
or assert that its later phases ran remotely.

## Compatible instruction execution

CI uses unmodified [Intel Software Development Emulator (SDE)](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
10.13.1 (2026-07-28) to execute the original code under Skylake-server instructions,
including forced emulation on hosts that already implement them. OpenBLAS selects
SkylakeX **inside that process**, never as a native AVX-512 override on an
AVX2-only host. The [runtime variable](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)
selects the numerical kernel; it does not alter scientific inputs or assertions.

The official Linux download is:

```text
https://downloadmirror.intel.com/924984/sde-external-10.13.1-2026-07-28-lin.tar.xz
SHA-256 94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50
```

CI verifies this digest before extraction. The accompanying Intel licenses remain
with the unmodified kit. The binary is downloaded from Intel, not redistributed in
this repository, wheel, Hugging Face artifacts or CI receipts. This runtime is
external licensed software, not part of the project's Apache-2.0 source.

The launcher prefix is `sde64 -skx -force_emulate skx --`. It instruments only the
new CPU replay process and follows its newly spawned children. There is no attach
to existing processes, disabled child following, security-setting modification,
permission bypass or GPU use. If the platform refuses execution, the gate fails;
do not disable SELinux, Yama or other host protections to make it pass.

CI first recomputes the original functional diagnostic and requires exact tables,
exact decisions and zero differences. It then invokes the same complete Make
release target through the launcher, followed by every source-version test and
the unchanged distribution/style gates. The small diagnostic cannot replace the
complete numerical-to-paper execution. All failed outputs remain retained.

This is an engineering reproduction condition, not an optimizer finding. It does
not change source locks, reference values, tolerances, training or the manuscript.
Actual independent success is reported only from terminal hosted receipts, not
from the existence of this workflow or a local emulator check.

## Preserve subprocess isolation

Run34769598250 actually passed the forced-emulation functional comparison on the
hosted CPU, with exact tables, exact decisions and zero differences. Its full Make
replay nevertheless failed before the first numerical child began: Intel's child
launcher reported `PreparePindForFollowExecve` / `ENOENT`. The same failure is
reproduced locally with a minimal Python subprocess and default `close_fds=True`.
Keeping descriptors open avoids that failure, but is **not** the delivered remedy:
the original replay's isolation behavior remains unchanged.

CPython's [original subprocess implementation](https://github.com/python/cpython/blob/v3.12.3/Modules/_posixsubprocess.c)
already has both bulk and per-descriptor closing paths. The isolated builder
`scripts/build_ci_python.py` compiles original CPython3.12.3 with the configure
cache setting `ac_cv_func_close_range=no`, selecting the existing per-descriptor
path. There are no edits to CPython, the study's subprocess code or its comparisons.
The builder authenticates the official complete source archive:

```text
https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tar.xz
SHA-256 56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1
```

The interpreter is installed only into the new output directory. Its path-only
dependency file reuses the already hash-verified scientific package directory;
it contains no executable Python and changes no packages in that environment.
The original Python license remains with the source. This is a narrowly scoped
historical CPU replay runtime, not a recommended general-purpose Python service.

CI runs `scripts/check_cpu_replay_fds.py` under SDE before the scientific checks.
Two real subprocess cases require unwanted inheritable descriptors to be closed
and only an explicitly selected descriptor to survive. The original `close_fds`
and fresh-process behavior are preserved. Both cases have passed locally; the
complete numerical-to-paper execution is a separate mandatory check, not implied
by these smaller controls. No OS security setting, existing process or GPU is touched.

The current-paper parent receives an explicit checkout `src`/repository
`PYTHONPATH`. Merely adding the external package directory does not execute that
directory's editable-install `.pth` hooks. The original numerical child still
clears `PYTHONPATH` and loads its authenticated source closure; these two source
roles are not merged. All three numerical thread limits are explicit as well.
