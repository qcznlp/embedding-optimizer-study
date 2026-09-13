# Original CPython subprocess path and complete local CPU replay

Engineering evidence only: no training, evaluation, scientific source, reference
value, numerical tolerance or manuscript result is changed. This archive is
sealed only after the actual owned execution handles are observed terminal and
the collector verifies the original complete receipts. The archive is not a
claim that the replacement hosted workflow has passed.

## Preceding failure, retained rather than reclassified

Hosted run34769598250 / job103756645249 at b0fa533f passed the actual forced-SDE
functional diagnostic with exact tables, exact decisions and zero differences.
Its full Make command failed before its first numerical child began, reporting
`PreparePindForFollowExecve` / `ENOENT`; the child exited255. The preceding ZIP
is retained verbatim and matches GitHub artifact10322110477, 15,742bytes,
SHA256 `9ed168a34b37b463f7dc45006c90136639c9c62ea7ce5b5ae87d52edb6161842`.
The raw hosted log and failed local full replay are retained as well.

Owned local subprocess controls reproduce this error with default FD closure,
including the explicit follow-subprocess and no-vfork variants. Pure exec
replacement succeeds, as does a diagnostic that keeps FDs open. The latter is
**not the remedy**: original `close_fds=True` isolation remains required. Failed
unsupported debug-option attempts and the accepted debug invocation are retained.
Actual tool exits are recorded in `observed-terminal-exits.json`; outputs alone
are never used to invent an unobserved exit.

## Original implementation, configured without source patches

CPython3.12.3 already implements both bulk and individual descriptor closure.
The builder selects its existing individual-descriptor path with
`ac_cv_func_close_range=no`; neither CPython nor study subprocess code is patched.
Official complete source archive:

```text
https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tar.xz
SHA256 56bfef1fdfc1221ce6720e43a661e3eb41785dd914ce99698d8c7896af4bdaa1
```

The manual preparation also checks all4490 file hashes in the official SPDX
document. The original Python license, source receipt, build commands and logs
are retained. The isolated build lives only in a new temporary prefix and reuses
already verified scientific packages through a path-only dependency file. It
does not install or upgrade the study/system environment. Optional interactive
and DBM modules are not claimed as a complete default Python distribution.

The unsuccessful preliminary SQLite import, intermediate build logs and partial
header download failure are preserved. Required development headers were merely
extracted under the temporary build directory, not installed system-wide. The
final native package check uses genuine Torch, NumPy, SciPy and FlashAttention.
This is a narrowly scoped historical replay runtime, not a general service image.

## Actual checks and interpretation

The manually configured interpreter passes full forced-emulation reconstruction
of the original primary/factorial numerical graph and reviewed manuscript:
session21593 / f5cf6c / exit0. Original assertions and source identities remain
unchanged. Eight shared manuscript inputs agree exactly; the reviewed document
snapshot is `6ef28bcf1f2cd873bcb0a663c2baea57ec6dc98a4e441d0850608a61f46ae6b9`.

The reusable repository builder actually constructs a separate interpreter.
Its real FD checks require all unwanted inheritable descriptors to be closed and
only an explicit `pass_fds` descriptor to survive. Its original functional probe
also passes exactly. Full repository-built-interpreter reproduction and all3800
source-role regressions are accepted only through their terminal records and
original native completion objects; see `verified.json` and the two complete
replay receipt groups. The collector rehashes all17 primary publication outputs,
all eight shared inputs and both actual reviewed PDFs before sealing this report.
PDF binary digests may differ between builds; no byte-identical final PDF claim
is made. Original document source/text/layout checks remain mandatory.

The current document parent has an explicit checkout src/repository PYTHONPATH:
adding an external package directory does not execute its editable-install hooks.
The original numerical child still clears PYTHONPATH and executes its own exact
source closure. This is not a merger of historical and current scientific roles.
All numerical thread limits are explicit and GPUs are hidden.

The official unmodified SDE kit and its licenses remain outside the archive;
no SDE binary is redistributed. No existing process was attached, no OS security
setting was changed and no protected GPU helper or process was touched. A local
pass is not a physical second-host run or final GitHub acceptance. Earlier failed
archives remain unchanged. The owner's no-PR/no-intermediate-push rule applies.

See [CPU replay conditions](../../../docs/cpu-numerical-replay.md) and the
[preceding CPU evidence](../dense-v3-ci-cpu-emulation-v1/README.md).
