# Exact CPU backend diagnosis and instruction-emulation check

This is engineering evidence only. Scientific source, the original closure and
all exact comparisons remain unchanged.

Hosted CI34768399864 (job103753403488, ccdbcf2713596722b77f3c32cb0d3d97b90feab3)
failed the original functional-decisions comparison. Its passive diagnostic
completed successfully without accepting that failure. The retained ZIP matches
GitHub artifact10321395722's SHA-256 and 14,432-byte size. It records AMD EPYC7763,
AVX2 without AVX-512, and NumPy OpenBLAS Haswell. The whole actual decision object
and all 136 differences exactly match the separately preserved local Haswell
control. Functional result tables match exactly; the largest diagnostic difference
is 4.440892098500626e-15. The preceding sealed diagnostic archive was correctly
only a hypothesis; this successor records its confirmation.

The unmodified official Intel SDE10.13.1 Linux download was SHA-256-verified
against 94e97d623fec54385686e1e7ba65ebc9941748c05ee451423948334892bf2b50.
Its accompanying licenses were read and retained with the local kit. No binary
is redistributed here. No system package or OS security setting was changed;
no existing process was attached and no GPU process was inspected.

An owned Python smoke test exited zero (54643/fba808). The first `-skx` diagnostic
artifact shows exact tables/decisions, but its tool handle was lost: its OS exit
is unobserved and is not asserted to be zero. Asking the documented
`-force_emulate list` option printed the supported chips and then an Intel internal
assertion, exit127/b28b2d; its original pin-log is retained. This was not a host
permission refusal and no host protection was relaxed.

The separate `-skx -force_emulate skx` functional diagnostic then actually exited
zero (59283/e3d770), with exact tables, exact decisions and zero differences.
This is local emulated-instruction validation, not a remote full-replay pass.
The original 37 focused source-role/distribution tests also pass (2edcaa/0).

The workflow now prepares the official hash-bound kit, requires the strict
functional probe, and invokes the same complete Make release through its CPU
launcher. Every source-role test and original numerical/document gate remains
mandatory; full output retention includes emulator diagnostics but not the kit.
Complete local emulated and independent hosted execution are pending at this
archive's creation. Their later terminal receipts must not be inferred here.

See docs/cpu-numerical-replay.md for the platform condition and safe failure rule.
