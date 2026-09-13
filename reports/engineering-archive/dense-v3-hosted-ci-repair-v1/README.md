# Hosted CI repair — verification pending

The first public-source hosted run [34756580699](https://github.com/qcznlp/embedding-optimizer-study/actions/runs/34756580699)
failed. Its unmodified artifact ZIP is retained: 2,873 current cases had five
failures and 34 errors; 733 original-analysis cases had one failure; 194 original-
factorial cases had 77 failures. No test cases were skipped; the later paper step
did not run. Local 3,800-case acceptance did not establish hosted CI acceptance.

The hosted developer environment differed from the frozen scientific runtime
and lacked FlashAttention. CI now installs the genuine hash-locked runtime and
compiles its real CUDA extension. It neither fabricates package metadata nor
changes native guards. Subsequent commands use no-sync to preserve formal pins.
Two synthetic fixtures now explicitly relocate an authenticated runtime spec
and create an uncommitted repository, instead of depending on the producer host.
Production source, immutable runtime locks, scientific outputs, source-role
overlays, numerical assertions and tolerances are unchanged.

Local corrected original-factorial execution: all 194 cases pass, coordinator
27932 / b71966 / exit 0. The focused current XML records 85 passing cases; its
original terminal tool result was lost, so its OS exit is not asserted. Full
hosted verification of this repair is pending. No new training or GPU work.

Before/after edited files, exact failed hosted artifact, local logs/XML/source
inventories and actual distribution audit are retained. This report does not
supersede old failures or label an unexecuted CI revision successful.
