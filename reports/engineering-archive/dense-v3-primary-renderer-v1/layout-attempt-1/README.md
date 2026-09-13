# Preserved first sample-generation failure

At 2026-09-07 08:00:50 UTC, the first `render_demo.py` invocation exited 1.
It requested nonexistent `paper/acl.sty` instead of the actual vendored
`paper/vendor/acl.sty` used by the paper Makefile. `file_identity` raised
`ValueError: Require an ordinary source/payload file` before producing any
sample. Its output directory `/tmp/dense-v3-primary-render-demo.X8dSgw`
remains empty. The exact initial generator and candidate renderer are preserved
here. No numerical or manuscript source changed; the retry changes only the
new sample generator's style-file address and uses another new directory.
