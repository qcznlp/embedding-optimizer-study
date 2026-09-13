# Full hosted tests pass; numerical child failure needs its internal logs

Run 34766616352 / commit 562eef8f34db15a58afe3a12580bbb58b6116a9e is terminal
failure (watcher 39291 / fa90ae / exit 1). All 3800 source-role cases passed with
zero failures/errors/skips: current2873, original-analysis733, original-factorial194.
Early actual twelve-source identity, formal runtime, genuine extension, package
build, original distribution audit and portable evidence all passed too.

The subsequent actual full numerical-to-paper command failed after about 22
seconds. Its wrapper reported `Original numerical replay failed; outputs retained,
no retry`. The workflow retained only final complete.json/PDF, not internal failed
output/logs, so this evidence does not identify the inner cause. Do not infer a
numerical, document, dependency or source defect from the outer exception alone.

Collector 49188 / ec7e69 exited 1 as designed because the hosted job failed. Before
that refusal it downloaded the exact job metadata, log and original test artifact,
and verified the ZIP against GitHub's digest:
c5a1f61d641c4c60ebbd6c6683267ca5654b4c6ef95f4598aa2ad924c4a4c13f.
No complete paper receipt was created or claimed.

The next workflow retains the entire bounded complete-paper output on success or
failure and executes the unchanged full numerical gate before the expensive test
matrix. Every original test, source identity, numerical assertion/tolerance and
paper gate remains mandatory. This is diagnostics and fail-fast ordering, not a
scientific fix or accepted reproduction. Focused workflow/distribution tests pass
(5e3565 / exit 0); the inner failure remains pending actual evidence.
