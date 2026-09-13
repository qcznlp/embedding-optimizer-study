# Exact resumed-endpoint comparison: failure retained

2026-09-13. Both restoration-only attempts have ended. They are **not live**
and must not be polled or automatically restarted.

| Check | Actual result |
| --- | --- |
| AdamW recovery, session 39663 | Coordinator exit 1, terminal 8ceca3 |
| Muon recovery, session 75985 | Coordinator exit 1, terminal 1672c5 |
| All eight training ranks | Exit 0; each reaches 391 from 313, 78 additional updates |
| Fresh native comparison readers | Both exit 1; first exact mismatch is `model/embeddings.norm.weight` |
| Read-only complete difference analysis, session 39062 | Exit 0, terminal b7f32d; diagnostic only, not acceptance |

Original native checkpoint inventories were authenticated before deserializing
any diagnostic state. The difference analysis retains every leaf, not just
the first mismatch:

| Component | AdamW unequal / total leaves | Muon unequal / total leaves |
| --- | ---: | ---: |
| Model tensors | 134 / 134 | 134 / 134 |
| Named optimizer state and groups | 268 / 1,336 | 180 / 1,163 |
| Scheduler | 0 / 10 | 0 / 10 |
| Each rank's RNG state | 0 / 637 | 0 / 637 |
| Selected Trainer counters | 0 / 5 | 0 / 5 |

Maximum absolute model differences are 0.0004420429468154907 (AdamW) and
0.0023330599069595337 (Muon). These are descriptions of the failed comparison,
**not proposed tolerances**. Optimizer differences are in moment tensors;
counter/group/scheduler equality does not establish exact training equivalence.
The CPU-to-CUDA scalar-counter repair resolved the former execution failure,
but it did not establish equality with uninterrupted training.

The cause of the remaining numerical divergence is **unresolved**. Matching
RNG and schedule does not exclude data-order, loading, arithmetic-order or
backend differences. The first resumed loss log averages seven new updates,
where the uninterrupted log spans ten; that log alone is not a matched-batch
loss comparison. Later matched logging windows also differ. Deterministic
attention backward is not a proof of globally bitwise-deterministic distributed
training. No particular backend is blamed without a discriminating check.

All original science runs, checkpoints, retrieval results and failed attempts
remain unchanged. The restored checkpoints are diagnostic outputs only, not
replacements for scientific states. No tolerance, kernel, original acceptance
guard, process protection or source binding was changed. Nothing was uploaded.

Next narrow the first divergence using actual loaded-weight/moment readback and
matched sample/gradient boundaries before scheduling another endpoint attempt.
Do not rerun the full scientific matrix or reuse these failed output paths.
Paper revision and source-distribution work can proceed independently.
