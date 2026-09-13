# GPU checkpoint-replay update localization

Completed 2026-09-06 07:10 UTC. **Engineering evidence only; no scientific or production release.**
This extends, and does not replace, the failed/control measurements in
[`dense-gpu-correctness-v1`](../dense-gpu-correctness-v1/README.md).

## What the experiment establishes

For AdamW 3e-5, Muon 3e-4, and NorMuon 3e-4, all four ranks exactly reproduce the first
post-resume training update using the actual production optimizer and scheduler from the same
entry state and the captured clipped gradients. This means **24 exact next-step reproductions**
(three recipes × four ranks × uninterrupted/resumed gradients), plus **12 exact same-gradient
repeat controls**. Full weights, optimizer state/groups and scheduler state match, not merely a
norm or selected matrix. Rank copies are correctness checks, not independent experiments.

The two paths have tiny differences in their first replayed gradients. Replaying those two
gradients from identical state exactly reproduces the subsequent weight difference. Thus the
tested difference is accounted for by the observed gradient-to-update mapping, rather than an
unobserved checkpoint-state change or irreproducible optimizer call on identical inputs.
This does **not** identify why the first gradients differ.

| First replayed update, all 134 parameter tensors | AdamW | Muon | NorMuon |
| --- | ---: | ---: | ---: |
| Maximum clipped-gradient difference | 1.4901161e-8 | 1.4901161e-8 | 1.4901161e-8 |
| Maximum immediate weight difference | 2.9802322e-8 | 3.4198165e-6 | 2.3059547e-6 |
| Direct update/state/scheduler reproductions | 8/8 | 8/8 | 8/8 |
| Identical-gradient repeat controls | 4/4 | 4/4 | 4/4 |

These are **immediate single-update differences**, not the previous archive's end-of-epoch
measurements. Do not interchange their numbers or pool the two captures as independent trials.

All 88 hidden matrices are traced for each Muon-family recipe on rank zero. Each of the 176
paired traces calls both paths: **352 operator-output/state comparisons** must match the actual
production helpers bitwise. The trace is a validated observation of the implemented arithmetic,
not an independent mathematical correctness oracle.

The Muon and NorMuon first-update traces share the same input pair and hidden momentum state;
they are not independent replications. Across their 110,297,088 hidden-matrix elements:

| Stage | Matrices containing differences | Unequal elements |
| --- | ---: | ---: |
| FP32 Nesterov combination | 88 | 1,089,035 |
| BF16 conversion | 26 | 30 |
| BF16 norm denominator | 0 | 0 |
| Newton–Schulz iteration 1 | 17 | 172 |
| Newton–Schulz iteration 2 | 17 | 2,825 |
| Newton–Schulz iteration 3 | 17 | 107,884 |
| Newton–Schulz iteration 4 | 17 | 1,137,027 |
| Newton–Schulz iteration 5 / Muon operator output | 17 | 4,795,454 |
| NorMuon final row-rescaled/renormalized output | 17 | 21,776,734 |

Hidden-matrix Nesterov differences are at most 2.3283064e-10 before conversion. At iteration 5,
the maximum absolute update difference is 0.00732421875 and the maximum per-matrix relative L2
difference is 0.00815744. The original hidden-matrix norm denominators remain identical.
The NorMuon second-moment difference is at most 5.4431439e-8; its final operator difference is at
most 0.00769179687. None of these different scales is a retrieval-quality or stability verdict.

The narrower conclusion is that a few BF16 rounding differences spread through this actual
Newton–Schulz calculation. This rules in a specific numerical propagation path on this fixture;
it does not establish a universal optimizer property, incorrect optimizer mathematics, an
accuracy benefit, harmlessness, or the cause of any historical primary result.

## Execution boundary

The wrapper calls the unchanged `audit_dense_gpu_resume.py` campaign: authenticated full DenseOn
weights, three fresh diagnostic optimizers, four NCCL ranks, model-local BF16 autocast, FA2 with
the original nondeterministic-backward flags, TF32, zero dropout, non-reentrant checkpointing,
eight loader workers per rank, and 288 deterministic short synthetic rows in groups 128/128/32.
Checkpoint resumes at steps 1 and 2 run in the same process group. No historical optimizer state
is continued and no independent process/host restart or long-horizon run is tested.

The new wrapper observes the exact named-parameter/gradient order and production parameter-group
routing. It captures the real LambdaLR functions, restores the exact entry optimizer/scheduler,
and performs the same optimizer-then-scheduler order. It does not alter a Trainer loss, gradient,
optimizer formula, library file, primary checkpoint, tolerance, or production source.

The parent campaign still reports **0/24 strict full-replay rank checks passed** and exits 1.
That failure is deliberately preserved even though the narrower single-update controls pass.
All six continuations still restore entry state, row order, local CUDA/CPU/Python/NumPy RNG,
scheduler and final progress exactly. Worker-process RNG is not separately measured.

All diagnostic launchers have exited. BEIR remains paused in the exact retained three-process
chain from the previous handoff. No new primary training/evaluation, deployment, ledger migration,
source push, or HF deletion occurred. The 12 primary runs / 60 preserved checkpoints remain
execution/durability counts, not scientifically accepted evidence. Nothing here enters the paper.

## Evidence and rerunning

- `result.json`, `gpu-resume.json`, and the three recipe JSON files are byte-identical raw receipts.
- `observed-source/` contains the exact executed wrapper and its main helpers plus the new tests.
- `worker-logs/` preserves all four ranks' stdout and stderr.
- `commands.json` records all three GPU attempts and both full-suite commands.
- `before/` preserves the four previous handoff files; all 219 bindings in the prior validation
  remain recoverable byte-for-byte. Prior failures and source identities are not rewritten.
- The three captured gradient payloads total **7,153,135,725 bytes** and remain in
  `/tmp/dense-replay-update-probe.LY5GmY/`, with digests in the recipe receipts. They contain raw and
  clipped gradients for both paths, names and group layout. They are not committed or uploaded.
  Diagnostic checkpoint files remain in `/tmp/dense-gpu-resume.36BUfj/`.

The immutable input remains the verified 18-file download of NorMuon 3e-4 checkpoint 3126 in
`/tmp/dense-checkpoint-restore.Og1kZs/download`, selected against trusted audit SHA-256
`9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6`.
The campaign rechecks every source checkpoint file after execution. New diagnostic checkpoints
are not primary checkpoints and must not be mixed into the study's 60-checkpoint matrix.

The artifact audit takes no GPU lease and launches no training:

```bash
python3 -B -m scripts.validate_dense_gpu_replay_localization \
  --repository /root/embedding-optimizer-story-refactor \
  --output /tmp/new-replay-localization-validation.json
```

It verifies the raw outcome structure, all current executed-source bindings, old evidence,
captured gradient-file digests, unchanged live numerical source, and the frozen matrix hash.
It refuses `python -O`. The full GPU command in `commands.json` requires new `mktemp` output/log
directories and the exact paused project ownership check; never reuse an existing output directory
or bypass that guard on another host. JSON evidence is portable; full numerical replay also needs
the named digest-bound tensor inputs or an explicitly labeled new capture.

## Failed diagnostic attempts retained

1. `attempt-1/`: the new direct-step wrapper constructed a plain dict while the observed model
   state is an OrderedDict. Strict topology checks rejected it before numerical acceptance.
   The correction preserves the original state container and leaves the strict comparator intact.
2. `attempt-2/`: the direct replay omitted the scheduler step, so its next group learning rate
   differed from the post-scheduler training snapshot. The correction executes the observed real
   LambdaLR function and verifies both pre/post scheduler state. It does not ignore LR differences.
3. `pytest-full-relative-path.xml`: two synthetic-paper subprocess tests loaded the live editable
   checkout after changing directory because the shell used relative `PYTHONPATH=src:.`. Repeating
   the entire suite with this checkout's **absolute** source paths passes **1,337 JUnit test cases**,
   with zero failures, errors or skips, in 133.570 seconds. No tests were excluded or assertions
   relaxed. The new file has nine test methods / 18 JUnit cases including subtests.

These are diagnostic-development failures, not optimizer findings. Their old sources, receipts,
logs and failed suite remain archived. The final successful narrower measurements do not erase
the parent numerical replay failure.

## Next scientific gate

Isolate the first gradient discrepancy with controlled backward/reduction-order comparisons, then
test independent-process checkpoint restoration. Keep the normalization correction and official
NorMuon conformance boundary separate. Only after numerical acceptance and a reviewed, source-bound
transition should a new formal replication namespace be launched from the untrained base model.
Do not repair a historical trajectory by resuming halfway with a new objective scaling, silently
rescale learning rates, or choose new tolerances/configurations based on favorable retrieval scores.
