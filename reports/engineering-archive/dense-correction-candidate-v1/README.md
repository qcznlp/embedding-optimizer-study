# Prepared Dense numerical correction: integrated acceptance, not deployment

GPU campaigns completed **2026-09-06 09:28 UTC**. This is engineering provenance only.
No part of this incident or diagnostic belongs in the manuscript.

The explicitly versioned correction is now implemented in a new detached preparation checkout,
`/tmp/dense-correction-source.0YHywN`, based on `f231a6430712388778f32ad1736a4cb6de3bec3e`.
The six original numerical files in live main and the isolated paper checkout remain unchanged.
No formal run, evaluation resume, ledger transition, commit/push or HF deletion occurred.

## Verified outcomes

| Check | Observed result | Boundary |
| --- | --- | --- |
| Actual NorMuon factory vs pinned official functions | 96/96 exact weight, momentum and row-second-moment updates on CPU; separately 96/96 on GPU | Four shapes × six scales × four updates, not retrieval evidence |
| Unchanged Muon branch | 96/96 bitwise update/state matches per device against the old implementation | This correction changes only NorMuon's denominator policy |
| Actual integrated Trainer, short inputs | 9/9 raw and clipped global-gradient checks, all 134 parameter tensors | Default DDP communication; 288 predetermined synthetic rows |
| Same integrated Trainer, 8192-token stress | 9/9; maximum length observed on all four ranks; no OOM | Unchanged stress fixture and tolerance, not a natural-text quality test |
| Actual new checkpoint save/restore | 24/24 exact controlled replay comparisons, three algorithms × two restart points × four ranks | Same process group; fixed-layout reduction is diagnostic-only |
| Actual unpatched `run_training` function | Three baseline calls and three step-2 continuation calls complete | Local authenticated trained weights, short synthetic dataset, three-step horizon |
| Candidate regression suite | 923 cases: 912 passed, **11 failed**, no errors/skips or exclusions | All 11 are unwaived old source-contract gates; the candidate is not whole-repository green |
| Targeted configuration/entry guards | 75/75 passed | Does not replace full-stack or scientific acceptance |

The short-input maximum raw gradient difference is `1.1920928955078125e-7`; the stress maximum
is `4.76837158203125e-7`. Maximum clipped difference is `1.4901161193847656e-8` in both.
The independent manual-autograd oracle, `atol=5e-6` / `rtol=5e-4`, global query groups
`128/128/32`, and Trainer accumulation counts `4/4/1` are unchanged. The first update has zero
warmup learning rate; the other two have active learning rates. No zero-step-only acceptance.

The GPU is an **NVIDIA L20Z**, not a literal H100. The stress maximum allocated memory is
52,909,864,448 bytes, including the second full model used by the oracle; it is not a formal
training-memory estimate or an optimizer-performance comparison.

## Exactly what changed

The [sealed candidate source](candidate-source/) and [source manifest](prepared-source-v2.json)
are the reviewable artifacts; they are not installed into the live training checkout.

1. A new explicit `single-normalization-deterministic-backward-v1` policy assigns the mean-loss
   divisor to the pinned inherited Trainer step. Accelerator's second divisor becomes one;
   Trainer's real full/tail accumulation schedule remains unchanged. Unsupported stacks fail.
2. The model loader sets and verifies the deterministic-backward flag on all 22 owned
   ModernBERT attention instances. No global library, kernel or communication patch is installed.
3. NorMuon explicitly selects `unfused-bfloat16-additive-eps-v2`, matching the
   [pinned official denominator](https://github.com/zichongli5/NorMuon/blob/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py).
   Muon's existing clamped-denominator branch remains unchanged, consistent with the separately
   [pinned PyTorch implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/torch/optim/_muon.py).
4. Scheduled checkpoints record a numerical contract. Restore rejects missing/changed contracts
   and incompatible optimizer-group markers. Legacy state is not relabeled as corrected state.
   This receipt is a numerical contract, not a substitute for future full recipe/data/source
   resume validation by the revised execution protocol.
5. New Dense execution requires the explicit policy and a fresh output namespace. A write-before-
   rank-check race was found in preparation and fixed with distributed initialization/barrier
   before shared artifact creation; the actual four-rank entrypoint now exercises that path.
6. Historical metadata remains readable: optimizer serialization omits an irrelevant null operator
   field for AdamW, and metadata-only old W&B configuration objects retain their original IDs.

The [12-run proposal](candidate-source/configs/dense_correctness_candidate.yaml) changes only the
declared numerical choices and new run/output/W&B namespaces. Every other resolved scientific
field matches the old matrix: fixed base revision, 500k data view, seed, seven own negatives,
no in-batch negatives, temperature, four-rate grids, context, batch, clipping, scheduler and five
checkpoint fractions. This is a **proposal**, not a newly frozen formal protocol.

## Checkpoint and entrypoint coverage

[Controlled replay](resume/result.json) uses the actual prepared `OptimizerTrainer`, its actual
save/restore methods and native GPU deserialization. Eight persistent workers per rank are used.
Entry/intermediate/final weights, optimizer and scheduler state, raw/clipped gradients, losses,
actual row order and per-rank Python/NumPy/CPU/CUDA RNG match exactly at both restart points.
Every new saved contract is read back, and the original authenticated legacy checkpoint is
rejected as a corrected continuation input for each optimizer.

As in the [preceding diagnostic control](../dense-canonical-replay-v1/README.md), the replay
campaign explicitly attaches fixed lexical-vector communication. It does **not** imply that
default DDP will reproduce every post-restart trajectory bitwise. The independent-process result
in the preceding archive covers that older controlled candidate, not these new source bytes.
New baseline fingerprints and file inventories are preserved for a later process-restart check.

[Entrypoint execution](entrypoint/result.json) makes six actual `run_training` calls without
replacing Trainer, model loader, loss, collator, callbacks, checkpoint functions or communication.
It exercises dataset loading/column selection, shared output setup, all three optimizers, native
resume, final model saving and `completed.json`. Each final inference model exactly equals its
last scheduled checkpoint. The source checkpoint is unchanged after each continuation.
The explicit diagnostic configuration uses 288 rows, an authenticated local trained checkpoint,
three steps and fractions `1/3, 2/3, 1`; W&B is disabled. No formal CLI lock was bypassed.

## Failures and source history are retained

- The original NorMuon branch still fails 44/96 small-matrix comparisons on each device. The
  correction does not erase that counterexample or retroactively repair any primary checkpoint.
- Initial CPU/GPU reference launch attempts stopped on a missing audit-support import before
  entering the reference update loop. The exact pre-existing `artifact_inventory.py` support
  module was then copied into the detached checkout; the numerical reference script was unchanged.
  These startup failures are described in [preparation history](preparation-history.json), not
  misrepresented as numerical results or original raw logs.
- The first [full candidate suite](tests/candidate-base-full.xml) has 911 cases / 17 failures:
  six real metadata compatibility problems and eleven frozen-source refusals. The six regressions
  were fixed, then the [new full suite](tests/candidate-base-full-v2.xml) ran without exclusions:
  923 cases / eleven refusals. No protocol hash, tolerance or assertion was loosened.
- [The first source manifest](prepared-source.json) is preserved, not overwritten with the final
  source identity. Seven [initial source files](v1-source/) exactly match its pre-existing hashes.
  Three files were reconstructed from the recorded subsequent edits and checked against those
  earlier hashes; four remained byte-identical. This is disclosed source recovery, not a rerun.
- The preceding accepted control and its 86 bound artifacts remain byte-for-byte preserved;
  superseded top-level documents are under [before/](before/).

## Reproduce, preserve, then review the transition

[commands.json](commands.json) records the exact commands, directories, process-group IDs and
observed exit codes. All four GPU launchers exited zero; all 32 per-rank output/error logs are
archived here. Reruns require new `mktemp -d` output/log directories and the exact original
paused-dispatch handoff. Never overwrite an audit receipt or substitute a changed source manifest.

The actual source selection for numerical diagnostics is:

```text
PYTHONPATH=/tmp/dense-correction-source.0YHywN/src:/root/embedding-optimizer-story-refactor:/tmp/dense-correction-source.0YHywN
```

The detached candidate's full suite is distinct from the isolated paper checkout's regression
suite. Their counts must never be combined into a fictitious all-pass corrected-repository result.
To recheck this evidence without training, use the isolated checkout's source path:

```bash
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
python -m scripts.validate_dense_correction_candidate \
  --repository /root/embedding-optimizer-story-refactor --output /tmp/new-candidate-validation.json
```

Large synthetic checkpoint/fingerprint payloads remain under the exact temporary roots in the
commands and receipts. They are not uploaded or deleted. The primary 12/12 runs and 60/60 preserved
checkpoints remain on scientific hold. BEIR's same three exact handles stay stopped with their
original completed prefix and lease; no old zero-step migration applies.

The [next-step proposal](replication-transition-proposal.md) separates scientific invariants,
new source/protocol acceptance, durability, controller handoff and publication authority.
Do not merely rehash eleven old locks or import their old results into the new namespace.
GitHub issue writes remain blocked by the already recorded integration 403; this local milestone
was not published remotely, and no alternate identity or credential was used to bypass it.
