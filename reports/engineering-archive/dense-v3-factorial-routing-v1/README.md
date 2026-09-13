# Named factorial optimizer and exact 50K batching preparation

Updated 2026-09-07. These are actual CPU optimizer, model-loading and sampler
checks, not formal continuation runs. **The full factorial Trainer, identities,
whole-run reader and reviewed release are still not integrated.** There remain
zero accepted full-horizon v3 primary runs. The manuscript is unchanged.

## Implemented behavior

`src/embed_optim/factorial_v3_optimizer.py` wraps the existing numerical factory
without modifying it or the primary policy. Only the frozen routed AdamW and
Muon settings are admitted; the separately calibrated hidden LR is still an
external scientific input. The three groups retain the same named hidden,
auxiliary-decay and auxiliary-no-decay parameters. Fresh construction observes
empty state and refuses inherited gradients. Every step requires every declared
parameter's gradient; live reordering is rejected before an update.

Saved state carries parameter names, group roles, IDs/order, shapes, FP32 dtypes,
configuration and an actual optimizer step counter. Restore validates these and
all moments plus the actual linear scheduler before either object is modified.
The caller must supply the externally admitted checkpoint step. There is no
automatic legacy relabeling or calibration-state import. Names prevent accidental
positional rerouting; they are not signatures and cannot independently detect an
adversary relabeling equal-shaped moment values. External seals remain required.

## Numerical and complete-model evidence

- Initial 77-case tests exposed a new adapter's incorrect assumption that the
  old development OptimizerConfig supplied `as_dict` (71 failures). The adapter
  now serializes the shared dataclass settings explicitly. Both distinct source
  trees are tested; this is not acceptance of the old training policy.
- The next attempt had two rectangular-Muon oracle failures. Its reference had
  reassociated a BF16 expression, `c*(A@A)` instead of `(c*A)@A`. The independent
  reference now uses scalar matrix products with the declared intermediate
  rounding. Three matrix shapes over eight updates match exactly, **rtol=atol=0**;
  the original looser comparison was tightened, not relaxed. Production kernels
  never changed. Routed AdamW is compared with independent torch.optim.AdamW
  groups, using the declared 2e-7 absolute tolerance for differing arithmetic.
- Four later counterexamples exposed a real omission in the new restore wrapper:
  unchecked extra scheduler fields could replace its optimizer or step method.
  The complete scheduler field set is now checked before either load. The exact
  failed sources/XML and all prior attempts remain preserved.
- The final **82 optimizer cases pass** both in development and in a fresh
  assembly containing the exact corrected candidate config/kernel. Actual
  torch.save/weights-only-load and resumed ten-step toy trajectories match
  uninterrupted parameters exactly. These are CPU fixtures, not a distributed
  save/resume experiment or DenseOn quality measurements.
- `full_model.py` actually loads the immutable pretrained model, checks all 134
  parameters against saved weights before and after constructing both optimizers,
  and observes identical **88 / 1 / 45** routing over **149,014,272 parameters**.
  Both initial states are empty, with the declared 391-step / 40-step-warmup
  scheduler. Corrected config/kernel hashes and every pinned model file are
  checked. The current-source retry is `full-model-retry.json`, SHA-256
  `3eee249cbf189cac920733b4c9fa505cd68158458c091222acf1f5794ccab1ff`.
  There is no full-model forward, backward or optimizer update in this call.

## Newly verified 50K tail issue and prepared fix

`audit_batch_tail.py` uses the actual pinned SentenceTransformer sampler factory
and Accelerate BatchSamplerShard on synthetic row-identity labels. It is not a
repetition of the complete real 500K data audit. All three branch seeds show that
direct reuse of the primary loader (`drop_last=True`, `split_batches=False`,
ST's `even_batches=False`) uses **49,984** unique groups, omits **16**, and ends
with a global **64**-group update. The 500K primary cardinality control consumes
all 500,000 groups and ends at 32; it is not affected by this particular remainder.

Simply setting drop_last=False in the shard creates unequal micro-batch counts
between ranks (1,563 versus 1,562). Ordinary ST arguments also force the setting
back to True under DDP. Neither is an acceptable complete 50K integration.
No factorial run has executed with either setting.

`src/embed_optim/factorial_v3_batches.py` now supplies the actual sampler/factory
needed by a dedicated entrypoint. It retains the upstream permutation exactly.
The first 390 logical batches use sixteen micro-batches of eight; the last 80
groups use sixteen micro-batches of five. Round-robin four-rank sharding gives
each rank 1,564 micro-batches and 391 updates, with four equally sized micros
per update. Each group occurs exactly once. The existing mean-loss/accumulation/
DDP factors therefore give every final group coefficient **1/80**.

The factory intentionally requires explicit non-dropping arguments and refuses
the ordinary primary defaults. A dedicated factorial TrainingArguments/Trainer
must actually retain and verify this setting and callable; do not claim that the
old Trainer has been fixed by merely adding this component.

All 19 sampler/order/coverage/resume-prefix cases pass. Two additional actual
project-InfoNCE FP64 tests compare simulated four-rank gradient reduction with
the complete logical batch for 128 and 80 groups (rtol=atol=1e-12). Combined
development focus **40320 is terminal/pass: 103 cases**. This does not execute
distributed collectives or actual Trainer checkpoint recovery.

## Scope, preservation and next work

Follow [RUNNING.md](RUNNING.md) for exact handles, hashes and test-scope limits.
Whole regression 73858 passes the 3,044-case population collected before the
21 sampler cases were added; it must not be advertised as covering them.
The newer whole regression **21015 is terminal/exit 0**, observed 14:18 UTC.
Its XML reports 3,065 tests, zero failures/errors/skips, and 3,056 serialized
testcase elements. Corrected-source combined focus **64963 also passes all
103 cases**. No call in this archive remains live. Neither result accepts the
distinct whole training candidate or constitutes full factorial integration.

Preserve the six current source/test/helper files under `source-with-batches/`.
The earlier `source-current/` preserves the optimizer checkpoint and initial
tail-audit source; source-first, source-retry and counterexample copies retain
the exact earlier failures. Original candidate, primary numerical/core readers,
old protocols, ledgers, checkpoints and manuscript were not modified. The
candidate's separate eleven source-contract failures remain unwaived.

Next integrate the **actual** factorial Trainer and complete run identity:

1. Retain all 50K groups under the observed four-rank loader, match the fixed
   schedule, and verify the inherited single-normalization/mean-loss policy.
   The sampler requires explicit new argument/source admission, not a silent
   override of the old primary field or a smaller branch dataset.
2. Admit only the new v3 AdamW 3e-5 / Muon 3e-4 source states at step 2345, with
   complete checkpoint and durable-copy verification. Bind revised-parent data
   compatibility and freshly recomputed calibration; never inherit its moments.
3. Use this named optimizer in the real save/load hooks and a distinct deep
   whole-run reader. The old primary hooks still reject hybrid_adamw; do not
   rename it to primary AdamW, monkeypatch guards or waive rejected fields.
4. Integrate all subsequent probe/BEIR/summary/publication requirements and the
   explicit reviewed assembled-source/runtime/parent transition. The old main
   ledger includes paper release before factorial, so it cannot stand in for a
   complete v3 final-paper dependency graph. No old migration is authorized.

No formal run, deployment, controller transition, commit/push, HF action or
protected-helper interaction occurred. Engineering counterexamples and synthetic
tests remain repository provenance only, never manuscript findings.
