# Full-model NorMuon reference probe

Observed September 5, 2026, 18:35 UTC. Engineering-only evidence; no scientific optimizer result.

## Result and why it matters

**All 352 hidden-matrix update comparisons are bitwise equal** between the local implementation
and the authenticated pinned official NorMuon functions. The comparison covers all 88 hidden
matrices, saved/fresh optimizer state, and two probe-loss multipliers. The full
[measurement receipt](full-model-probe.json) records zero update, row-second-moment and hypothetical
next-weight differences. No real model parameter or saved optimizer state was updated.

This narrows the [earlier small-gradient finding](../dense-code-correctness-v1/README.md): the
definition discrepancy is real, but **an effect on actual primary training has not been established**.
The earlier 44/96 nonconforming small-matrix cases remain valid and unchanged. This one full-model
probe does not certify every historical gradient, checkpoint, data batch or the GPU execution path.
The separately reproduced duplicate Trainer/Accelerator normalization remains a scientific blocker.

## Fixed inputs

- Checkpoint: NorMuon 3e-4, step 3126, independently downloaded from immutable HF revision
  `e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae`.
- All 18 downloaded files / 1,351,464,827 bytes are reauthenticated against the trusted
  [ten-run content audit](../../experiment-integrity/huggingface-digest-audit-ten-run.json),
  SHA-256 `9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6`, before deserialization
  and again after the probe. Payloads remain unchanged.
- Probe: exactly materialized rows **0 and 1**, fixed before observing gradients. Both are FiQA
  rows (query IDs 1812 and 4797); the report binds their positive/seven-negative IDs and text digests.
  This is a deliberately bounded engineering input, not a representative training sample.
- Training data manifest SHA-256:
  `9facc18bcd1cad8378cea94746a95ab09804bdf3610796bf9013cdfcc486aee8`.
  Selected identities agree with the canonical first two `rows.jsonl` records before/after the test.
- Model: 149,014,272 FP32 parameters, 134 tensors, 768 embedding dimensions, CPU SDPA,
  training mode, zero dropout, non-reentrant gradient checkpointing. Configured length remains
  8192; actual inputs are **11–611 tokens**, without truncation. This is not maximum-length testing.
- Actual production collator and explicit mean InfoNCE loss, temperature 0.02, one positive plus
  seven negatives per query, no in-batch negatives. All 134 parameter gradients are finite/present.
- Official source: [NorMuon commit c6989a8](https://github.com/zichongli5/NorMuon/blob/c6989a8354730695d9f5a9faa6c55eeb24865209/normuon.py),
  13,062 bytes, SHA-256 `706c1a35fb35342f6ff207f8310b814a1c8f55c6ffba0e843537db434a696141`.
  The two actual upstream functions are executed only after exact authentication.

## Comparisons and limitations

The CPU probe mean loss is 0.09071870. The direct and quarter-scaled raw gradient norms are
4.87362289 and 1.21840572. **Both exceed the declared clipping threshold of 1.0.** Thus the two
scale conditions both activate clipping; their agreement cannot establish that quarter-scaling
is harmless. They are not independent repetitions. In particular, the earlier observation that
many primary norms fall in a different clipping regime is not superseded.

For each condition, the diagnostic compares a local update, the unmodified official update and a
control that changes only the official denominator statement. Only cloned gradients/momentum/
second moments are passed to the functions. The two state conditions use either the checkpoint's
real saved state or zero initial state at those same trained weights. The fresh-state condition is
not pretraining initialization. Hypothetical weights use the saved hidden LR 6.6638225e-5 and
declared decay, but those hypothetical tensors are never installed into the model.

| Loss multiplier | Incoming state | Hidden matrices | BF16 input norm range | Exact local/official updates |
| --- | --- | ---: | --- | ---: |
| 1 | Saved | 88 | 0.00357056–0.03784180 | 88/88 |
| 1 | Fresh | 88 | 0.00265503–0.02709961 | 88/88 |
| 0.25 | Saved | 88 | 0.00357056–0.03784180 | 88/88 |
| 0.25 | Fresh | 88 | 0.00265503–0.02709961 | 88/88 |

The matrix shapes are 768×768, 768×1152 and 2304×768. For every measured input, the BF16
`norm.clamp_min(1e-7)` and `norm + 1e-7` denominators are themselves identical: the added epsilon
rounds away at these norms. This directly explains the denominator agreement **in this probe**.
It is not a claim that epsilon always rounds away, that every training update matches upstream,
or that the separate small-fixture NorMuon replay discrepancy has been explained.

The earlier reference tolerances (`atol=1e-5`, `rtol=1.3e-6`) are unchanged, but this result is
stronger than tolerance agreement: actual tensors match bitwise. The 352 matrix cases are
components/conditions of one checkpoint/data probe, not 352 independent experiments or seeds.

## Reproduce

This command uses the existing independently authenticated download. It does not download model
weights, launch primary training, evaluate BEIR, acquire a GPU lease or mutate production code.
The only network access is a read of the exact pinned official optimizer source.

```bash
audit_work=$(mktemp -d /tmp/normuon-full-reference.XXXXXX)
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
/usr/bin/python3 -B -m scripts.audit_normuon_full_model_reference \
  --repository /root/embedding-optimizer-story-refactor \
  --data-root /root/embedding-optimizer-study/data/denseon-sft-500k-seed42 \
  --download-root /tmp/dense-checkpoint-restore.Og1kZs/download \
  --audit /root/embedding-optimizer-story-refactor/reports/experiment-integrity/huggingface-digest-audit-ten-run.json \
  --audit-sha256 9e1cfceea1d3a821bbe7c6e4881f2b7dbe5faa5b21c92d107bec4eb4c995c5d6 \
  --workdir "$audit_work"
```

The nine focused diagnostic guards are in [focused-tests.xml](focused-tests.xml); the complete
regression receipt is [full-tests.xml](full-tests.xml). Source-addressed validation is separate
from this measurement. Source checkpoints, optimizer payloads, materialized rows and model
parameters are unchanged; the original small-gradient failure receipt remains preserved.

## Next step

Do not introduce this engineering result into the manuscript. Keep the scientific hold and the
existing no-deployment/no-retraining/no-HF-withdrawal boundaries. The pending owner decision is
still a safe pause/handoff of the active evaluation before GPU correctness validation. The
existing normalization candidate is not deployed. `gpu.py` remains outside scope and untouched.
