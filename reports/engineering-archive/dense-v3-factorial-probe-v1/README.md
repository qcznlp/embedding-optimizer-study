# Five-stage continuation probe: queued real execution

The new source-bound entry is live in
`/tmp/dense-v3-factorial-probe.ryorjXJK`. It is a separate operational consumer
of the genuine v3 continuations, not a transition of the stopped historical
controller or an assertion that its old primary identities passed.

## Actual status and exact handles

Both coordinators started at **2026-09-12 17:25:20 UTC** and were observed
live/S at **17:26:40 UTC**, without a GPU lease or worker:

| Pool | Tool session | Registered coordinator | Target |
| --- | ---: | --- | --- |
| a | 54795 | PID 809602 / start 320566873 | 6 runs × 5 stages |
| b | 34578 | PID 809603 / start 320566875 | 6 runs × 5 stages |

Each waits for its **whole six-run training pool** before requesting one token
from that released pool. It holds both original lease namespaces and passes
their descriptors to each direct worker. The unchanged full-BEIR waiters can
use the other released tokens. The probe does not interrupt training, inspect
unrelated processes or use a gap between training cells.

At **17:28:59 UTC**, continuation training was **6/12 complete**. The fourth
pair had started at 15/391 and 3/391; both coordinators and eight exact ranks
were live. Native completion of a branch and its new HF upload are distinct:
only the first four branches' backup verification was observed at that point.

## Input checks and unchanged measurements

The final **19 tests pass**, including actual source-namespace/fixed-probe
loading, an independent NumPy implementation of the six metrics on synthetic
inputs, pessimistic tie handling, and refusal to request leases while training
is incomplete. They are not nineteen experiments or GPU worker tests.

The real preparation finished with exit zero (session **6270**, terminal
`22b197`). It authenticated the genuine first continuation's checkpoint-79
through the original complete-run receipt and reread its eight inference files.
The original native pretrained reader verified every saved tensor fingerprint
and all four actual vector arrays. The pretrained model was **not re-encoded**.
The separate inspect process **86472** also exited zero.

The encoder is the unchanged v3 `encode_state`: BF16, FA2, maximum length
8192, batch 8, normalized query/document prompts and complete before/after
saved-weight comparison. All 60 checkpoint encodings retain their FP32 vectors.
The frozen factorial metric path consumes their explicit FP16 cast and uses
the unchanged `dense_probe_scores` and `_summary` bodies: loss at temperature
.02, positive margin mean/p05, reciprocal rank, strict top-one accuracy and
pretrained top-one agreement. Every sample, task and all five stages remain.
The reference uses the identical recorded encoder settings and cast.

After each actual worker exits zero, the coordinator uses the original native
vector reader, reconstructs all four stored arrays and complete six-metric
overall/fourteen-task output, and records the result. Failure preserves the
attempt and stops its own queue; there is no automatic overwrite or retry.

The first preparation's missing import failed before model reading, output
creation or GPU work. Its source, passing 18 bounded tests and failure history
remain intact. The new source closure is the union of **66 byte-identical
encoder and inference files**, authenticated against both existing source
parents. No prior source binding or numerical function was changed.

## Anchors and observation

- Live entry SHA: `eb4fc27738adaecd9614ba5822255a5dc37de242afabfa53ff8cd8a6c8c73334`.
- Authority SHA: `00d6b42f922b4428352d5b9fc5e5fa7e0436a4aa20ee0c693e57b9f645398ded`.
- Actual input read SHA: `06f7defd44e5f23571e1e75cbcd594378cd099bf84f7cfdbc013f595f9a37933`.
- Tests SHA: `ac0de0699294c46d95add4e9bdb55e96ed2b6d02d274c338d1840ba23fc3b2be`.
- Source copy SHA: `bb6327e4a5d578fe3a5e8c12b79928be76310e8484468b35152d4952c0c2aad0`.
- Outputs: `/root/embedding-optimizer-v3-experiment/analyses/dense-v3-factorial-probe-v1`.

Use only the exact registered-handle observer, with a new output path:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /tmp/dense-v3-factorial-probe.ryorjXJK/observe.py --output /tmp/probe-status-NEW.json
```

Do not edit/restart the live entry. **Zero of 60 new checkpoint probes** are
complete at this snapshot; the one reused reference is complete. No factorial
effect, full-corpus probe result, GPU resume equivalence, source release or
manuscript finding is claimed. The complete target remains 60 checkpoints,
840 checkpoint/task probe rows, and the separate 168 final BEIR evaluations.
