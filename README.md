# Muon for embedding-model training

A reproducible study of AdamW, Muon, and NorMuon for supervised fine-tuning of
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised) and
[LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised). The repository builds one
shared 500,000-query training set, runs 24 controlled training jobs, evaluates five checkpoints from
every job on 14 pinned decontaminated BEIR datasets, and produces publication-ready tables and plots.

The live experiment dashboard is [Weights & Biases: embedding-optimizer-study](https://wandb.ai/stevezenguom/embedding-optimizer-study).
The full research write-up is in [docs/blog.md](docs/blog.md).

## Experimental contract

| Variable | Value |
| --- | --- |
| Model families | DenseOn-unsupervised and LateOn-unsupervised |
| Optimizers | AdamW, Muon, NorMuon |
| Hyperparameter sweep | Four learning rates per optimizer |
| Training examples | 500,000 identical query groups per run |
| Contrastive group | 1 positive + 7 seeded random hard negatives |
| In-batch negatives | Disabled |
| Auxiliary objectives | No knowledge distillation; no Matryoshka loss |
| Context length | 8,192 query and document tokens |
| Epochs / nominal global batch | 1 / 128 (final partial step: 32) |
| Checkpoints | 20%, 40%, 60%, 80%, 100% of optimizer steps |
| Evaluation | 14 decontaminated BEIR tasks, main score nDCG@10 |
| Default hardware layout | Dense on GPUs 0–3, late interaction on GPUs 4–7 |

Following the [original Muon recipe](https://kellerjordan.github.io/posts/muon/), Muon and NorMuon are
applied only to 2-D matrices in the transformer's hidden layers. Embeddings,
projection heads, norms, and biases use AdamW at `3e-6`, following the optimizer authors' routing
recommendation. This auxiliary AdamW uses `betas=(0.9, 0.999)` and `eps=1e-8`; AdamW applies its swept
learning rate to all parameters with the same moments. The exact 24-run matrix is
versioned in [configs/experiment.yaml](configs/experiment.yaml), including immutable revisions for
both base checkpoints.

## Install

Python 3.10–3.13 and CUDA GPUs with bfloat16 support are expected. The formal runs currently record
Python 3.12, PyTorch 2.9.1, SentenceTransformers 5.7, CUDA 12.9, and four NVIDIA L20Z devices per
job, with two jobs running concurrently across eight devices. The exact formal package contract is
machine-readable in [configs/formal_runtime.json](configs/formal_runtime.json).

```bash
git clone https://github.com/qcznlp/embedding-optimizer-study.git
cd embedding-optimizer-study

uv sync --extra eval --extra analysis
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate
```

The checked-in `uv.lock` is the portable development/CI environment and currently resolves PyTorch
2.9.0; it is intentionally not presented as the runtime that produced the formal artifacts. Before
formal training or evaluation, use the provisioned CUDA 12.9 environment and verify its interpreter:

```bash
/usr/bin/python3 -m embed_optim.runtime --spec configs/formal_runtime.json
```

The command prints the complete observed runtime and exits nonzero on any Python, Torch CUDA-build,
training-library, or evaluation-library mismatch. Use that same verified interpreter for training,
the matrix supervisor, the evaluation supervisor/coordinator, and `--worker-python`. The formal matrix,
individual formal training workers, and evaluation coordinator also run this verification
automatically; smoke and stress configs remain portable by omitting `formal_runtime`.

The package can also be installed with pip:

```bash
python -m pip install -e '.[eval,analysis,flash]'
```

Pip follows PyLate 1.6's declared SentenceTransformers 5.3 pin. The uv lock uses the tested
SentenceTransformers 5.7 stack through an explicit compatibility override.
Built distributions include the frozen study configs, evaluation workers, blog, citation metadata,
and third-party notices. If a bundled `configs/*.yaml` path is absent from the working directory, the
CLI automatically resolves it from the installation prefix.

Authenticate separately for services that need it; credentials are never stored in this repository:

```bash
huggingface-cli login
wandb login
```

## Reproduce the study

### 1. Materialize the shared dataset

```bash
embed-optim-prepare --output data/denseon-sft-500k-seed42
```

The builder pins source revision `1ca463331ed637d25c1058567e932e0d3bad2983`, intersects query
IDs with the mined-score tables, allocates 500,000 examples proportionally across the seven sources,
and applies the DenseOn paper's NV score threshold (`negative < 0.95 × positive`). It considers the
first ten qualifying negatives and samples seven with a stable BLAKE2-derived seed. The output
contains:

- a Hugging Face Dataset used by both model families;
- `manifest.json` with source revisions, quotas, parameters, and dataset fingerprint;
- `rows.jsonl` with every query, positive, and negative document ID;
- a SHA-256 over the canonical row manifest.

Re-running with the same inputs produces the same query ordering and negative selections.

### 2. Run all training jobs

```bash
embed-optim-matrix \
  --matrix configs/experiment.yaml \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
```

The scheduler starts one four-GPU DenseOn job and one four-GPU LateOn job concurrently, then advances
each queue independently. Once one family drains, its pool automatically steals remaining jobs from
the other family, avoiding an idle four-GPU tail. Completed jobs are skipped on restart. Individual
runs can be launched with:

```bash
torchrun --standalone --nproc-per-node=4 -m embed_optim.train \
  --matrix configs/experiment.yaml \
  --model-family dense \
  --run-id muon-lr3e-4
```

For a bounded checkpoint replay, use `--stop-after-step N` rather than changing `--max-steps`.
The former preserves the full-run scheduler horizon, saves the stop-step checkpoint and timing
ledger, and writes `diagnostic_completed.json` instead of a formal completion marker.

For long matrices, omit `--fail-fast` as above so a failure on one pool does not terminate an
unrelated healthy job on the other pool. The command returns nonzero after the remaining queue drains
if any job failed; rerunning it resumes only incomplete runs from their latest structurally valid
and deeply audited checkpoint, falling back to an earlier declared checkpoint if the latest payload
is corrupt. The deep resume gate validates the mixed-optimizer group algorithms, hyperparameters,
scheduled learning rates, per-parameter state fields/shapes and AdamW step counters, together with
the scheduler's base/last learning rates and step count. Floating-point model tensors must be finite,
and consecutive formal checkpoints must not contain identical model payloads. It stops instead of
silently restarting from the base when no valid checkpoint remains.
Use `--fail-fast` only for smoke tests or when immediate cross-pool shutdown is
intentional.

Each output directory contains the resolved configuration, five complete model/optimizer/scheduler
checkpoints, a final model, Trainer state, and a completion record. The loss is explicit group-only
InfoNCE: documents belonging to other queries in the microbatch or on other ranks never enter the
logit matrix. Four-GPU microbatches contain 32 examples and accumulate four times; because 15,625
microbatches is not divisible by four, the last of 3,907 optimizer steps contains one microbatch and
all 500,000 examples are consumed exactly once.

For unattended multi-day execution, wrap the matrix in the restart supervisor:

```bash
embed-optim-supervise \
  --matrix configs/experiment.yaml \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
```

The matrix already requeues failed distributed child jobs from their latest deeply resumable
checkpoint. The supervisor adds protection around the top-level orchestrator itself: after any exit,
it recomputes structurally complete runs from their terminal artifacts and relaunches only the
remaining matrix. `--wait-for-pid PID` adopts an already-running matrix without interrupting it;
`--max-launches N` is available when bounded retries are preferred. Use `--sequential-families` to
finish the families in the order supplied to `--families` while still allowing both four-GPU pools
to work on that family.

Deep-audit newly written checkpoints continuously from a CPU-only sidecar:

```bash
embed-optim-watch-checkpoints \
  --matrix configs/experiment.yaml \
  --watch \
  --state logs/checkpoint-audit.json
```

The watcher waits until each scheduled checkpoint is atomically resumable, then fully loads its
model/optimizer/scheduler/training-argument payload, checks all four RNG archives, validates the
runtime contract, and rejects an unchanged model relative to the preceding checkpoint. Its atomic
JSON state makes progress externally observable and ensures an unchanged payload is audited only
once; changing any checkpoint file causes a fresh audit. It performs no GPU work. Add
`--fail-on-problem` when the watcher should terminate immediately on an audit failure, or omit
`--watch` for a single scan.

After all runs finish, publish deterministic canonical W&B curves reconstructed from each final
Trainer state:

```bash
embed-optim-sync-wandb --matrix configs/experiment.yaml
```

The canonical run ID is content-addressed by the normalized history. Re-running the command verifies
and skips an identical remote run, while checkpoint-resume segments remain available as raw system
telemetry. This avoids backward or duplicated optimizer steps in the comparison dashboard without
deleting source runs. Resume-local Trainer terminal summaries are excluded; canonical system
summaries use useful wall time and throughput reconstructed from the audited non-overlapping timing
ledger.

### 3. Evaluate every checkpoint

```bash
embed-optim-evaluate \
  --matrix configs/experiment.yaml \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
```

For the full unattended handoff from training to evaluation, use the persistent supervisor instead:

```bash
embed-optim-supervise-evaluation \
  --matrix configs/experiment.yaml \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7 \
  --python /usr/bin/python3 \
  --worker-python /usr/bin/python3
```

It remains CPU-only while it waits for all 24 structurally complete training runs. It then launches
the resumable evaluator and runs the strict aggregation audit after every attempt. A worker,
coordinator, or task failure causes another launch after `--restart-delay`; already validated MTEB
results are reused. The supervisor exits successfully only after the audit proves all 1,680 expected
run/checkpoint/task results and the complete training/data/runtime contract. It then idempotently
publishes all canonical W&B histories and reruns strict aggregation with final Markdown-blog
rendering. Failures in either finalization step are retried without relaunching GPU evaluation.
`--max-launches N` provides an optional retry bound; zero, the default, keeps recovering unattended.
Use `--skip-wandb-sync` only for reproductions that intentionally have no W&B destination.

Dataset revisions for all 14 LightOn decontaminated BEIR repositories are pinned in
[`decontamination.py`](src/embed_optim/decontamination.py). Dense evaluation runs independent tasks
across four GPUs. Late-interaction evaluation uses PyLate for multivector encoding, the fused
Late Interaction Kernels scorer during training, and FastPLAID retrieval during evaluation.
FastPLAID is explicitly pinned to `nbits=4`, `n_ivf_probe=8`, `n_full_scores=8192`, and seed 42.
GPU 4–7 begin the LateOn checkpoint queue while GPU 0–3 run dense evaluation; when dense evaluation
finishes, GPU 0–3 automatically join the remaining LateOn queue. Evaluation caches are resumable, so
an interrupted command computes only missing task/checkpoint pairs. Dense and distributed LateOn
workers are launched by the same Python interpreter as the orchestrator (override it explicitly with
`--worker-python`). Before any GPU work, a runtime preflight requires Torch, Transformers, and
SentenceTransformers, PyLate, and Late Interaction Kernels to exactly match the versions recorded
during training. The same preflight reconstructs the audited training-data view, parses every selected
safetensors payload, verifies the serialized batch/warmup/precision/runtime arguments, validates
optimizer topology and state finiteness, checks scheduler steps, and CRC-checks every rank RNG
archive, so a damaged or protocol-divergent checkpoint cannot consume days of formal evaluation
before being rejected. It then records an
immutable manifest including MTEB, FlashAttention, and FastPLAID so a resumed results directory cannot
silently mix evaluator stacks.
The same manifest hashes all eight evaluation/aggregation source files and verifies that the selected
worker interpreter imports the identical package sources. Resuming after any scoring-code change fails
before GPU work instead of mixing implementations in one result directory.
LateOn's corpus multivectors are released in place as soon as FastPLAID finishes building each
on-disk index, before the search phase begins. Temporary indexes are removed after every task,
including failed attempts, bounding both host-memory and disk accumulation across the 840 LateOn
task/checkpoint pairs.
Task results and model metadata are committed with atomic replacement, while cross-process updates to
MTEB's shared `run_settings.jsonl` are serialized with a file lock. Distributed LateOn workers use a
single metadata writer, and strict aggregation validates the singular MTEB 2.18 or plural MTEB 2.19+
run-settings schema against the exact runtime version recorded in the manifest.

For a targeted evaluation:

```bash
embed-optim-evaluate \
  --families dense \
  --run-ids adamw-lr3e-6 muon-lr3e-4 \
  --stages 1 5 \
  --tasks SciFact FiQA2018 TRECCOVID
```

### 4. Aggregate results

```bash
embed-optim-aggregate --strict
```

This writes long-form task results, checkpoint and optimizer summaries, per-task and paired-effect
comparisons, system metrics, training history, coverage checks, and training-dynamics/LR-sensitivity
plots under `reports/`. Once coverage reaches `24 × 5 × 14 = 1,680`, it also replaces the marked
results and systems sections in `docs/blog.md`. `--strict` fails unless the entire evaluation matrix is present
and all 24 runs have five non-empty, step-consistent model/optimizer/scheduler/RNG checkpoints plus a
final model and Trainer state. Checkpoint loss histories must also have unique increasing steps and
finite loss, gradient-norm, learning-rate, and epoch values.
The dynamics deliverables include optimizer-level mean ± LR-configuration standard deviation,
separate three-panel figures showing every one of the four LR runs at all five checkpoints, the
five-point trajectory of each optimizer's best final configuration, and all 120 raw run/checkpoint
summaries in `checkpoint_summary.csv`.
The same gate verifies each resolved run config against the matrix, checks the copied dataset manifest
and row count against the shared source, requires an identical runtime dataset-view fingerprint across
all runs, requires four recorded ranks, and enforces an identical parameter partition across learning
rates and optimizers within each model family. Every result must also identify the exact local
run/checkpoint, pinned decontaminated dataset revision, expected split/subset, MTEB version, positive
evaluation time, and nDCG@10 main score. Its recorded evaluation runtime must match both every other
result and the corresponding training runtime. Model metadata must independently record the 8,192
token limit and the expected representation/scorer pair (768-dimensional cosine for DenseOn;
128-dimensional PyLate MaxSim for LateOn).
The final gate also recomputes every evaluation-source SHA-256 recorded before the first GPU job.
It parses every safetensors header and tensor extent, fully loads every optimizer and scheduler state
with PyTorch's restricted weights-only loader, and CRC-checks every rank RNG archive. Non-empty but
truncated or structurally invalid checkpoint payloads therefore cannot satisfy `--strict`.
It also verifies the matrix itself against the frozen 24-run experimental contract: exact base models
and revisions, both model families, all optimizer/LR combinations, the shared data and seed, batch and
8,192-token settings, optimizer hyperparameters, and five checkpoint fractions. Thus a modified but
internally self-consistent matrix cannot satisfy the final completion gate.
The dataset gate independently streams all 500,000 canonical rows, recomputes their SHA-256, verifies
the exact seven-source quotas, rejects duplicate queries or positive/negative overlap, and requires
seven distinct seeded choices from each ten-negative candidate pool. It also reloads the materialized
Hugging Face Dataset, checks its row count, columns, and fingerprint, then reconstructs the fixed
ten-column training view and requires every completion record to match that exact view fingerprint.
Wall time, Trainer throughput, CUDA memory peaks, checkpoint/optimizer-state sizes, GPU identity, and
the five key package versions are required and checked for finite values and cross-run consistency.
Each successful checkpoint atomically appends a non-overlapping maximum-rank segment to
`accepted_timing.json`. Strict aggregation verifies step continuity, positive finite durations, the
recorded sum, and the final step before using that ledger for throughput. Work after the latest
durable checkpoint on a failed attempt, restart initialization, and downtime are therefore excluded
without relying on a human to reconstruct every retry. Historical work retained before the atomic
ledger existed must provide timezone-aware start/end evidence for each segment; strict audit checks
non-overlap, timestamp-derived duration, checkpoint boundaries, and the exact recorded sum.

## Performance engineering

- FlashAttention-2, bfloat16 autocast, TF32, non-reentrant gradient checkpointing, and fused AdamW.
- An unfused bfloat16 Muon Newton–Schulz path that preserves PyTorch Muon's polynomial while avoiding
  a reproducible CUDA `addmm` failure; checkpoints pin implementation ID `unfused-bfloat16-v1`.
- Multi-step numerical regression tests compare AdamW and Muon with PyTorch references, lock Muon's
  unfused expression, and compare NorMuon with official commit
  `c6989a8354730695d9f5a9faa6c55eeb24865209`.
- Dynamic per-column padding for the nine explicit contrastive fields; no global padding to 8,192.
- Fused Late Interaction Kernels MaxSim scoring during training.
- Length-grouped training and token-budget-packed evaluation with automatic OOM backoff.
- Concurrent four-GPU pools with work-stealing for both training and evaluation, corpus-size-aware
  longest-processing-time-first task scheduling, and resumable split/subset-aware evaluation caches.

The narrow compatibility shims in [`pylate_compat.py`](src/embed_optim/pylate_compat.py) make PyLate
1.6 work with SentenceTransformers 5.7 without changing scoring semantics. A checkpoint smoke test
verifies save, reload, context lengths, and query-expansion settings.

## Development

```bash
uv sync --extra dev --extra eval --extra analysis
pytest
ruff check src tests scripts/eval
ruff format --check src tests scripts/eval
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment-change and reporting requirements.

## Provenance and license

This repository started from LightOn's open
[`mdenseon-mlateon`](https://github.com/lightonai/mdenseon-mlateon) training/evaluation code and keeps
its Apache-2.0 license. The study additionally implements the optimizer experiment, deterministic data
materialization, explicit no-in-batch losses, checkpoint matrix, pinned decontaminated tasks, and
reporting pipeline.

Please cite the original Muon work, the DenseOn/LateOn paper, and NorMuon when using this study;
complete citation metadata is in [CITATION.cff](CITATION.cff), with linked references in
[docs/blog.md](docs/blog.md).
