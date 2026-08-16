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
| Context length | 8,192 query and document tokens |
| Epochs / global batch | 1 / 128 |
| Checkpoints | 20%, 40%, 60%, 80%, 100% of optimizer steps |
| Evaluation | 14 decontaminated BEIR tasks, main score nDCG@10 |
| Default hardware layout | Dense on GPUs 0–3, late interaction on GPUs 4–7 |

Muon and NorMuon are applied only to 2-D matrices in the transformer's hidden layers. Embeddings,
projection heads, norms, and biases use AdamW at `3e-6`, following the optimizer authors' routing
recommendation. AdamW applies its swept learning rate to all parameters. The exact 24-run matrix is
versioned in [configs/experiment.yaml](configs/experiment.yaml), including immutable revisions for
both base checkpoints.

## Install

Python 3.10–3.13 and CUDA GPUs with bfloat16 support are expected. The tested environment uses
Python 3.12, PyTorch 2.9.1, SentenceTransformers 5.7, CUDA 12.9, and eight 80 GB GPUs.

```bash
git clone https://github.com/qcznlp/embedding-optimizer-study.git
cd embedding-optimizer-study

uv sync --extra eval --extra analysis
uv pip install flash-attn --no-build-isolation
source .venv/bin/activate
```

The package can also be installed with pip:

```bash
python -m pip install -e '.[eval,analysis,flash]'
```

Pip follows PyLate 1.6's declared SentenceTransformers 5.3 pin. The uv lock uses the tested
SentenceTransformers 5.7 stack through an explicit compatibility override.

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
  --gpus-b 4,5,6,7 \
  --fail-fast
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

Each output directory contains the resolved configuration, five complete model/optimizer/scheduler
checkpoints, a final model, Trainer state, and a completion record. The loss is explicit group-only
InfoNCE: documents belonging to other queries in the microbatch or on other ranks never enter the
logit matrix.

### 3. Evaluate every checkpoint

```bash
embed-optim-evaluate \
  --matrix configs/experiment.yaml \
  --gpus-a 0,1,2,3 \
  --gpus-b 4,5,6,7
```

Dataset revisions for all 14 LightOn decontaminated BEIR repositories are pinned in
[`decontamination.py`](src/embed_optim/decontamination.py). Dense evaluation runs independent tasks
across four GPUs. Late-interaction evaluation uses PyLate for multivector encoding, the fused
Late Interaction Kernels scorer during training, and FastPLAID retrieval during evaluation.
Evaluation caches are resumable, so an interrupted command computes only missing task/checkpoint
pairs.

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

This writes long-form task results, checkpoint and optimizer summaries, per-task comparisons, system
metrics, training history, coverage checks, and training-dynamics/LR-sensitivity plots under
`reports/`. Once coverage reaches `24 × 5 × 14 = 1,680`, it also replaces the marked results and
systems sections in `docs/blog.md`. `--strict` fails unless the entire matrix is present.

## Performance engineering

- FlashAttention-2, bfloat16 autocast, TF32, non-reentrant gradient checkpointing, and fused AdamW.
- PyTorch's fused Muon functional kernel and an in-repository NorMuon update matched to the official
  implementation.
- Dynamic per-column padding for the nine explicit contrastive fields; no global padding to 8,192.
- Fused Late Interaction Kernels MaxSim scoring during training.
- Length-grouped training and token-budget-packed evaluation with automatic OOM backoff.
- Concurrent four-GPU pools and resumable per-task evaluation caches.

The narrow compatibility shims in [`pylate_compat.py`](src/embed_optim/pylate_compat.py) make PyLate
1.6 work with SentenceTransformers 5.7 without changing scoring semantics. A checkpoint smoke test
verifies save, reload, context lengths, and query-expansion settings.

## Development

```bash
uv sync --extra dev --extra eval
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

Please cite the DenseOn/LateOn paper and NorMuon when using this work; complete BibTeX entries are in
[CITATION.cff](CITATION.cff) and [docs/blog.md](docs/blog.md).
