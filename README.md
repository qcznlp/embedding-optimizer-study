# Muon for embedding-model training

A reproducible study of AdamW, Muon, and NorMuon for supervised fine-tuning of
[DenseOn-unsupervised](https://huggingface.co/lightonai/DenseOn-unsupervised) and
[LateOn-unsupervised](https://huggingface.co/lightonai/LateOn-unsupervised). The repository builds one
shared 500,000-query training set, runs 24 controlled training jobs, evaluates five checkpoints from
every job on 14 pinned decontaminated BEIR datasets, and produces publication-ready tables and plots.

The live experiment dashboard is [Weights & Biases: embedding-optimizer-study](https://wandb.ai/stevezenguom/embedding-optimizer-study).
The final sync reads back and rehashes every remote history row, verifies exactly one current run per
matrix identity, then marks the content-verified 24-run discovery matrix with the
`canonical-current` tag; historical retries and superseded content-addressed histories remain
visible for provenance.
The full research write-up is in [docs/blog.md](docs/blog.md). A follow-on NAACL study connecting
optimizer updates, weight trajectories, representation geometry, and retrieval behavior is specified
in [docs/naacl-paper-plan.md](docs/naacl-paper-plan.md). The result-safe, ACL-formatted manuscript
source is under [paper/](paper/README.md); unresolved evidence gates render as visible placeholders
rather than silently becoming prose claims.

The complete training-only systems and loss-dynamics tables can be regenerated independently of
retrieval evaluation with `embed-optim-summarize-training`; its manifest binds all 24 completion,
checkpoint-schedule, and canonical Trainer-history sources.
`embed-optim-plot-training` then renders the complete five-stage loss trajectories and native-recipe
systems trade-offs from those declared tables, with its own content-addressed plot manifest.
After strict retrieval coverage reaches 1,680/1,680,
`embed-optim-summarize-retrieval-dynamics` reconstructs all 120 checkpoint means directly from the
provenance-valid task files, joins audited useful wall time, and reports observed first passage to a
within-family AdamW reference without interpolation or silently dropping right-censored runs. The
rule and its 160/1,680-unit visibility disclosure are content-locked in
`configs/retrieval_dynamics_protocol.json`; this is a prospective completion analysis rather than a
preregistration.
The cross-space mechanism bridge also joins the audited ten-observation trailing training loss at
all 120 checkpoints to their BEIR means. Its checkpoint-level and within-run first-difference
associations are explicitly post-hoc—the partial-result visibility and claim boundary are recorded
in `configs/loss_retrieval_diagnostic.json`—and never substitute lower loss for retrieval evidence.

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
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
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
Built distributions include the frozen study configs, evaluation workers, blog, NAACL follow-on
plan, result-safe manuscript source and generated-table placeholders, checked-in weight-space
tables/figures and their content-addressed manifest, citation metadata, and third-party notices. The
source archive preserves the repository-relative paper build topology; the wheel carries the same
manuscript sources for inspection. The installed `docs/` and `reports/` topology preserves the
blog's local links. If a bundled `configs/*.yaml` path is absent from the working directory, the CLI
automatically resolves it from the installation prefix.

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
The Arrow directory is intentionally excluded from Git. The small distributable
`configs/training_data_contract.json` receipt locks its manifest hash and paper-relevant constants;
five independently frozen downstream protocols bind the same hash. Formal experiment commands still
require and byte-verify the original local manifest and row ledger, while a clean source checkout can
audit static documentation without pretending the full training data is packaged.

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
unrelated healthy job on the other pool. Each run is retried from its latest audited checkpoint up to
`--max-retries` times after the initial launch (default: two); a zero process exit without the strict
terminal artifacts is treated as an incomplete attempt. The command returns nonzero only when a run
exhausts that budget, while unrelated queued work still drains. Rerunning it resumes only incomplete
runs from their latest structurally valid and deeply audited checkpoint, falling back to an earlier
declared checkpoint if the latest payload is corrupt. The deep resume gate validates the
mixed-optimizer group algorithms, hyperparameters,
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
/usr/bin/python3 -m embed_optim.checkpoint_watch \
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
`--watch` for a single scan. When the matrix declares `formal_runtime`, the watcher verifies its own
interpreter and every frozen package version before reading any checkpoint; use the same verified
Python that writes the training artifacts.

After all runs finish, publish deterministic canonical W&B curves reconstructed from each final
Trainer state:

```bash
embed-optim-sync-wandb --matrix configs/experiment.yaml
```

The canonical run ID is content-addressed by the normalized history. Re-running the command reads
all remote rows, normalizes and rehashes them, audits current-run uniqueness, and skips only an
exactly matching run; use
`--skip-remote-history-verification` only for an explicitly non-formal availability check. The local
training-dynamics report contains 9,384 finite loss rows (391 per run), while each canonical W&B
history adds one explicit step-3,907 terminal/system row, for 9,408 remote rows in total. Raw
checkpoint-resume segments remain available as system telemetry. This avoids backward or duplicated
optimizer steps in the comparison dashboard without deleting source runs. Resume-local Trainer
terminal summaries are excluded; canonical system summaries use useful wall time and throughput
reconstructed from the audited non-overlapping timing ledger.

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

To protect evaluators that are already running under independent Dense and Late coordinators,
adopt both processes instead of launching a duplicate matrix:

```bash
embed-optim-supervise-evaluation \
  --wait-for-pid DENSE_COORDINATOR_PID \
  --wait-for-pid LATE_DISPATCHER_PID \
  --wait-for-command scripts/eval/dense_parallel.py \
  --wait-for-command scripts/eval/late_interaction.py \
  --evaluation-only \
  --python /usr/bin/python3 \
  --worker-python /usr/bin/python3
```

The supervisor waits until every adopted PID exits and no matching worker command remains, then
launches the resumable evaluator only as a recovery pass. Command-fragment adoption covers orphan
workers even if their coordinator exits and prevents duplicate checkpoint scoring. `--evaluation-only`
stops after strict coverage succeeds, leaving W&B publication, mechanism experiments, and final report
rendering to a separately armed
`embed-optim-post-eval-pipeline`. This mode is safe for split-family or externally recovered
evaluators because no second GPU coordinator starts while any adopted owner is alive.

If retrieval evaluation is already running, arm the resumable post-evaluation handoff instead of
starting a second coordinator:

```bash
embed-optim-post-eval-pipeline \
  --wait-for-command scripts/eval/dense_parallel.py \
  --wait-for-command scripts/eval/late_interaction.py
```

It waits for the strict progress snapshot to reach 1,680/1,680, allows evaluator processes to settle,
then executes the strict evaluation/W&B/blog gates, common-state matrix and exact spectra, both
representation tiers, all summaries and plots, the cross-space bridge, and final repository
validation. It also completes the eight-run hybrid-AdamW routing control, all three confirmatory
training/evaluation seeds, and the three-seed shared-checkpoint short branches before rebuilding the
ACL-format paper draft. A terminal strict paper audit prevents the ledger from completing while any
headline or generated result table remains pending, differs from its recorded hash, or any declared
evidence manifest is incomplete. Dense and LateOn probe
exports run separately with conservative
family-specific batch sizes while using all declared GPUs. Every step has an isolated log and an
atomic JSON ledger under `logs/post-eval-pipeline/`; failed steps retry and leave an exact restart
diagnosis. `--wait-pids` can additionally hold the handoff until known evaluator coordinators exit.
Repeatable `--wait-for-command` fragments also cover replacement or orphan workers, preventing the
first mechanism job from sharing GPUs with an evaluator whose coordinator PID changed. The matcher
ignores fragments appearing only as another supervisor's adoption declaration. `--dry-run` prints
the complete command plan without waiting or launching work.

The final blog has two independent generated sections. `embed-optim-render-mechanism-report` binds
retrieval dynamics, common-state spectra, representation geometry, and the descriptive temporal
bridge. `embed-optim-render-outcome-report` then binds hybrid AdamW routing, scale-matched virtual
steps, shared-start short branches, and the validation-frozen three-seed BEIR confirmation. Both
sections are content-hashed and idempotently replace only their declared Markdown marker regions;
the terminal outcome renderer also verifies that the mechanism marker still byte-matches its source
report before signing the final whole-blog hash.
For compatibility with an already-armed older handoff process, the mechanism renderer materializes
the strict retrieval-dynamics report if its manifest is absent; that fallback still requires the
complete 1,680-unit coverage contract and cannot render from partial results.

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

For a continuously refreshed count that uses the same strict collector instead of trusting file
names, run:

```bash
embed-optim-evaluation-progress --watch-seconds 300
```

It atomically updates `logs/evaluation/live-audit.json`, records family/task/run coverage, and stops
only when all 1,680 provenance-valid units are present. A transient collector error is written into
the snapshot and retried rather than being mistaken for zero coverage.

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

### 5. Analyze weight-space trajectories

The NAACL follow-on can stream the 88 transformer hidden matrices from every checkpoint without
instantiating a model or loading the complete state into memory:

```bash
embed-optim-geometry \
  --run-dir outputs/dense/muon-lr1e-4 \
  --reference /path/to/pinned/DenseOn-unsupervised \
  --output-dir results/weight-space/dense/muon-lr1e-4
```

The command reuses the exact parameter-routing predicate used during training and refuses a
checkpoint whose hidden/auxiliary tensor counts differ from `completed.json`. It records exact
Frobenius, row-balance, column-balance, Gini, and energy-concentration statistics plus a deterministic
rank-64 randomized singular-spectrum sketch. Full SVD is used automatically when `--sketch-rank`
reaches a tensor's smaller dimension; set it to zero for an inexpensive exact-statistics-only pass.
Each checkpoint is committed as an atomic JSONL file, hashes and settings are captured in
`manifest.json`, and an identical rerun resumes by skipping finished checkpoints.

For a prespecified high-resolution tier, `--steps` selects exact saved steps and
`--tensor-regex` selects named matrices. These options make full-SVD layer/depth probes practical;
they are mutually exclusive with `--max-checkpoints`. In a selected sequence,
`delta_from_previous` is relative to the previous selected checkpoint.

After all runs finish, strictly validate and aggregate the records:

```bash
embed-optim-summarize-geometry \
  --geometry-root results/weight-space \
  --output-dir reports/weight-space \
  --verify-inputs
```

This requires the exact run set in `configs/experiment.yaml`, verifies every model/input digest when
requested, always verifies every JSONL digest and finite tensor metric, and writes a 120-row
`checkpoint_trajectory.csv`, a 24-row `run_trajectory_summary.csv`, and a content-addressed summary
manifest. Use `--allow-partial` only for exploratory work before all runs are available.
The checked-in exact-statistics tables and their interpretation notes are under
[reports/weight-space](reports/weight-space/README.md).

With `--reference`, records include displacement from the pinned pretrained model; checkpoints after
the first also include displacement from the preceding saved checkpoint. These are trajectory
displacements, not optimizer steps. Claims about actual AdamW/Muon update directions still require
the common-state gradient and virtual-update experiments specified in
[docs/naacl-paper-plan.md](docs/naacl-paper-plan.md).

### 6. Prepare and analyze fixed representation probes

Materialize the frozen training-distribution probe with:

```bash
embed-optim-prepare-probe --spec configs/representation_probe.json
```

The frozen specification selects 1,024 complete query/positive/seven-negative groups from the
canonical 500K dataset. It balances the seven source datasets (146 or 147 groups per source), ranks
samples within each source with a seeded BLAKE2b-128 digest, and publishes the expected source,
selection, selected-ID, Dataset-fingerprint, and final-manifest digests. The command builds into a
temporary directory and verifies every expected value before atomically publishing the probe. Its
`selection.jsonl` contains IDs and selection ranks but no text; the materialized Dataset retains the
exact text and negative order required by both model families.

This canonical probe is a **training-distribution mechanism probe**: all 1,024 groups came from the
500K examples seen during the one-epoch study. It may support trajectory, margin, and token-usage
diagnostics, but it is not held-out evidence.

The separate unseen probe protocol is checked in as
`configs/beir_representation_probe.json`. It selects 16 eval queries from each of the 14 pinned
decontaminated BEIR tasks without consuming a checkpoint score, ranking, or embedding. For every
query it chooses one deterministic highest-qrel positive, then seven non-relevant positives from a
24-query same-task pool using IDF-weighted lexical overlap and a seeded digest tie break. This yields
224 positive-plus-seven-negative groups that are held out from the 500K training view and balanced
across tasks. Qrels for the current query are always excluded, so a document relevant to multiple
queries cannot silently become a false negative.

The one-time derivation used an explicit `--allow-unfrozen` flag before the expected hashes existed.
The checked-in specification now freezes the audited manifest, selection, sample-ID, and Dataset
fingerprints. Reproduce it in a fresh output directory with:

```bash
embed-optim-prepare-beir-probe \
  --spec configs/beir_representation_probe.json \
  --output data/probes/decontaminated-beir-224-reproduction
```

The frozen rerun reproduced the manifest, selection ledger, Arrow data, Dataset metadata, and state
files byte-for-byte. An independent qrel audit checked 224 positives and all 1,568 negatives against
the pinned raw judgments. The specification records that 98 of 1,680 retrieval units and partial
scores existed when the protocol was written; it is a prospective completion lock, not a claim of
preregistration before outcome inspection. The two pre-output candidate-pool amendments are also
recorded, and its selection rule remains outcome-independent.

Encode the same probe with a selected checkpoint:

```bash
embed-optim-export-probe \
  --checkpoint outputs/dense/muon-lr1e-4/checkpoint-782 \
  --probe data/probes/training-1024-seed1729 \
  --probe-spec configs/representation_probe.json \
  --family dense \
  --output results/representation-space/exports/dense-muon-lr1e-4-step782.npz
```

The exporter applies the exact training-side DenseOn query/document prefixes or PyLate's
query/document encoding and document skiplist, preserves positive-first candidate order, normalizes
embeddings, and stores LateOn token arrays as packed vectors plus integer offsets instead of padding
every document to the longest item in the probe. On the frozen 1,024-row probe this reduces the
tokenizer-side document payload estimate from 7.19 GiB to 0.29 GiB per checkpoint (25.2x before
container overhead); the exact encoded size is recorded per export. The adjacent
`.npz.manifest.json` hashes every checkpoint JSON/safetensors input, the frozen probe spec
and manifest, and the final archive; it also records array shapes/dtypes, package/CUDA versions,
context length, prompts, query-expansion state, device, and GPU. The default archive is uncompressed
to avoid wasting CPU on nearly incompressible fp16 vectors. `--allow-unfrozen-probe` is available for
exploratory inputs, but such outputs retain their own hashes and are not part of the frozen formal
tier.

`embed-optim-analyze-probe` turns a versioned embedding export into a provenance-checked JSON report.
The `.npz` contract is deliberately model-independent so the same fixed sample IDs can be encoded by
the pretrained model and any selected checkpoint:

- `sample_ids`: unique `[samples]` identifiers;
- `sample_groups`: optional pickle-free string/integer source labels for per-group ranking metrics;
- `query_embeddings`: `[samples, dim]` for DenseOn or packed `[query_tokens_total, dim]` for
  LateOn;
- `document_embeddings`: `[samples, candidates, dim]` for DenseOn or
  packed `[document_tokens_total, dim]` for LateOn;
- `query_offsets` and `document_offsets`: required for new LateOn exports and strictly increasing;
  the analyzer also accepts legacy padded arrays with `query_mask` and `document_mask`;
- `reference_scores`: optional `[samples, candidates]` scores from a declared reference checkpoint.

Candidate index zero is always the positive; the remaining candidates are the seven explicit hard
negatives. For the unseen BEIR probe they are fixed lexical cross-query negatives rather than the
training set's mined hard negatives; tables and captions must preserve that distinction. Run:

```bash
embed-optim-analyze-probe \
  --input probes/dense/muon-lr1e-4-checkpoint-782.npz \
  --output results/representation-space/dense/muon-lr1e-4-checkpoint-782.json \
  --family dense \
  --require-export-manifest \
  --reference-input results/representation-space/exports/dense-pretrained.npz \
  --label dense/muon-lr1e-4/checkpoint-782
```

For both families, the report includes positive–hardest-negative margins, MRR, candidate-score
dispersion, representation covariance effective rank, stable rank, leading-variance concentration,
mean-vector anisotropy, and optional ranking drift from `reference_scores`. The LateOn path computes
the training recipe's MeanMaxSim and additionally reports query-token evidence entropy/Gini,
positive-document token coverage, and repeated MaxSim-token dominance. Large representation sets are
subsampled with a recorded deterministic seed; scoring itself is exact and batchable. The input
SHA-256, every array shape/dtype, metric settings, and positive-index convention are written with the
result. This command analyzes exported representations; the checkpoint encoder and probe-selection
manifest remain separate so sample selection cannot be silently changed during metric computation.
When the adjacent export sidecar is present it is always validated; `--require-export-manifest`
makes its absence a hard error for the canonical analysis tier.
`--reference-input` requires the exact same ordered sample IDs, verifies that both sidecars bind the
same probe manifest/selection, and adds score RMS drift, top-1 agreement, and top-k overlap relative
to the pretrained or other declared reference export.

Run the complete pretrained-plus-120-checkpoint representation matrix after retrieval evaluation
releases the GPUs:

```bash
embed-optim-probe-matrix \
  --matrix configs/experiment.yaml \
  --output-root results/representation-space/training \
  --gpus 0,1,2,3,4,5,6,7
```

The dispatcher binds every job to both the selected probe-manifest hash and frozen-spec hash. A
stale export from the other tier is treated as incomplete and recomputed; a mismatched
`--probe`/`--probe-spec` pair is rejected before any GPU worker launches.

Run the same pretrained-plus-checkpoint matrix on the frozen unseen tier with a disjoint output and
log root so no training-probe archive can be mistaken for held-out evidence:

```bash
embed-optim-probe-matrix \
  --matrix configs/experiment.yaml \
  --probe data/probes/decontaminated-beir-224-seed4242 \
  --probe-spec configs/beir_representation_probe.json \
  --output-root results/representation-space/decontaminated-beir \
  --log-dir logs/representation-space/decontaminated-beir \
  --gpus 0,1,2,3,4,5,6,7
```

The dispatcher resolves each pinned pretrained Hugging Face snapshot, runs the two reference exports
before their dependent checkpoints, assigns one checkpoint per GPU, and retries interrupted jobs.
Every export and metric report is content-hash audited before it is skipped on resume. Use
`--dry-run` to list only incomplete jobs without downloading a model or starting a GPU process.

After a tier reaches all 122 jobs, aggregate it without hand-editing JSON:

```bash
embed-optim-summarize-probes \
  --matrix configs/experiment.yaml \
  --result-root results/representation-space/decontaminated-beir \
  --probe data/probes/decontaminated-beir-224-seed4242 \
  --probe-spec configs/beir_representation_probe.json
```

The strict summarizer rehashes every export and metric sidecar, requires exactly two pretrained plus
120 checkpoint jobs, verifies the eight-candidate and per-group sample contracts, and refuses
unexpected JSON files. It writes `checkpoint_metrics.csv`, long-form
`representation_metrics.csv`, per-source/task `group_metrics.csv`, and a hashed
`summary_manifest.json`. `--allow-partial` is available only for explicitly labeled diagnostics;
the resulting manifest records `complete: false` and cannot support final-paper claims.

After both formal tiers pass strict aggregation, render their shared dynamics panel with:

```bash
embed-optim-plot-representation-dynamics
embed-optim-plot-late-token-dynamics
```

The plotter rehashes both complete 122-job summaries, their three source tables, and both frozen
probe identities. It then plots positive–hardest-negative margin, query effective rank, and top-1
agreement with the pretrained ranking for DenseOn/LateOn on both the training-distribution and
unseen-BEIR probes. Lines are medians and bands are interquartile ranges over all four learning-rate
runs; the shared pretrained state is shown explicitly at training fraction zero. The deterministic
SVG and its content-hashed manifest are written under `reports/representation-space/`.
The second command uses the same strict inputs to render LateOn-specific query-token evidence
entropy/Gini, document-token coverage, and repeated-token dominance. These are kept in a separate
panel so architecture-specific MaxSim evidence is not visually conflated with the shared Dense/Late
metrics.

Once BEIR coverage is also complete, join the three analysis spaces without manual spreadsheet
matching:

```bash
embed-optim-build-mechanism-bridge
```

This command requires the verified 120-checkpoint weight summary, both complete 122-job
representation summaries, and strict 1,680/1,680 BEIR coverage. It emits one joined checkpoint table,
96 within-run first-difference rows, and prespecified descriptive Spearman correlations at both the
family and optimizer-within-family levels. The manifest explicitly marks these one-seed checkpoint
associations as observational; common-state and short-branch interventions remain necessary for a
causal claim.

After the bridge and all three strict figures exist, render the fixed mechanism section into the
final Markdown blog:

```bash
embed-optim-render-mechanism-report
```

The renderer rehashes the 20-anchor common-state summary, 360 exact spectra, 120-checkpoint bridge,
and each figure sidecar before replacing the blog's mechanism markers. Its tables use frozen
aggregations—ten anchors per family/operator, four learning rates at the final stage, and seven
prespecified within-run associations—so it cannot silently select favorable layers, learning rates,
or correlations after seeing the results.

### Query-disjoint recipe validation

The exploratory sweep must not choose a winning learning rate from the same BEIR labels used for
the headline result. [`validation_probe.json`](configs/validation_probe.json) therefore freezes
4,096 queries from the unused part of the pinned 1.22M-query source. The materializer first
recomputes the canonical 500K training ledger and excludes every training query ID within its
source split; it then selects 585 queries per source and 586 from FEVER, with one positive and seven
newly sampled negatives per query.

Prepare or independently audit the validation data with:

```bash
embed-optim-prepare-validation
embed-optim-prepare-validation --audit-only
```

The strict audit rehashes all 16 Arrow/state files, replays the 500K exclusion ledger, verifies zero
query overlap, and checks all 4,096 positive-first groups for seven distinct non-positive negatives.
After the main BEIR jobs release the GPUs, evaluate only the final checkpoint of each exploratory
run and select one rate per optimizer and family:

```bash
embed-optim-validation-matrix --gpus 0,1,2,3,4,5,6,7
embed-optim-validation-matrix --audit-only --verify-hashes
embed-optim-summarize-validation
```

The 24 validation jobs retain per-example and per-source loss, margin, reciprocal-rank, and top-1
records. Selection minimizes mean eight-way contrastive loss, then breaks ties by higher positive
margin and lower hidden learning rate. `reports/recipe-validation/recipe_selection.json` is the only
allowed source of learning rates for the later confirmatory seeds; BEIR scores are not an input.

### Confirmatory seeds

[`confirmatory_protocol.json`](configs/confirmatory_protocol.json) prospectively freezes three new
seeds (`314159`, `271828`, and `161803`) before any query-disjoint validation or confirmatory model
output existed. Every view preserves the exact 500,000 query IDs, positive IDs/text, sample IDs, and
source quotas from seed 42. Only the seven-of-ten negative draw and Trainer/data seed change; seed 42
remains exploratory and is not counted as confirmatory evidence.

Materialize all three views in one pinned-source scan, or independently re-audit them with:

```bash
embed-optim-prepare-confirmatory-data
embed-optim-prepare-confirmatory-data --audit-only
embed-optim-prepare-confirmatory-data --audit-only --verify-source
```

The materialization command always reconstructs the cached pool from the pinned score source before
writing its formal receipt and proves that the original seed-42 indices reproduce the frozen source
ledger. The view audit requires exact
query/positive text identity, seven distinct non-positive negatives, unchanged quotas, content
hashes and Dataset fingerprints, and at least 98% changed negative groups for every seed pair.
Audit-only mode is read-only and normally trusts the content-addressed pool cache; adding
`--verify-source` makes that audit repeat the expensive raw score-table reconstruction.

After `embed-optim-summarize-validation` produces the six non-BEIR-selected recipes, generate three
family-specific matrices (six runs per seed, 18 runs total):

```bash
embed-optim-generate-confirmatory-matrices
embed-optim-generate-confirmatory-matrices --audit-only

embed-optim-matrix --matrix configs/generated/confirmatory/seed314159.yaml
embed-optim-matrix --matrix configs/generated/confirmatory/seed271828.yaml
embed-optim-matrix --matrix configs/generated/confirmatory/seed161803.yaml
```

The generated matrices are bound to the validation-selection digest and each data-view manifest.
They retain five training checkpoints for artifact/resume auditing, but only the final checkpoint is
part of the confirmatory BEIR claim: `18 × 14 = 252` formal retrieval units. This yields seed-level
uncertainty without reusing the discovery seed or selecting a learning rate on BEIR test outcomes.

After the 18 validation-selected runs complete, evaluate and summarize only their final checkpoints:

```bash
embed-optim-evaluate-confirmatory --gpus-a 0,1,2,3 --gpus-b 4,5,6,7
embed-optim-evaluate-confirmatory --audit-only
embed-optim-summarize-confirmatory
```

The confirmatory report requires all 252 seed/run/task units and reports all six frozen
family-by-optimizer contrasts with deterministic two-level seed×task bootstrap intervals. It emits
both nominal 95% intervals and Bonferroni familywise 95% intervals across those six comparisons;
only the latter govern positive, negative, or inconclusive headline language. Aggregate MTEB result
JSON does not contain per-query rankings, so this table explicitly does not claim query-level
inference.

### Shared-checkpoint scale-matched short branch

The local virtual-step intervention is complemented by one accumulated-trajectory control frozen in
[`short_branch_protocol.json`](configs/short_branch_protocol.json). It selects an exact proportional
50,000-group subset from the original data and starts both families from the fixed 60% AdamW
checkpoint at step 2,345. Prepare or audit the shared subset with:

```bash
embed-optim-short-branch --subset-only
embed-optim-short-branch --subset-only --audit-only
```

After the formal common-state matrix exists, run `embed-optim-short-branch` to derive each
family/operator hidden learning rate from the raw update norms at that shared state. The fixed rule
targets global hidden `||ΔW||F / ||W||F = 5e-4`; AdamW uses the same hidden/auxiliary routing as
Muon, and auxiliary AdamW remains `3e-6`. The command emits three explicit matrices with six runs
each:

```bash
embed-optim-short-branch
embed-optim-short-branch --audit-only

embed-optim-matrix --matrix configs/generated/short-branch/seed314159.yaml
embed-optim-matrix --matrix configs/generated/short-branch/seed271828.yaml
embed-optim-matrix --matrix configs/generated/short-branch/seed161803.yaml

# After all 18 branches finish, score every retained checkpoint on both frozen probes.
embed-optim-short-branch-evaluate --gpus 0,1,2,3,4,5,6,7
embed-optim-short-branch-evaluate --audit-only
embed-optim-summarize-short-branch
```

All five branch checkpoints are evaluated on the frozen query-disjoint validation and 224-query
unseen functional probes, not another full-corpus BEIR sweep. This 18-run branch tests whether an
immediate scale-matched directional effect accumulates under three different example orders. The
strict summary joins all 90 checkpoints across both probes and emits paired seed-level dynamics for
loss, margin, reciprocal rank, top-1 accuracy, pretrained ranking drift, representation rank, and
LateOn token utilization.

### Hybrid AdamW routing control

The prospectively locked NAACL fairness control is separate from the 24-run discovery matrix:

```bash
embed-optim-matrix \
  --matrix configs/hybrid_adamw.yaml \
  --log-dir logs/hybrid-adamw

embed-optim-evaluate \
  --matrix configs/hybrid_adamw.yaml \
  --stages 5 \
  --results-root results/hybrid-adamw-beir \
  --log-dir logs/hybrid-adamw-evaluation

embed-optim-summarize-hybrid-control
```

It contains eight runs: both model families crossed with the four original AdamW learning rates.
Every parameter still uses AdamW, but the 88 hidden matrices receive the swept rate while auxiliary
parameters receive the same fixed `3e-6` AdamW recipe used by Muon/NorMuon. This isolates parameter
routing from the matrix update rule. Only the final checkpoint is assigned formal BEIR evaluation
(8 runs × 14 tasks = 112 units); the original matrix remains the source of five-stage dynamics.
[`configs/hybrid_adamw_control.json`](configs/hybrid_adamw_control.json) records that 140/1,680
discovery evaluations and the completed weight trajectories were visible when this protocol was
frozen, so it is a prospective completion lock rather than a preregistration claim.
The summarizer independently deep-audits all 40 control checkpoints, requires the complete native
AdamW five-stage source plus exactly 112 hybrid final-stage units, and emits paired task, aggregate,
and system tables under `reports/hybrid-adamw/` with a content-hashed manifest.

### 7. Compare common-state optimizer updates

The checkpoint trajectories above cannot isolate an optimizer rule because each completed run visits
different weights and gradients. The formal matrix is frozen before complete BEIR coverage: it uses
the pretrained state plus the 20%, 60%, and 100% checkpoints from the nominal interior-rate
`adamw-lr1e-5`, `muon-lr1e-3`, and `normuon-lr1e-3` trajectories in both families. These are
mechanism anchors, not configurations selected as BEIR winners. The spec transparently records that
98 of 1,680 strict BEIR units and partial scores had already been observed when this anchor grid was
frozen, so this is a prospective completion lock rather than a claim of preregistration before all
outcome inspection. It also records a pre-execution amendment from evaluation to training mode so
the declared gradient-checkpointing policy is operational; no common-state GPU artifact existed at
that amendment. Inspect the exact 20-job matrix without using a GPU:

```bash
embed-optim-common-state-matrix --dry-run
```

After retrieval evaluation releases the GPUs, run the resumable matrix with:

```bash
embed-optim-common-state-matrix \
  --matrix configs/experiment.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

Each worker first caches a fixed sequence of gradients without advancing its checkpoint, then runs
the update analyzer on the same visible GPU. The final content audit can be repeated independently:

```bash
embed-optim-common-state-matrix --audit-only --verify-hashes
```

Once all 20 anchors pass that audit, turn the tensor JSONL files into strict, paper-ready tables:

```bash
embed-optim-summarize-common-state \
  --matrix configs/experiment.yaml \
  --result-root results/common-state \
  --output-dir reports/common-state
```

The summarizer rehashes every gradient shard, matched-update tensor file, update record, and
manifest. It requires the exact frozen anchor grid and an identical hidden-tensor signature across
both families. The outputs preserve raw per-tensor gradient and update metrics, pairwise direction
cosines, parameter-weighted anchor summaries, and Muon/NorMuon contrasts against AdamW at the same
weights and ordered gradient history. As with the representation summarizer, `--allow-partial`
produces a manifest explicitly marked incomplete and is only for diagnostics.

The rank-64 sketches above support broad layer coverage, but they do not preserve an exact singular
value curve. A separately frozen protocol selects six matrices by architecture alone: attention
input and MLP expansion projections at layers 0, 10, and 21. The ledger records that 110/1,680 BEIR
units and the exact-statistics weight trajectories were already visible, while no formal
common-state or representation output existed. It is therefore a prospective completion lock, not
a preregistration claim. Preview the exact 20-anchor, 360-spectrum tier with:

```bash
embed-optim-common-state-spectra --dry-run
```

After the common-state matrix is complete, run it across the available GPUs:

```bash
embed-optim-common-state-spectra \
  --matrix configs/experiment.yaml \
  --common-state-root results/common-state \
  --output-root results/common-state-spectra \
  --gpus 0,1,2,3,4,5,6,7
```

Each job computes exact float32 `torch.linalg.svdvals` from the float16, per-tensor
Frobenius-matched intervention directions. The matrix command rehashes all common-state inputs,
resumes only matching outputs, and automatically writes `spectrum_metrics.csv`, a long-form
`singular_values.csv` with Frobenius/spectral normalization and cumulative energy, and a strict
summary manifest. Reaggregate without GPU work using `--summarize-only`; add `--audit-only
--verify-hashes` to validate the per-anchor tier itself.

Render the deterministic 12-panel publication figure only from a complete summary:

```bash
embed-optim-plot-common-state-spectra \
  --summary-dir results/common-state-spectra/summary \
  --output reports/common-state/exact-update-spectra.svg
```

Each panel shows the median and interquartile range over all ten frozen anchors in one model family,
not a visually selected checkpoint. Rows separate DenseOn/LateOn and attention/MLP matrices;
columns are layers 0, 10, and 21. The plotter rehashes its source CSV and frozen protocol, requires
all 360 spectra, and writes a content-addressed sidecar next to the SVG.

For a single ad hoc checkpoint, the equivalent first stage is:

```bash
embed-optim-export-gradients \
  --checkpoint outputs/dense/muon-lr1e-3/checkpoint-2345 \
  --probe data/probes/training-1024-seed1729 \
  --probe-spec configs/representation_probe.json \
  --common-state-spec configs/common_state_probe.json \
  --family dense \
  --train-mode \
  --output-dir results/common-state/dense/muon-lr1e-3/checkpoint-2345/gradients
```

The frozen specification selects 32 probe examples with a seeded, source-balanced round robin and
forms eight ordered four-example gradients using micro-batches of one. Like formal training, model
parameters and accumulated gradients remain float32 while forward operations use bfloat16 autocast.
The model remains in training mode so non-reentrant gradient checkpointing is actually active;
dropout is deterministic because every shard resets its recorded RNG seed, and each cached gradient
is shared by all three optimizer replays. Every gradient is computed at identical weights, clipped
at the training threshold of 1.0 across all model parameters, and then saves only the exact
hidden-matrix partition used by Muon. The runtime-to-safetensors name mapping is required to be a
complete one-to-one match and is recorded explicitly. The checkpoint, probe, selection, clipping
factor, loss, tensor partition, runtime, and every shard are content-hashed. A partially completed
export resumes only after validating every committed shard.

Replay that shared gradient history through the three optimizer state machines with:

```bash
embed-optim-analyze-updates \
  --checkpoint outputs/dense/muon-lr1e-3/checkpoint-2345 \
  --gradient-manifest \
    results/common-state/dense/muon-lr1e-3/checkpoint-2345/gradients/manifest.json \
  --common-state-spec configs/common_state_probe.json \
  --operator-device cuda \
  --storage-dtype float16 \
  --output-dir results/common-state/dense/muon-lr1e-3/checkpoint-2345/updates
```

This advances AdamW's coordinate moments, Muon's momentum, and NorMuon's momentum plus row-wise
second moment while holding the parameters fixed. It reports the final raw update spectra, row/column
balance, energy concentration, gradient/weight angles, and optimizer-pair direction cosines. Weight
decay is excluded and labeled as such. It also exports one intervention direction per optimizer whose
Frobenius norm is matched to the corresponding layer's weight norm; multiplying every tensor by the
same `alpha` therefore gives all optimizers the same per-layer update-to-weight budget. Numerical
tests replay multiple gradients and compare all three directions directly with the optimizer used in
formal training. The frozen analyzer executes Newton–Schulz on CUDA, matching the formal bfloat16
kernel backend rather than substituting a CPU matrix product. This protocol isolates the stateful
transform; it does not claim that the cached fixed-weight gradients reproduce an optimizer's native
training trajectory.

### 7b. Audit optimizer basis sensitivity

The frozen [`basis_sensitivity.json`](configs/basis_sensitivity.json) protocol reuses the 20
common-state gradient histories to test a function-preserving ModernBERT attention symmetry. It
applies identical seeded SO(2) rotations to query and key coordinates within each split-half RoPE
plane, leaves value coordinates unchanged, and inverse-maps every optimizer direction before
comparison. Arbitrary orthogonal head rotations are not used because they do not generally commute
with RoPE.

Run the 540 full-tensor comparisons and 3,240 selected-head comparisons after the common-state grid
is complete, then rehash every input and output independently:

```bash
embed-optim-basis-sensitivity \
  --protocol configs/basis_sensitivity.json \
  --device cuda:0
embed-optim-basis-sensitivity \
  --protocol configs/basis_sensitivity.json \
  --audit-only \
  --verify-inputs
```

The analysis covers layers 0, 10, and 21; heads 0, 5, and 11; three rotation seeds; both model
families; and AdamW, Muon, and NorMuon. Before replay, a float64 calibration verifies attention-logit
invariance at both model rope bases and multiple position pairs. The outputs report inverse-mapped
direction cosine and error, norm preservation, predicted-descent preservation, and Q/K head
singular-spectrum drift. This isolates coordinate dependence in the actual optimizer implementation;
it is not a retrieval-quality intervention.

### 8. Run scale-matched functional interventions

Update geometry is only mechanistically useful if it changes the retriever's function. The frozen
[`functional_intervention.json`](configs/functional_intervention.json) protocol applies each
common-state AdamW, Muon, and NorMuon direction at the *same checkpoint* and with the *same
per-layer relative Frobenius budget*. It uses three fixed descent scales (`1e-4`, `3e-4`, `1e-3`)
and a sign-reversed `1e-3` control. The baseline plus 12 conditions are scored on the separately
frozen 224-query decontaminated-BEIR probe, which was never used to construct the gradients.

Preview or run the 20-anchor matrix after the common-state jobs finish:

```bash
embed-optim-functional-intervention-matrix --dry-run
embed-optim-functional-intervention-matrix \
  --matrix configs/experiment.yaml \
  --gpus 0,1,2,3,4,5,6,7
```

Then perform the independent content audit and paired aggregation:

```bash
embed-optim-functional-intervention-matrix --audit-only --verify-hashes
embed-optim-summarize-functional-interventions
```

Each anchor writes 2,912 paired sample records and 13 condition summaries; the formal aggregate
therefore requires exactly 58,240 sample records. It reports change from the unmodified checkpoint
and Muon/NorMuon-versus-AdamW contrasts at the same scale. The metrics are contrastive loss,
positive and hardest-negative scores, positive margin, reciprocal rank, and top-1 accuracy under the
exact positive-first eight-way DenseOn/LateOn scorer. Every checkpoint, common-state direction,
probe, protocol, record table, and summary is content-hashed. This is a local fixed-weight causal
intervention—not evidence that one virtual step reproduces a complete native optimizer trajectory;
the NAACL plan retains a short shared-checkpoint branch for that stronger claim.

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for experiment-change and reporting requirements,
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations, and [SECURITY.md](SECURITY.md)
for private vulnerability reporting.

## Provenance and license

This repository started from LightOn's open
[`mdenseon-mlateon`](https://github.com/lightonai/mdenseon-mlateon) training/evaluation code and keeps
its Apache-2.0 license. The study additionally implements the optimizer experiment, deterministic data
materialization, explicit no-in-batch losses, checkpoint matrix, pinned decontaminated tasks, and
reporting pipeline.

Please cite the original Muon work, the DenseOn/LateOn paper, and NorMuon when using this study;
complete citation metadata is in [CITATION.cff](CITATION.cff), with linked references in
[docs/blog.md](docs/blog.md).
