# Dimension utilization: protocol and preliminary result

## Why this analysis exists

The paper's central question is how AdamW and Muon change a pretrained retriever's weight state. A
useful intermediate map is therefore

`weight trajectory -> embedding dimensions -> corpus retrieval`.

[Takeshita et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1410/) show that random
removal of half the coordinates of several text embeddings often causes only a modest retrieval
loss. They also show why effective rank or isotropy alone is not an adequate explanation: deleting
some individual coordinates improves downstream performance, and these degrading coordinates can
be distributed throughout the representation.

Our source-bound protocol is `configs/dense_dimension_utilization_protocol.json`. It was frozen
before corrected BEIR or corrected dimension-utilization outputs existed. Historical packed
checkpoints are explicitly exploratory; the corrected independently padded grid is the prospective
comparison.

## Primary export and analysis handoff

The primary export implementation is locked separately in
`configs/dense_primary_dimension_export_protocol.json`. The command
`python -m embed_optim.primary_dimension_probe --dry-run` lists the exact pretrained plus 60
checkpoint cells and reports missing training completions without initializing GPU work or
creating output files. The executed exporter requires the complete 12-run deep training audit,
uses the pinned pretrained revision and frozen 224-query probe, and acquires one cooperative GPU
lease. It does not change training or inspect unrelated GPU processes.

Exports retain float32 vectors from the formal BF16 model instead of adding float16 archive
quantization; the ablations continue to use float64 calculations. Every export binds its complete
model/configuration inventory, training completion, probe, encoding settings and observed model
loading. Cache reuse checks these identities again. Changed or untagged artifacts are preserved
and rejected, not overwritten. A completed NPZ/manifest pair can be sealed after an interruption
only when its matching receipt records the verified model loading.

The post-main command order is now locked in
`configs/dense_primary_dimension_handoff_protocol.json`. The augmented factorial controller adds
eleven primary-publication/dimension steps before calibration, retaining the original 36 commands
and their relative order. Its 47-step plan is read-only with
`python -m embed_optim.state_operator_factorial_completion --dry-run`. The existing live controller
is still on its original contract; the zero-step migration has **not** been executed.

Deployment prerequisite: the combined evaluation-source and publication-header repair under
`reports/engineering-archive/main-handoff-v1/` must be deployed through its exact controlled main
transition. Do not assume the existing main contract can finish its release unchanged, use the
earlier header-only proposal as a complete repair, or deploy the entire narrative branch during
training. No runtime migration has been performed.

The successor's parent update is now prepared in
`configs/dense_no_packing_state_operator_handoff_repairs_amendment.json`. It requires repaired
main contract `af646ecf...`, not the still-live `4152531e...`, and preserves all twelve-run, step
order and backup requirements. The original live factorial remains on `6605090d...` and must stay
untouched until the narrow main repair and its full completion are verified.

`python -m scripts.audit_successor_contract --repository /root/embedding-optimizer-story-refactor`
validates both preparation layers and all five actual consumer protocol loaders, then projects only four checkout-local path arguments onto the
experiment root. It preserves the original live `/usr/bin/python3` setting and all 36 original
commands. The observed isolated contract is `a64b2658...`; the exact projected experiment-host
contract is `37a4485c...`. These are different by design, because four commands contain absolute
checkout paths. Neither is the currently running controller. The full receipt is
`reports/experiment-integrity/combined-successor-contract-projection.json`. The prior header-only
amendment and its receipts remain historical. Recompute on the actual
experiment checkout after approved source deployment before restarting the zero-step successor;
do not substitute the isolated dry-run hash for the deployed hash.

After the live main completion controller has finished and the controlled zero-step transition is
installed, the augmented controller runs the following sequence automatically. These commands
document the manual equivalent; do not launch a second sequence alongside the controller:

```bash
PYTHONPATH=src /usr/bin/python3 -m embed_optim.primary_dimension_probe --dry-run
PYTHONPATH=src /usr/bin/python3 -m embed_optim.primary_dimension_probe --gpu 0
PYTHONPATH=src /usr/bin/python3 -m embed_optim.primary_dimension_probe --audit-only

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_utilization --analysis-scope corrected \
  --input-root results/dense-primary-dimension-probe/exports/dense \
  --output-dir reports/dimension-utilization-primary

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_publication
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_publication --audit-only

CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_release
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_publication --portable-audit

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_archive
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src \
  /usr/bin/python3 -m embed_optim.dimension_archive --audit-only
```

The analysis now requires `primary_exports.json`; renaming historical NPZ files cannot satisfy the
primary input gate. The publisher re-audits its upstream inputs and recomputes contrasts, bridge
folds, decisions and exact LaTeX, so updating only an output digest cannot validate edited findings.
Tests exercise the real export wrapper with a synthetic encoder, source/cache mutations, numerical
recomputation and actual compilation of generated findings. These are implementation tests, not
model or retrieval results. The primary matrix has not yet traversed this chain end to end.

The publisher also implements the protocol's descriptive dimension panels. One four-panel native
vector figure shows final-stage random-removal retention and all five stages of helpful mass share,
helpful participation and degrading mass. Every optimizer curve uses all four learning rates and
14 equally weighted tasks; random removal additionally averages all 20 shared masks per fraction.
The pretrained curve or horizontal reference is shown throughout. No confidence band is implied,
no stage/rate is selected, and shortlist retention is not labeled full-corpus nDCG.

All 68 displayed means are retained in `figure_points.csv`. The exact PGFPlots coordinates and
caption are generated inside `paper/generated/dimension-utilization.tex`, and both the full-source
and portable publication audits recompute them. Rehashing an edited point table or a changed plot
coordinate cannot make it pass. The existing portable/HF selectors include these files automatically;
the 47-step handoff gains no new command. The full synthetic paper-layout test renders all three
result includes and a conspicuous synthetic-data label; its numbers are never copied into the paper.

The portable release is generated only after the full checkpoint-backed authoring audit passes.
It closes the small source tables, exact publication outputs, source/configuration identities and
per-state export provenance. A clean clone can then recompute all publication statistics and the
exact LaTeX without model payloads, raw embedding exports, training data or network access. Tests
perform that audit in a renamed clone and show that a corrupt clone cannot fall back to a still-
valid producer directory. The paper's final gate now requires this closed, recomputable evidence.

This is deliberately a **publication-level** audit: it does not repeat model encoding, recompute
coordinate ablations from the raw vectors, or revalidate checkpoints. Those operations belong to
the full-source audit and the larger Hugging Face reconstruction archive. The release receipt
explicitly records `checkpoint_revalidated=false` and `embedding_features_recomputed=false`.

Still required before final release: deploy the controlled zero-step transition, execute the real
primary analysis, archive the new export/analysis evidence on Hugging Face, and regenerate the
paper/closure from those real results. No synthetic test output is an eligible scientific input.

## Durable raw-vector and evidence archive

`configs/dense_primary_dimension_archive_protocol.json` freezes the CPU/network-only archive.
The command `python -m embed_optim.dimension_archive --dry-run` validates that lock and reports
missing prerequisites without creating files or contacting Hugging Face. The actual uploader first
runs the full checkpoint-backed publication audit, then archives the small publication closure and
all 61 raw float32 vector exports. It does not upload a whole repository, model checkpoint trees,
raw training/probe text, cache directories or credentials.

The manifest's SHA-256 names a new prefix under
`qcz/embedding-optimizer-study-analysis-artifacts/primary-dimension/v1/`. Every file is copied to a
private temporary staging directory and checked before an atomic additions-only commit. A
conflicting prefix is rejected and preserved. Exact paths, sizes and LFS SHA-256/Git-blob SHA-1
are then compared at the returned immutable commit, not mutable `main`. Later audits keep that
commit even if another upload advances the repository. A lost receipt can recover a fully matching
immutable snapshot, explicitly labeled as recovery rather than claiming it is the original upload.

The local receipt is `reports/dimension-utilization-archive/ARCHIVE_SHA256.json`. To inspect a future
real archive on another machine, substitute its `commit_oid` and `archive_sha256` below:

```bash
hf download qcz/embedding-optimizer-study-analysis-artifacts --repo-type dataset \
  --revision VERIFIED_COMMIT_OID \
  --include 'primary-dimension/v1/ARCHIVE_SHA256/**' --local-dir /path/to/hf-download

python -m embed_optim.dimension_archive \
  --verify-download /path/to/hf-download/primary-dimension/v1/ARCHIVE_SHA256 \
  --expected-sha256 ARCHIVE_SHA256
```

The offline check requires the manifest hash from the trusted receipt and verifies the complete
downloaded file set without a network or producer-directory fallback. `payload/` retains
repository-relative paths. Preserve the downloaded version and restore into a new checkout for
analysis; do not overwrite an existing evidence tree. It includes enough raw vectors to perform
new coordinate analyses without encoding again. The existing primary authoring path still requires
its checkpoint/source locations; the independent replay command below does not. Do not mistake the
download check alone for re-encoding, ablation recomputation or scientific completion.

The transport tests use a simulated immutable HF repository and synthetic vectors, including the
real publication-statistics/closure selector. They exercise source mutation, corrupt remote digests,
conflicting prefixes, lost receipts, changing remote heads and offline download verification.
No actual primary archive has been uploaded yet because its source analysis is still pending.

## Recompute features after moving machines

The source-bound `dimension_replay` command recomputes covariance spectra, every native-coordinate
ablation, all shared random masks, and all three endpoint rotations from the verified vectors.
It compares all cells of the four CSV tables and all coordinate-attribution arrays against the
archived results. It never consults a still-existing producer directory or loads model/checkpoint
payloads. The original training and checkpoint-backup implementations remain unchanged.

After the trusted-hash download check, run the archived implementation as follows. Replace
`ARCHIVE_SHA256` using the immutable archive receipt; choose an output directory that does not exist.

```bash
dimension_snapshot=/path/to/hf-download/primary-dimension/v1/ARCHIVE_SHA256
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH="$dimension_snapshot/payload/src" \
  python -B -m embed_optim.dimension_replay \
  --archive-root "$dimension_snapshot" --expected-sha256 ARCHIVE_SHA256 \
  --output-dir /path/to/new-reconstruction
```

`-B` and `PYTHONDONTWRITEBYTECODE=1` prevent Python from adding bytecode caches inside the immutable
download. Do not relax the exact archive inventory if an accidental import created extra files;
use a fresh verified download. The command checks running implementation hashes and the canonical
file roles, not the old machine's absolute paths. A new directory outside the archive is required,
and neither successful nor failed reconstructions overwrite original evidence.

The receipt `replay.json` reports exact equality separately from tolerance-based agreement. Frozen
tolerances are `rtol=1e-9, atol=1e-12` for CSV numbers and `rtol=1e-6, atol=1e-12` for float32
attribution arrays; row identities, array metadata, shapes and dtypes must match exactly. A numeric
mismatch produces a nonzero CLI exit and retains the recomputed files for inspection. These
tolerances handle numerical-library variation and cannot be tuned after seeing a failed replay.

The reconstruction manifest has scope `reconstruction_dense_dimension_utilization`, never the
primary authoring scope. It explicitly records `scientific_completion=false`; it cannot replace
the full training/export gate or become a new primary publication input. The receipt also states
that model encoding, checkpoint validation and publication inference were not repeated. To repeat
publication inference from the separately closed original tables, use the portable-publication
command above; that is a different check from recomputing raw features.

Validation includes a renamed, network-disabled cold subprocess over a complete synthetic 61-state
panel (224 paired queries, 14 tasks, 8D vectors for test speed), with no PyTorch or dataset library
loaded. Separately, a full 61-state, 768D recomputation from real historical exports matches every
old CSV cell and attribution array exactly. Its receipt is
`reports/experiment-integrity/dimension-replay-historical-equivalence-v2.json`. This verifies numerical
refactor equivalence on those inputs, not primary findings or successful primary archive replay.

## What is measured

For each DenseOn checkpoint, the existing probe exports contain 224 queries, balanced across 14
decontaminated BEIR tasks, with one positive and seven fixed candidates per query. The analysis:

1. deletes each of the 768 native embedding coordinates separately;
2. recomputes cosine scores after re-normalization;
3. records the change in shortlist nDCG and positive--hardest-negative margin;
4. separates helpful and degrading attribution mass and computes their participation ratios;
5. removes 10%, 25%, 50%, and 75% of coordinates under 20 shared random masks; and
6. repeats endpoint attribution after three shared orthogonal rotations.

The rotation control is essential. Coordinate deletion is basis-dependent even though cosine
retrieval is invariant to a shared orthogonal rotation. A result that changes under rotation may
describe the model's native coordinate basis, but not coordinate-free use of representational
dimension.

Agreement across the three sampled rotations is evidence of robustness to those rotations, not a
proof of arbitrary-basis invariance. The crossed state-by-operator continuation does not manipulate
individual geometry features and cannot, by itself, establish one as a causal mediator.

The implementation enforces both parts of the frozen invariance check: score error no greater than
1e-6 and unchanged positive ranks. A within-tolerance score perturbation that changes a rank fails
the control; numerical tolerance is not permission to change the measured ranking.

## Preliminary historical result

The currently audited regeneration is
`reports/dimension-utilization-historical-validated/summary_manifest.json`; its exact implementation
and input/output hashes are recorded there. The older output directory is preserved as provenance,
but its manifest lacked the implementation identity now required by the audit. The regeneration
leaves all existing numerical cells unchanged and adds the checkpoint-level query/document spectrum
columns. See `reports/experiment-integrity/historical-dimension-regeneration.json` for the bytewise
and cellwise comparison. Neither version is primary-matrix evidence.

The first complete exploratory pass covers the pretrained model and all 60 historical Dense
checkpoints. It does **not** support the simple hypothesis that Muon's retrieval region comes from
using more embedding dimensions.

At the final stage, averaging all four rates per optimizer:

| Representation metric | AdamW | Muon | NorMuon |
|---|---:|---:|---:|
| Query entropy effective rank | 150.69 | 149.24 | 149.88 |
| Query stable rank | 29.12 | 30.78 | 31.53 |
| Document entropy effective rank | 167.59 | 155.52 | 156.88 |
| Document stable rank | 28.26 | 24.51 | 25.01 |
| Relative shortlist nDCG after random 50% removal | 0.9920 | 0.9936 | 0.9916 |

The rank definitions themselves disagree: Muon has slightly higher query stable rank but lower
query entropy rank, and substantially lower document rank under both measures. Randomly deleting
half the coordinates leaves all three optimizer families within roughly one percent of their
eight-document shortlist nDCG.

The more functional leave-one-coordinate-out margin analysis is also nearly indistinguishable. For
Muon minus AdamW at the final stage, averaging all four rates and bootstrapping the 14 tasks:

| Metric difference | Estimate | Exploratory task-bootstrap 95% interval |
|---|---:|---:|
| Degrading-coordinate fraction | -0.00184 | [-0.00939, +0.00609] |
| Degrading attribution mass | -0.00122 | [-0.00337, +0.00106] |
| Helpful mass share | -0.00396 | [-0.01271, +0.00414] |
| Helpful participation ratio | +0.00064 | [-0.00398, +0.00491] |

All four intervals include zero. After three common random rotations, the pooled Muon--AdamW
ordering remains negligible for degrading mass and helpful participation, while helpful mass share
slightly favors AdamW. These rotation results reinforce the absence of a simple Muon dimensionality
advantage on this probe.

## Interpretation boundary

This is currently a useful negative result, not the final answer. It says:

- Muon's larger hidden-weight displacement is not mirrored by an obvious increase in embedding
  effective rank;
- high random-deletion robustness is shared by all optimizer families;
- the tested coordinate-level utility statistics do not yet explain Muon's historical BEIR region;
  and
- “more dimensions are active” should not be used as the paper's mechanism without new evidence.

It does not say that dimension use is irrelevant. The probe ranks only eight candidates, so its
nDCG is coarse and cannot substitute for full-corpus BEIR. Corrected checkpoints may also differ
from the historical packed states. The prospective test will export the same fixed probe for all 60
corrected checkpoints, repeat the frozen analysis, and ask whether any dimension-utility feature
predicts full-corpus nDCG outside its learning-rate dose.

The strongest full-corpus extension, if time permits, is to cache embeddings for a small frozen
decontaminated retrieval suite and compute exact random-removal and single-coordinate attribution
without re-encoding. That extension must be fixed before inspecting its optimizer contrast.
