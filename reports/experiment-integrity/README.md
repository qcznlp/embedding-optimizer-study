# Experiment integrity audit

Audit observations began at 2026-09-04 19:49 UTC. The host's local clock is UTC+8.
Audited experiment checkout: `f231a6430712388778f32ad1736a4cb6de3bec3e`.
Paper and analysis repairs are isolated on `narrative/weight-space-spine`.

## Verdict

**New limitation, September 5 06:10 UTC:** the actual accumulated Trainer path fails a four-rank
CPU reference check. It produces one-quarter gradients through duplicate normalization. A paired
diagnostic compensation passes, without changing production code. See the
[source-bound diagnosis](../engineering-archive/dense-ddp-accumulation-v1/README.md). This supersedes
any broad inference that earlier unit/payload/CPU-loss passes certified the full training stack.
The matrix must not supply paper conclusions until this is resolved. Pausing the still-running
jobs has been requested, not performed; all checkpoints remain preserved.

The stored-data and checkpoint portions passed the earlier checks documented below. The whole study is
**incomplete**, and this report does not certify final retrieval or mechanism claims. Four AdamW,
four Muon and NorMuon 1e-4/3e-4 are complete; NorMuon 1e-3 and 3e-3 are underway. A new ten-run
audit now repeats the full stored-row and deep checkpoint checks, and an independent HF audit
verifies every uploaded file for those ten runs. The earlier eight-run receipts remain unchanged.
The complete
840-unit primary BEIR result grid and final analysis/publication manifests do not yet exist.

The machine-readable [primary audit](primary-audit.json) records
`audited_portion_valid=true` and `scientific_completion=false`. Its exact executed source is
preserved in [audit-source.py](audit-source.py). Later formatting of the reusable audit command did
not replace those executed bytes.

## Evidence and coverage

### Independent HF download and offline checkpoint restoration

The [fresh-directory restoration receipt](checkpoint-download-normuon-3126.json) covers a forced,
anonymous HF download of NorMuon 3e-4 at step 3126 from immutable commit
`e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae`. All 18 files / 1,351,464,827 bytes match the trusted
ten-run audit by SHA-256, Git-blob digest and size before model/optimizer deserialization. The
downloaded run configuration exactly matches the frozen matrix. Offline CPU encoding produces
two finite 768D unit vectors; all 134 parameters and optimizer states continue exactly after
serialization across two fixed synthetic-gradient updates. The schedule matches and all downloaded
payloads remain unchanged. No original live checkpoint is read by this diagnostic.

The new source is `scripts/audit_checkpoint_download.py`, reusing the already verified
`exercise_resume` implementation. It rejects untrusted audit bytes, wrong revisions/namespaces,
unsafe paths, symlinks, missing/extra files and corrupted payloads before deserialization. The
[restoration guide](../../docs/checkpoint-restoration.md) documents exact commands and source
availability. This is a fresh download on the same physical host, not a second-host, rank-RNG,
data-loader, four-GPU, BF16, full-context or scientific optimizer result.

The first status-reader attempt used an unavailable W&B timestamp attribute; re-polling the same
two exact handles with supported fields confirmed both running at 03:24 UTC. The first download
command combined mutually exclusive local/cache directory options and was rejected before any
download; the supported local-directory-only command then succeeded. Neither diagnostic error
indicates a training failure or changed any live controller. These details are engineering-only.

### Ten-run milestone: data, checkpoints and geometry

The [new primary audit](primary-audit-ten-run.json) verifies all 500,000 materialized row identities,
with zero mismatches, duplicate sample IDs or invalid negative groups, and all ten complete runs /
50 checkpoints. All 89 frozen bindings match. Deep payload checks include finite model and optimizer
tensors, optimizer-state topology, scheduler/trainer metadata and the expected rank RNG payloads.
This does not numerically resume every checkpoint or establish distributed runtime equivalence.

The [new HF content audit](huggingface-digest-audit-ten-run.json) compares all **1,010 files /
82,373,100,558 bytes** with the ten original immutable upload commits. Missing, extra, size and
digest mismatch sets are empty, and local payloads are stable. The two new original whole-run
commits are `2ba6b3100c163a9f4e31aa6dea49af867987dee4` (NorMuon 1e-4) and
`e0c85c8d8d4e22c76faf660b0fdcd477f37df0ae` (NorMuon 3e-4). Original upload receipts were not edited.

The original frozen CPU geometry entry point also completed both new NorMuon trajectories under
the existing partial-analysis runbook. Coverage is now **50 stages / 4,400 hidden-matrix rows**;
102 unique model/metadata inputs and all 60 outputs match their manifests and frozen sources.
The [ten-run geometry archive](primary-geometry-ten-run-archive.json) verifies **60 files /
10,146,027 bytes** at immutable commit `489c606076b0f8a0ef82adc7b85f1fb29da794bc`.
It adds the 12 missing NorMuon files / 2,030,743 bytes and preserves all 48 prior geometry files.
The [anonymous source proof](primary-geometry-normuon-public-source-proof.json) checked all ten
underlying NorMuon weights at their original public commits. Exact additions-only audit/upload
and public-proof sources are retained under `reports/engineering-archive/geometry-increment-v2/`.
No numerical source changed and no old HF file/card/history was deleted or overwritten.

These are raw weight/displacement measurements only. Full subspace overlaps, retrieval prediction,
functional dimension results and crossed continuation remain pending; no optimizer-quality verdict
follows from the new archival coverage. The strict paper gate must remain unsatisfied until actual
scientific outputs exist.

### Earlier eight-run geometry and public archival

The original frozen CPU geometry entry point now covers all five stages of Muon 1e-3 and 3e-3,
adding ten new checkpoint analyses. Raw coverage is 40 stages from all four AdamW and four Muon
rates, not the new ninth completed NorMuon run. A separate read-only/content audit checks all
48 files, 3,520 hidden-matrix rows and 82 unique checkpoint/reference model and metadata inputs.
Every record matches its manifest; all 88 hidden matrices are present at each stage and all
numerical values are finite. Numerical source and parent protocol hashes match their original lock.

The first upload request was rejected before execution over possible disclosure of private
checkpoint information. The [anonymous public-source proof](primary-geometry-public-source-proof.json)
then checked all 20 underlying Muon weights at their original immutable public commits. Both existing
project repositories are public. After re-review, the same bounded additions-only upload proceeded:
24 missing Muon files / 4,053,859 bytes were added, with the existing 24 AdamW files preserved.

The [verified archive receipt](primary-geometry-eight-run-archive.json) matches all 48 files /
8,115,284 bytes at commit `5aba281fd92fa9e6f8336423646cdb17acbfff17`, with no missing, extra, size
or digest mismatches. Exact audit/upload and public-check sources are retained under
`reports/engineering-archive/geometry-increment-v1/`. This adds only raw weight/displacement
features. It does not produce all-rate subspace overlaps, retrieval prediction, functional dimension
results or an optimizer-quality claim. No old HF artifact/card/history was overwritten or deleted;
the separately rejected erroneous-result withdrawal remains unexecuted.

### Original eight-run integrity audit

| Requirement | Observation | Evidence strength |
| --- | --- | --- |
| One deterministic 500k-query training view | All 500,000 materialized row identities match the canonical sampling ledger | Full stored-row comparison |
| Seven hard negatives per query | Zero repeated negative IDs within a row and zero positive/negative ID overlap | Full stored-row comparison; upstream texts were not re-downloaded |
| Shared training data across runs | All eight completions use training-view fingerprint `cc0598ffd4f5454f` | Dataset audit plus completed-run receipts |
| Four configurations per optimizer | 12 declared runs, four rates each; all AdamW and Muon cells complete | Frozen matrix and observed artifacts |
| Five retained checkpoints | 40 completed checkpoints at steps 782, 1563, 2345, 3126, 3907 | Deep payload audit; planned total is 60 |
| Recoverable training state | Model, optimizer, scheduler, trainer state and four rank RNG payloads validate | Payload validation; this audit did not resume every checkpoint numerically |
| Frozen implementation | 89 current source/parent/configuration bindings match | Byte count and SHA-256 checks |
| Remote durability | 808 files, 67,658,898,978 bytes match eight immutable HF commits | Full local hashing versus remote LFS SHA-256/Git-blob SHA-1 |
| W&B provenance | 10 visible runs valid: eight finished, two running; zero identity/configuration problems | Fresh read-only API audit |
| Complete primary retrieval outcomes | Not yet available | Missing complete source-bound manifest |
| Complete weight/representation explanation | Not yet available | Final dimension, retrieval-bridge and factorial results remain required |
| Submission-ready paper | Not yet available | Active generated result includes still contain explicit pending markers |

The training row ledger SHA-256 is
`735ef35b7195f3dae3172496b5bc534d39f2b7594d216c685eaebb37134fc347`.
Source quotas match exactly: FEVER 52,489; FiQA 2,629; HotpotQA 40,630; MSMARCO 240,405;
NQ 72,725; SQuADv2 62,244; Trivia 28,878. There are 500,000 unique sample IDs and zero
materialized-row identity mismatches.

The [HF content audit](huggingface-digest-audit.json) retains both inventories for every completed
run. All eight have empty missing, extra, size-mismatch and digest-mismatch sets. This is stronger
than the previous whole-run upload receipt, which checked paths and sizes. Original controller
receipts were preserved. Intermediate checkpoint receipts already included digest checks.

## Implementation review

### Main recovery migration and consistent successor

The [prepared recovery runbook](../engineering-archive/main-resume-v1/README.md) records a real
post-training controller defect: resume clears prior steps, repeats completed work and overwrites
old attempt logs. The archived loop reproduces it; the isolated candidate preserves the completed
prefix, attempts and orphan logs and rejects altered records. All 17 commands and arguments remain
unchanged. This affects interrupted post-training work; it is not evidence of corruption in the
currently running training jobs, whose main ledger has executed zero post-training steps.

The new [exact migration protocol](../engineering-archive/main-resume-v1/migration-protocol.json)
binds a separate implementation for the combined seven-file target `4380a363...`. Its
[25 dedicated checks](main-handoff-recovery-migration-tests.xml) execute the actual migration on
small temporary copies, preserving the source ledger byte-for-byte and all eight original backups.
Read-only preflight leaves no files, held leases prevent mutation, both simulated archive/write
interruption points recover, and stale sources or changed migrated evidence are rejected. The old
JSON-only migration cannot apply this Python source change and was not relaxed or reused.

The [new successor projection](main-recovery-successor-projection-v2.json) reconstructs three
prepared dependency layers and exercises all five real protocol loaders, with all 47 commands and
arguments unchanged. The new local/projected identities are `ca0c7f5b...` / `6749baa1...`; it requires
complete main `4380a363...`, rejecting the previous six-file target and incomplete parents. The
prior source bytes are archived at their original receipt digests. The first direct-script command
failed on package import; the successful module invocation is the v2 projection, not a changed
scientific implementation.

The [focused suite](main-recovery-successor-focused-tests.xml) passes 81 checks and the
[full suite](main-recovery-successor-full-tests.xml) passes 1,124 (zero failures/errors/skips/
exclusions). The [actual controller dry run](main-recovery-successor-dry-run.json) starts no work
because main is incomplete. The [strict paper gate](main-recovery-successor-paper-gate.json) still
fails for missing real primary/dimension publications and three pending scientific includes.
No live source, ledger, controller, GPU allocation or HF object was changed by this repair. This
is engineering evidence only and must not enter the manuscript.

### Real Dense loss, gradient and evaluation-encoding check

The [combined validation record](dense-loss-contract-validation.json) binds the exact sources,
real-checkpoint output, test receipts, new-stage HF inventories and unchanged original upload receipts.
It also verifies that diagnostic tolerances were not changed after the initial attempt.

The [CPU computation receipt](dense-loss-contract-real.json), observed at 01:38 UTC on September 5,
extends the earlier serialization-only audit with actual forward/backward computation. It loads
AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4 checkpoint-2345 through the real corrected Dense loader,
uses the production collator and the installed trainer's feature extraction, and verifies exactly
nine ordered columns with the same explicit query/document token prefixes and saved cosine setting.

For each checkpoint, two short synthetic queries each have one positive and seven own negatives.
The actual loss and all 134 trainable parameter-gradient tensors agree with an independent row-wise
float64 cosine/log-sum-exp reference under tolerances declared before execution. The real MTEB
encoder wrapper, using forced small length-budget chunks, reproduces the normalized training-path
embeddings and cosine scores in original input order. The largest score difference is below
1.2e-7 and the largest gradient-element difference is below 2.4e-9. These are numerical-equivalence
checks on this fixture, not comparisons of optimizer quality. Every source checkpoint inventory is
unchanged; no optimizer step or production evaluation is executed.

Run the check with the actual experiment source path, offline and without GPU visibility:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=/root/embedding-optimizer-study/src /usr/bin/python3 \
  /root/embedding-optimizer-story-refactor/scripts/audit_dense_loss_contract.py \
  --repository /root/embedding-optimizer-study \
  --output /tmp/dense-loss-contract-new-audit.json
```

Choose a new output filename; existing receipts are never overwritten. This is CPU float32 and
evaluation mode with autograd, using short synthetic texts. It does **not** certify training-mode
dropout, distributed gradient accumulation, BF16/FlashAttention equivalence, an 8192-token forward
pass, full-corpus retrieval or the full scientific experiment. The configured maximum length is
checked, but maximum-length execution is not. Do not put this engineering audit in the manuscript.

The [12 focused tests](dense-loss-contract-tests-v2.xml) include rejection of accidental in-batch
candidates and an intentionally wrong backward pass with an unchanged loss value. The
[full suite](dense-loss-contract-full-tests.xml) passes all 1,081 tests, zero failures/errors/skips/
exclusions, in 146.211 seconds; the later documentation/contract selection passes 13 checks.
The diagnostic's first real attempt stopped before loss evaluation
because its own token-comparison check treated a tensor as a scalar boolean. That diagnostic-only
source is retained under `reports/engineering-archive/dense-loss-check/`; the tensor/list regression
was added without changing any training/evaluation source or tolerance.

The separate [new-stage HF audit](normuon-3126-hf-digest-audit.json) independently verifies both
NorMuon checkpoint-3126 uploads against original immutable commits, including every local content
digest. All 34 files / 2,702,927,254 bytes match; original upload receipts are retained byte-for-byte.
This advances independently audited durability beyond the prior checkpoint-2345 pair, not run
completion. No HF object, card or history was changed, and cleanup remains awaiting scope approval.

### Real evaluation handoff preflight and source-topology repair

The [handoff receipt](evaluation-handoff-validation.json) records a real, offline, CPU-only dry run
of the live corrected BEIR entrypoint on the eight completed primary runs and all their five stages.
It passes the training-data, deep-checkpoint and formal-runtime checks and records 40 checkpoint
content identities, the ten actual evaluator sources and a 560-unit plan. The three generated JSON
manifests are retained under `beir-completed-subset-preflight/`. No evaluator workers or task scores
were produced, and the production result directory was not modified.

The next consumer gate exposes a live defect: `corrected_outcome_summary.build_report` calls
`collect_evaluations`, whose runtime reader does not recognize the corrected producer's ten-source
mapping. It rejects the actual dry-run manifest before reading any score. This is a targeted gate
reproduction, not an executed 840-unit outcome report. The isolated repair adds exactly that mapping;
every declared source must still match a reachable Git blob by its path, byte count and SHA-256.
Historical/general mappings are unchanged, and partial, mixed, unknown and changed-source manifests
remain rejected. The real 40-checkpoint manifest rehashes successfully and the repaired runtime
reader accepts its actual source identities without rewriting any metadata.

The [focused selection](evaluation-handoff-tests-v3.xml) passes 65 tests. The
[full suite](evaluation-handoff-full-tests.xml) passes 1,052 tests, zero failures/errors/skips and
no exclusions, in 147.571 seconds. The final
[source checks](evaluation-handoff-final-source-tests.xml) pass 22 tests after a comment-only
clarification. Two added regression cases verify that a partial input manifest can grow without
rebinding existing checkpoint bytes, and a failed extension leaves the original manifest unchanged.
These tests do not establish GPU scheduling safety or scientific outcomes.

The helper is a transitive dependency absent from the current frozen explicit source lists. Bind it
and perform a controlled, exact non-scientific transition before deployment; the earlier
publication-header-only proposal does not cover this newly found blocker. Live code, controller
ledgers and training remain unchanged. The last-run early-evaluation runbook also now warns that
the main controller's all-eight-GPU validation can race manual subset workers: metadata locks are
not GPU leases. A verified single-owner handoff is required; none was performed here.

The later [combined main repair](../engineering-archive/main-handoff-v1/README.md) prepares the
exact six-file deployment patch: source-verifier recognition, explicit outcome binding, dependent
protocol identities and the existing three-header fix. Its proposed main target is `af646ecf...`.
[Forty focused checks](main-handoff-repair-tests.xml) pass, preserving all scientific fields and
rehearsing the actual migration on a temporary copy with all original backup records intact. These
checks do not replace the earlier 1,052-test full suite or establish a real runtime transition.
The later [successor update](../engineering-archive/successor-handoff-v1/README.md) now consistently
binds the combined main target and refreshes the story, dimension and factorial-publication parent
identities. Its [read-only projection](combined-successor-contract-projection.json) reconstructs both
exact amendment layers, verifies seven byte-identical predecessor protocols and invokes all five
actual consumer loaders. The isolated `a64b2658...` and host-projected `37a4485c...` contracts differ
only through four checkout-local command arguments. All 47 commands remain unchanged, including
the original 36-step factorial sequence and the exact fully completed main gate.

The [first focused attempt](combined-successor-focused-tests.xml) records 54 passing checks and one
failure: a stale factorial-publication dimension-amendment binding. After that binding was refreshed
and explicitly tested, the [focused suite](combined-successor-focused-tests-v2.xml) passes all 56.
The [full suite](combined-successor-full-tests.xml) passes all 1,071 tests without failures, errors,
skips or exclusions, in 145.892 seconds. Rehashed scientific/configuration drift and weaker, incomplete,
old or header-only main gates remain rejected. These are preparation checks, not runtime migration
or scientific results. Live sources are untouched and the old receipts remain historical.
The [combined validation receipt](combined-successor-validation.json) binds exact executed sources,
the tests, both projected contracts, the real read-only controller plan and the strict paper gate.
That plan reports main not ready and starts no controller; strict paper release still exits 1 on
missing primary/dimension publications and pending result includes. A later 31-test documentation
and contract regression also passes. The supervisor reports 48/60 remote checkpoint coverage at
01:23 UTC; this is not a new independent full-payload audit.

### Real checkpoint optimizer-state continuation on CPU

The [CPU continuation receipt](checkpoint-cpu-resume.json), observed at 2026-09-05 00:21 UTC,
SHA-256 `4d0fd9f4e9f9cda67deb7aad4adc11e18200389bf1a6b8cd392fb510faebae59`, uses three real
primary checkpoint-2345 states: `padded-adamw-3e-5`, `padded-muon-3e-4`, and
`padded-normuon-3e-4`. Each model reloads 134 float32 CPU parameters and all 134 optimizer states.
All 51 source checkpoint files are content-addressed; source stat inventories remain unchanged.

The saved optimizer and scheduler reload exactly. After one fixed synthetic-gradient update, the
model, optimizer and scheduler are serialized in memory and reloaded into a fresh model/builder.
The next identical gradient update produces exactly equal weights, optimizer state and scheduler
state in the uninterrupted CPU branch and the reloaded CPU branch. All 134 parameters change at
both diagnostic steps for each optimizer. Learning rates at steps 2345, 2346 and 2347 match the
declared 391-step warmup and 3907-step linear schedule, including the auxiliary AdamW groups.

These are **engineering restoration checks, not scientific optimizer results**. The gradients are
explicitly synthetic, and no training forward/backward, data-loader order, rank RNG restoration,
CPU/GPU numerical equivalence or four-GPU training resume was tested. Only three checkpoints were
exercised; do not generalize this to every saved checkpoint or put it in the scientific narrative.
Parameter order follows the frozen model/optimizer builder; the original checkpoint format lacks
an independent per-slot name ledger. All source checkpoints, live controllers and HF files remain
unchanged. The receipt records PyTorch 2.9.1+cu129, Transformers 5.3.0, SentenceTransformers 5.7.0,
Safetensors 0.8.0, two CPU threads and the exact implementation identities.

The new diagnostic and existing optimizer reference checks pass 15 focused tests in
[checkpoint-cpu-resume-tests-v2.xml](checkpoint-cpu-resume-tests-v2.xml). The last whole-project
suite remains the earlier 1,040-test run; this addition does not claim a new full-suite count.
To repeat this isolated, offline check, use a new receipt path:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=/root/embedding-optimizer-study/src \
  /usr/bin/python3 /root/embedding-optimizer-story-refactor/scripts/audit_checkpoint_optimizer_resume.py \
  --repository /root/embedding-optimizer-study \
  --output /tmp/new-checkpoint-cpu-resume.json
```

### Training and evaluation implementation

The Dense objective normalizes query and document embeddings and forms exactly eight logits per
query: its positive plus its own seven negatives, at temperature 0.02. Other queries' documents do
not enter that denominator. Loss tests explicitly change another query's group to check this
boundary. Training and evaluation use the `query: ` and `document: ` prefixes.

The optimizer tests compare multi-step AdamW behavior to PyTorch AdamW, Muon to PyTorch Muon, and
NorMuon updates and optimizer state to the pinned reference equations. Embedding tables and
one-dimensional parameters follow the declared auxiliary AdamW routing. All eight complete runs
have the same partition: 88 hidden matrices (110,297,088 parameters), one auxiliary decay tensor
(38,682,624 parameters), and 45 auxiliary non-decay tensors (34,560 parameters).

Completed-run receipts identify the actual accelerators as NVIDIA L20Z. The manuscript now names
that hardware explicitly instead of describing it only as H100-equivalent.

An independent [checkpoint precision audit](checkpoint-precision-audit.json) reads all 40 completed
primary safetensors headers: each contains 134 FP32 tensors. The bound training source loads FP32
parameters and enables BF16 autocast, consistent with these saved parameter types. This checks
parameter/storage precision; it does not independently resume every optimizer or rule out every
possible mixed-precision numerical effect.

The training length is 8,192 for queries and documents, consistent with Table 11 of the
[DenseOn paper](https://arxiv.org/pdf/2607.27178). The shorter value on the model card is not the
SFT context specification used by this study. Saved checkpoints retain the configured length.
Training histories also show the declared warmup and decay: the learning-rate multiplier at logged
step 10 is 9/391 for every completed run, followed by linear decay across the 3,907-step horizon.

The primary BEIR wrapper pins all 14 released task revisions, deeply validates completed training
inputs, and records checkpoint content identities before result-cache reuse. The score aggregator
requires all 12 × 5 × 14 observations; a missing or duplicate cell cannot silently become a complete
primary result. These are implementation and protocol checks. End-to-end completion must still be
verified after the actual result grid is produced.

## Repairs made during this audit

### Complete-result publication checks

The native-vector dimension figure retains all 68 plotted means in a recomputed CSV and generates
its exact PGFPlots coordinates from the fixed all-rate/task/mask panels. Rehashing modified plot
values or coordinates does not bypass reconstruction. Synthetic values exist only in temporary test
papers and are explicitly marked; the manuscript's real result includes remain pending.

A full synthetic-results compile exposed malformed table-header newlines in the publication
renderer. The [complete live-source scan](live-publication-syntax-diagnosis-v2.json) finds three
affected headers on main (two scientific tables and one engineering-only sensitivity table). The
initial startswith-only scan found none because Python joins adjacent string literals; it is
superseded by this full decoded-string scan. The two tables retained by the rewritten manuscript
are repaired in isolation. The factorial appendix table was also widened to two columns after the
same test detected overflow. No scientific values or inference rules were changed.

The [minimal live patch](live-publication-header-hotfix.patch) and
[proposed transition](live-publication-header-migration-draft.json) are **undeployed drafts**.
They do not authorize an unchecked runtime migration. In particular, the old instruction to simply
wait for main release before any publication change is no longer sufficient: its renderer needs
this narrow repair, and its successor controller's exact main-contract dependency must be migrated
consistently. Training/metric code and live ledgers remain unchanged.

The later [offline rehearsal](live-publication-header-rehearsal.json) passes
[nine focused tests](live-publication-header-rehearsal-tests-v2.xml) using exact archived live
renderer, publication protocol, main ledger and successor gate bytes. It reconstructs the existing
two-file patch exactly; only three renderer bytes change. The real old renderer produces the
expected `Misplaced noalign` compilation error, while the patched three-table output compiles from
synthetic evidence. Rendered text differs only in the three header separators. This is a syntax
check, not a full-layout certification of the old main manuscript.

On a temporary ledger copy, the real migration implementation preserves the original archive
byte-for-byte, all eight backup records, prior migration records and completion state. A second
call is idempotent, and undeclared command, source or contract changes fail. The projected main
contract is `6d991052ebafcfd2e6069e1ae4864986e38066b8ef9105a1fc058b5672acdff0`. The unchanged
factorial dependency rejects it, as intended; a proposed gate changing only that exact identity
accepts a synthetic fully complete ledger, but still rejects missing runs, unfinished steps or
missing backups. That dependency edit is a test fixture, not a deployed or frozen new protocol.
The existing 47-step plan still pins the old main contract and must be updated and re-audited before
deployment. Neither actual controller lease was acquired, and both live ledgers remain unchanged.
These nine checks supplement the earlier 1,005-test suite; no combined 1,014-test run is claimed.

The ordinary `make -C paper` no longer invokes the obsolete historical headline writer; it compiles
only the currently declared inputs. Release runs the strict paper audit after PDF/layout checks.
The primary publication now has an independent read-only audit that reloads its source-bound
upstream evidence and reconstructs exact text, coverage and conclusion. Even a populated manuscript
with rehashed edited values cannot pass that gate. The retained broader engineering audit has not
been relaxed or removed.

The earlier figure-only full suite passes 996 tests in
`dimension-figure-full-tests-v2.xml`. The prior `dimension-figure-full-tests.xml` preserves an
invalid test invocation that imported the installed live checkout rather than this worktree and
failed collection; it is not an implementation pass. Always set the absolute worktree `src` path in
`PYTHONPATH` for this shared environment. The later final-suite receipt below includes the added
primary-publication and build checks; use its actual outcome rather than the older count.

The final suite passes **1,005 tests**, zero failures/errors/skips and no exclusions, in
`dimension-figure-final-tests-v2.xml`; the focused release/source audit passes 43 tests in
`primary-release-focused-tests.xml`. The earlier final-suite attempt preserves two obsolete test
expectations for the historical headline writer. Those tests now verify the actual strict
post-build source check and separation from historical generation; neither safety assertion was
removed. The [validation receipt](dimension-figure-validation.json) binds the exact tested source
and all observations. All 50 bindings across the six updated publication/handoff protocols match,
and every checked-in JSON configuration parses.

The actual draft and the complete synthetic-result paper both end main text on page 6. The dimension
figure is on page 5; appendix result tables follow the main boundary. The actual draft has no Type 3
fonts. Strict CLI release remains nonzero, with primary and dimension publications missing and all
three result includes visibly pending. The unchanged 47-step dry-run contract is now
`3873d6cc2e00f2e78d76378a14868548e4929c88347657c3cb20478696ec6316`; it starts no controller and
does not migrate the still-incomplete live main or factorial ledgers.

HF cleanup is a separate, pending owner-approval operation. Its read-only inventory and exact target
plan are in `reports/hf-obsolete-cleanup/`; safety review rejected execution before any mutation.
No removed HF file or history, newly verified post-cleanup archive, or completed deletion may be
claimed. Local historical evidence remains available for engineering diagnosis and equivalence
checks, never as primary scientific results.

### Prepared successor parent and deployment-path preflight

The [parent amendment](../../configs/dense_no_packing_state_operator_parent_publication_amendment.json)
now makes the isolated 47-step successor require the exact repaired main identity `6d991052...`.
The only prepared protocol changes are that parent hash, the source-bound amendment reference and
its amendment timestamp. All twelve-run, exact required-step and backup gates remain intact.

The [read-only projection](successor-parent-contract-projection.json) preserves the actual live
`/usr/bin/python3` setting. After explicitly relocating four checkout-local path arguments, every
one of the original 36 commands and all other arguments exactly match the archived zero-step live
ledger. The local dry-run contract is `9b577964...`; the projected experiment-host contract is
`eada94f3450d8af5d387fea2d6fdb29182c7c8bebc6adb5e9325396da0cd9b0a`. Do not confuse these
with the running `6605090d...` contract or assume that projection is deployment.

The original focused comparison failed because it compared absolute paths in different worktrees.
`successor-parent-focused-tests-v2.xml` preserves that failure; the first full-suite invocation was
interrupted and is not a pass. The final checks explicitly verify all four permitted relocations,
reject an undeclared path change and retain the original command equality assertion. The full
suite passes **1,040 tests**, with zero failures, errors, skips or exclusions, in
[successor-parent-full-tests-v2.xml](successor-parent-full-tests-v2.xml). The focused suite passes
37 tests, and post-documentation checks pass 30. The [validation receipt](successor-parent-validation.json)
binds 36 current source/configuration files and seven evidence files; the prior 1,005-test receipt
remains historical and must not be treated as the identity of the updated completion protocol.

The actual CLI dry run starts no controller. The current strict paper audit exits 1 because the
real primary and dimension publications remain absent and scientific includes remain pending.
Neither live controller lease was acquired and neither live ledger or source was changed. The
narrow main transition, complete main gate, approved source deployment and controlled zero-step
successor handoff are still required. Recompute the contract on the experiment checkout after
deployment; this preflight does not grant source-publication or migration approval.

### Withdrawal dependencies and new intermediate backups

The [read-only HF dependency audit](../hf-obsolete-cleanup/history-dependencies.json), observed
at 23:45 UTC, verifies 49 primary backup receipts at 39 readable immutable commits. Their referenced
files retain the same metadata digests at the observed current heads. It also identifies 52 model
LFS objects shared across withdrawn and retained paths, including 225 current checkpoint uses, and
one shared untrained baseline vector object. This is a concrete reason not to permanently purge
objects by old filename prefix. It is not a whole-history object inventory or cleanup approval.
The later second NorMuon checkpoint-2345 backup is outside that snapshot; refresh dependencies
before any approved rewrite. No HF file, card, history or live backup receipt was changed.

By 23:47 UTC both first-pair NorMuon checkpoint-2345 stages are remotely sealed, raising coverage
to 46/60 stages while complete runs remain 8/12. Separate CPU-only local rehashes compared all 34
files, totaling 2,702,894,592 bytes, to their original immutable upload commits with no missing,
extra, size or digest mismatch. Original receipts remain under
`reports/dense-no-packing/incremental-checkpoint-backup/`. Neither this durability observation nor
the zero-fatal-marker artifact snapshot certifies complete training or a scientific finding.

The new dependency checks plus existing cleanup checks pass 38 focused tests. The broader
dependency/backup/supervisor/header regression selection passes 65 tests with no failures, errors
or skips. Those runs supplement, not replace, the earlier 1,005-test full suite. The source-bound
dependency report and new uploader receipts remain in the isolated worktree; deployment, source
publication and irreversible deletion approvals are still pending.

### Earlier data, export and publication-integrity work

These changes are confined to the isolated paper/analysis checkout:

- Dimension export validation now checks finite embeddings, unique sample IDs, the ordered sample
  digest, frozen probe identities, task counts, and matching sample/task assignments across states.
- The dimension publisher checks exact run/stage/task/draw or rotation identities. Matching the
  total row count is insufficient; duplicate cells, changed optimizer labels, rates or stages fail.
- Primary retrieval rows consumed by the dimension bridge must cover every run/stage exactly and
  agree with the matrix metadata.
- The final paper gate rejects unresolved result includes; synthetic completion tests separately
  exercise this gate. Publication/migration fingerprints were refreshed for the new manuscript.
- The paper build imports the current checkout's source, avoiding accidental use of an installed
  package from another worktree.

Seventeen dimension integrity tests pass, including adversarial inputs whose file digest was
recomputed after corruption. Ninety-five focused manuscript, publication, dimension and factorial
tests pass. The initial full main-checkout test command was interrupted while its historical
full-model paper audit was still running, with no failed assertion reported. Its rerun excludes
only `test_current_dense_paper_constants_match_strict_sources` and passes all remaining 883 tests
with zero failures, errors or skips. [main-tests.xml](main-tests.xml) preserves that test output.
The current primary checkpoint audit above is independent of the excluded historical paper test.

After those repairs, the **entire isolated-checkout suite passes: 904 tests, zero failures,
errors or skips, with no test exclusions**. [refactor-full-tests.xml](refactor-full-tests.xml)
preserves the final run. Passing unit and integration tests is not a substitute for the pending
primary end-to-end result grid.

The subsequent primary-dimension handoff implementation passes a fresh full suite of **928 tests**,
again with zero failures, errors, skips or exclusions; see
[dimension-handoff-full-tests.xml](dimension-handoff-full-tests.xml). Its
[41 focused tests](primary-dimension-probe-tests.xml) exercise the export wrapper, changed model and
probe identities, changed execution settings, interruption recovery, recomputation of numerical
outputs and decisions, and compilation of synthetic generated LaTeX. The
[actual read-only plan](primary-dimension-plan.json) lists all 61 states but correctly reports the
four missing training completions. No primary embedding inference was launched by these checks.

The final automatic-handoff and portable-publication changes pass a fresh full suite of
**936 tests**, with zero failures, errors, skips or exclusions; see
[dimension-portable-full-tests.xml](dimension-portable-full-tests.xml). Tests preserve the original
36 factorial commands exactly while inserting nine primary/dimension steps, enforce the incomplete
main-ledger wait, and run the real publication recomputation in an offline renamed clone without
checkpoints, training data or raw embedding vectors. Corrupt clone files cannot be masked by the
original producer directory. The [actual controller dry run](dimension-factorial-plan.json) reports
45 planned steps, an incomplete main ledger and `controller_started=false`. This contract is not
deployed. Portable auditing verifies the small evidence closure and recomputes statistics and exact
LaTeX; it explicitly does not rerun encoding, coordinate ablations or checkpoint validation.
The [source-bound validation receipt](dimension-portable-validation.json) records the tested files,
test command, unchanged live controller contracts and remaining gates. The
[actual strict paper audit](dimension-portable-paper-gate.json) exits nonzero as required: the
primary dimension report is absent and all three active result includes still contain pending
markers. No release gate was relaxed to obtain a pass.

The later raw-vector archival addition passes **955 tests**, with zero failures, errors, skips or
exclusions; see [dimension-archive-full-tests.xml](dimension-archive-full-tests.xml). Its
[47-step dry run](dimension-archive-factorial-plan.json) inserts eleven steps while retaining the
original 36 factorial commands, and starts no controller before main completion. The archive tests
exercise the real publication/closure selector with synthetic inputs plus an immutable simulated HF
repository. They cover staged-file changes, mixed LFS/Git digests, corrupt remote contents,
conflicting prefixes, original-commit pinning after remote-head changes, lost-receipt recovery and
offline download integrity. A read-only live API check confirms the dataset is public and the new
primary archive prefix is absent. No primary analysis or upload has executed. Download integrity
does not repeat feature computations; the later replay addition below addresses that separate
capability. The archived [21:26 training snapshot](dimension-archive-progress.json) reports
eight complete runs and 42 resumable stages with no fatal error markers.
The [source-bound archive validation receipt](dimension-archive-validation.json) records the exact
tested source/configuration bytes, real dry-run contract and immutable destination observation.

The subsequent offline raw-feature replay implementation passes **965 tests**, with zero failures,
errors, skips or exclusions; see [dimension-replay-full-tests.xml](dimension-replay-full-tests.xml).
A renamed, network-disabled cold subprocess reconstructs all 61 synthetic states (224 paired
queries, 14 tasks, 8D vectors for test speed) without importing PyTorch or the dataset library.
It checks the trusted download identity, exact archived source, canonical input roles and every
output cell; corrupted numerical findings fail even after their file digests are updated.
Reconstructed evidence has a separate scope and cannot replace primary authoring inputs.

Separately, a full CPU recomputation of all 61 real historical 768D states matches every CSV cell
and attribution array exactly: 61 checkpoint rows, 854 task rows, 68,320 random-removal rows and
546 rotation rows, all with maximum numerical error zero. The
[final equivalence receipt](dimension-replay-historical-equivalence-v2.json) binds the old and new
manifests plus the archived pre-refactor implementation. The rotation guard now enforces the
protocol's existing score **and rank** invariance requirement; a tiny score change that flips a rank
is rejected. All [39 historical endpoint/pretrained rotation pairs](dimension-historical-rotation-ranks.json)
pass, with zero rank changes and maximum score error below 6.67e-16. This validates method equivalence
on those inputs, not primary optimizer findings or a completed primary archive replay.

The [47-step dry run](dimension-replay-factorial-plan.json) still starts no controller and leaves the
live main and factorial contracts unchanged. The
[22:18 training snapshot](dimension-replay-progress-final.json) reports eight completed runs and
44 resumable stages, with no fatal log markers. Both first-pair NorMuon checkpoint-1563 receipts
now report immutable, digest-verified HF uploads. Actual primary export, analysis, archive and
reconstruction remain pending. The [replay validation receipt](dimension-replay-validation.json)
records the exact tested implementation and evidence boundaries.

The older historical dimension manifest did not contain an implementation identity and did not
pass the current source-bound audit. Its original bytes were preserved. A CPU-only regeneration
from the same fixed embeddings now passes `--audit-only`: 60 historical checkpoints plus the
pretrained model, 854 task rows, 68,320 random-removal rows and 546 rotation rows. Four output files
are byte-identical to the originals. The 61-row checkpoint summary retains every existing cell
unchanged and adds eight query/document spectrum columns. The
[comparison receipt](historical-dimension-regeneration.json) identifies all these files and the
new manifest. This validates the historical method-development output, not a primary result or
an optimizer mechanism.

The ordinary PDF build and layout check pass: the current main-text endpoint is page 5, with the
declared result tables in the appendix. A Type 3 font in the new figure was replaced by embedded
TrueType fonts; the rebuilt PDF has no Type 3 fonts. This is a draft build, not evidence of
publication readiness.

## Repeatable checks

Use a new report path for each observation; preserve the original receipts. These commands read
the active training checkout without changing its source, training state, W&B or Hugging Face:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/embedding-optimizer-study/src \
  /usr/bin/python3 /root/embedding-optimizer-story-refactor/scripts/audit_primary_experiment.py \
  --repository /root/embedding-optimizer-study \
  --output /tmp/dense-primary-integrity-new.json

CUDA_VISIBLE_DEVICES='' PYTHONPATH=/root/embedding-optimizer-study/src \
  /usr/bin/python3 /root/embedding-optimizer-story-refactor/scripts/audit_hf_run_digests.py \
  --repository /root/embedding-optimizer-study \
  --training-audit /tmp/dense-primary-integrity-new.json \
  --output /tmp/dense-hf-integrity-new.json
```

The second command needs read access to the checkpoint archive and hashes all completed local
files; it does not upload. A newly completed run must have its original upload receipt before it
can pass that remote audit. The implementation used in the new receipt must not be confused with
the archived source used for the original observation.

## Scientific limitations and remaining work

1. Finish and deeply audit the four NorMuon runs, yielding the full 12-run/60-checkpoint matrix.
2. Complete and audit all 840 primary BEIR units and the independent validation selection.
3. Complete the weight-space summary and every predeclared held-out-dose retrieval bridge.
4. Finish the corrected dimension export/analysis handoff and connect it to final publication.
   The source-bound exporter and publisher recomputation gate now have synthetic integration and
   LaTeX compilation tests. Their 61-state read-only plan identifies the four still-incomplete
   NorMuon runs. Automatic post-main scheduling and portable final-paper recomputation are now
   implemented and tested in isolation. The controlled runtime transition, actual primary execution,
   remote archival and generation of the real evidence closure remain required; the chain has not
   run on the primary checkpoint grid.
5. Execute and audit the frozen crossed state-by-operator follow-up and its backups.
6. Render all findings from those outputs, pass the strict paper gates, and publish the reviewed
   repository state and final analysis artifacts. The isolated narrative branch is not yet merged.

The primary matrix uses one training seed. Its task bootstrap describes variation over the chosen
benchmark tasks, not variation across training seeds. The primary comparison averages the four
declared rates in each optimizer's own grid; the validation-selected recipe comparison is secondary.
Muon-family auxiliary parameters use AdamW at 3e-6, so these are optimizer recipes rather than a
pure causal isolation of the matrix transform. The crossed continuation experiment addresses a
bounded state-versus-continuation question; it cannot establish universal mediation of the full
training trajectory.

The controlled objective omits in-batch negatives as requested, and does not add the original
paper's MRL/KD terms. It should be described as controlled DenseOn adaptation. This study averages
all 14 released decontaminated tasks; the [LightOn post](https://huggingface.co/blog/lightonai/denseon-lateon)
uses a 12-task headline average excluding FEVER and ClimateFEVER, so those averages are not directly
comparable. Native-coordinate deletion is basis dependent and the 224-query dimension probe ranks
eight candidates; its metrics cannot replace full-corpus retrieval outcomes.

Implementation incidents remain engineering provenance in this report and other repository records.
They do not belong in the manuscript. No training process or `gpu.py` process was modified by this
audit.
