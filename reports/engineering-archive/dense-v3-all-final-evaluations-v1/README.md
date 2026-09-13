# All twelve final DenseOn checkpoint evaluations

At **2026-09-10 16:47:07 UTC** (reader-reported timestamp), every original v3
final checkpoint has a complete, verified fourteen-task BEIR readback. The
[full score grid](tables/summary.md) contains all four declared learning rates
for AdamW, Muon and NorMuon, each at step 3907. All **168 original task scores**
are in [task-scores.csv](tables/task-scores.csv).

Each macro score is the equally weighted mean over the same fourteen pinned
decontaminated full-corpus tasks, displayed as nDCG@10 multiplied by 100.
Rows follow optimizer/rate order; no best-performing cell is selected or hidden.
The complete endpoint grid is new empirical evidence. It does **not** complete
the 60-checkpoint experiment: the 48 intermediate checkpoints, validation-only
selection, functional analyses and crossed continuation remain required.

## Native evidence and independent reconstruction

The unchanged [native reader](source/readback.py) authenticates both original
56-file training assemblies, the existing dispatcher and authorization, each
run's deep-verified completion parent and the actual checkpoint payloads. It
then calls the original `inspect_evaluation` implementation and requires exact
agreement with each original `all-fourteen-tasks-verified.json` receipt.
All twelve native records pass, including all fourteen task identities, their
pinned revisions/splits/subsets and original workers' matching successful exits.
The actual CPU-only invocation terminated with exit zero; no CUDA context or new
model/retrieval computation was created by this readback.

The [complete original readback](actual/all-twelve-final-readback.json) is
2,537,171 bytes, SHA-256
`b8c9aff0f580e3fee1d8f239b2943ef649ec154ae5767ee576ee47b38e027fd6`.
It retains raw score/metadata snapshots and original native/operational receipts.
The six records already read in the [first complete cohort](../dense-v3-first-complete-evaluations-v1/README.md)
are overlapping evidence, not additional runs or independent replications.

A separate standard-library [exporter](source/export_final_tables.py) reconstructs
all 168 values directly from original MTEB score JSON, verifies the original bytes
and computes each 14-task mean using exact rational representations of the stored
binary64 inputs. It accepts exactly the union of the two original six-run queues,
with all twelve final checkpoints present. It does not accept partial-task means.
This is independent score-field/mean reconstruction, not a fresh reconstruction
of query-level rankings from embeddings.

The original six-checkpoint exporter is unchanged. The new copy changes only
the exact cohort requirement and its output descriptions/counts. It rejects the
old six-record input before creating an output directory. That refusal is a
component check, not another evaluation. A fresh invocation of the archived
exporter on the archived twelve-record input also exits zero and reproduces all
three CSV/Markdown outputs byte-for-byte. The replay manifest differs only in
its actual input/source paths, which must remain accurately recorded.

## Scope and continuation

This archive provides a complete **descriptive final-checkpoint grid**, not a
validation-selected optimizer winner, task/query significance test, multi-seed
training result, full-trajectory comparison or useful-dimension mechanism.
Learning-rate cells are not independent training seeds. Historical scores do
not fill any current cell. Source-release and whole-experiment scientific
completion flags stay false, and no generated manuscript result is installed.

The existing evaluation queues continue through all four intermediate stages.
Validation and functional work retain their original source/selection/resource
rules. This work does not change GPU scheduling or leases, any numerical source,
data version, checkpoint, scoring rule, historical controller, or protected helper.
There is no Git commit, remote write or source publication.

[observations.json](observations.json) contains separately timestamped live-handle
observations, not a permanent heartbeat. [commands.json](commands.json) retains
actual commands and terminal results. The initial archival-copy command used an
unsupported local `cp` option and exited 1 without copying files; the subsequent
supported no-overwrite copy and verified replay are separate recorded calls.
This tooling failure is not a training or evaluation failure.

[verification.json](verification.json) binds the scoped checks and files. The
[before copy](before/CURRENT_EXPERIMENT.md) preserves the handoff previously bound
by the coordinate-utility interpretation archive; neither prior archive is edited.

To reconstruct the tables on this original host, use a new output directory:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  source/export_final_tables.py \
  --input actual/all-twelve-final-readback.json \
  --output /absolute/new/output-directory
```

The reader/exporter still require the original authenticated local sources and
raw files. This is not a portable second-host experiment or final repository release.
