# Corrected DenseOn v3: complete native training-observation snapshot

All twelve completed primary runs and all five retained stages are represented.
This is a numerical/metadata backup, not a source-code release, complete paper,
retrieval ranking or useful-dimension mechanism finding.

The complete original 143-file observation bundle is under `native/`:

- `logged_training.csv`: all 4,692 native records, 391 for every run.
- `stage_training.csv`: all sixty trailing-ten-observation checkpoint summaries.
- `timing_segments.csv`: all sixty original accepted timing segments.
- `systems.csv` and `run_summary.csv`: all twelve complete runs.
- `tables.json`: the complete full-precision values of those five tables.
- `training-trajectories.svg`, `.pdf` and `.png`: identical full raw logged curves.
- `inputs/native/`: all 132 original run/checkpoint JSON metadata files.
- `inputs/admission.json`: the unchanged all-twelve completion/identity metadata.
- `manifest.json`: the original source-bound observation manifest.

`provenance/` contains both original independent numerical audit receipts, the
observation preservation receipt, and the exact sixty-checkpoint download index.
`artifact_manifest.json` is a separate transport manifest binding every payload.
Every original file is copied byte-for-byte. No model/optimizer tensors, pickle,
program source, query/document examples or credentials are part of this snapshot.
Source names/hashes and original machine paths are retained provenance, not code
execution instructions or local paths that must exist on another host.

All runs use the same fixed 500K query groups and one training seed. There is one
positive and seven own negatives, no in-batch negatives, maximum context 8192 and
global batch 128. Four rates per optimizer and every five-stage trajectory remain.
Learning-rate cells are not independent random-seed replications.

Native logging ends at step 3900; training finishes at step 3907. The final native
TrainOutput mean and a trailing-ten-logged-observation mean are distinct fields.
No terminal minibatch loss, per-step timestamp or extra observation is imputed.
The plotting code applies no smoothing; native loss logging already aggregates
over logging intervals. Gradient norms are sampled pre-clipping observations,
not an every-update clipping count. Sample deviations are not confidence intervals.

Accepted time is the sum of maximum-rank durations across five segments, including
batch-loading work and checkpoint saves. It excludes pre-training model setup and
external queue waits. No overhead was subtracted; this is not an optimizer-kernel
speed benchmark. Actual hardware is four NVIDIA L20Z devices per run.

The original tables and all saved log prefixes were independently checked using
native metadata, original pure systems/timing functions and a separate exact-
arithmetic reference for stage summaries. A fresh copied-input local replay
reproduced all 143 original files byte-for-byte. Those are same-host observations,
not proof of physical second-host training or complete source/scientific release.

All original false release/scientific flags and null/unobserved exit codes remain
unchanged. Ten runs have observed OS exit zero; the final two have genuine complete
artifacts and termination evidence with unobserved/null OS exit codes. Preservation
does not erase original guards or infer a missing exit code.

Primary protocol SHA-256:
`4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b`.
Original observation manifest SHA-256:
`98e3a7cae491d4bb06ae949a909f9466690e8f5d40edb600a3e5ce1e2ba17205`.

Authenticate the separate transport manifest against the immutable digest in the
project's restoration guide, then verify every file. Do not use a moving branch
or substitute historical checkpoints. Full BEIR, functional interventions, crossed
continuations, complete portable source and the NAACL paper remain separate work.
