# Complete final-checkpoint and validation artifact recovery

This milestone preserves the completed **12-run final BEIR grid and all twelve
validations**, not a new experiment or a completed paper. The immutable public
snapshot contains **660 files / 29,526,987 logical bytes**. All were recovered
anonymously; independent offline hashing and numerical replay passed.

## Actual sequence and scope

- Source preparation returned zero; all native outputs, original worker records,
  six derived tables and the sixty-checkpoint index were content-bound. The
  payload contains 264 native measurement files, 384 original worker records and
  the remaining selection/readback/index/table metadata. All 49,152 validation
  records were screened as JSONL, not merely by filename. No raw query/document
  texts, program bodies, model tensors or credential-shaped strings were included.
- Forty scoped transport tests passed with zero failures/errors/skips; the
  [original XML](actual/transport-tests.xml) is retained. These pure guards are
  not GPU tests or scientific experiments.
- Metadata-only upload-mode preflight verified all 660 paths would be new regular
  Git blobs, with no ignored or previously existing path. The single upload
  returned zero at **2026-09-10 17:21:08 UTC**, including strict remote verification.
  All twenty other root entries, the card, `.gitattributes` and both previous
  corrected subtrees stayed unchanged. No prior payload was overwritten/deleted.
- Full anonymous recovery returned zero at **17:28:18 UTC**, into a fresh local
  destination. All 660 files / 29,526,987 bytes match their immutable identities.
  The original client emitted an unauthenticated-rate warning but no failed
  transfer. There was no duplicate download or credential substitution.
- Independent recovered-input numerical replay returned zero at **17:37:09 UTC**:
  all 168 raw BEIR scores, twelve endpoint means, 49,152 validation rows,
  294,912 scalar metric checks, 576 group-metric means, four CSV tables and the
  original three validation-only choices agree. Maximum scalar loss discrepancy
  is 5.5888e-7 under the unchanged original 2e-5 / 2e-6 tolerances. Recorded FP32
  values are preserved rather than replaced by the independent FP64 reference.
- The guide's separate standard-library offline block was actually executed on
  the recovered copy and passed. Its download convenience block was syntax-
  checked only; the real transfer was the original uploader's anonymous mode.
  The earlier staging replay is explicitly staging-only, not recovery evidence.

Actual commands, session/chunk identities, exit results and the two harmless
missing-`jq` inspection errors are in [commands.json](commands.json). Those shell
inspection errors were not upload, numerical or GPU-worker failures.
The first local archive scan also refused the intentionally synthetic `hf_` plus
24 `a` characters in one negative test's XML parameter ID. Its failed call and
original checker source are retained in [verification-first-failed.json](verification-first-failed.json)
and [source/verify_archive_first.py](source/verify_archive_first.py). The final
local check binds that exact XML hash and named synthetic fixture; every other
archive field remains screened. This is not an exception for actual credentials,
and the public-payload scanner is unchanged. No test XML was uploaded.
All timestamps above are observer-reported UTC. This is same-host recovery,
not a physical second-machine experiment or full forward re-evaluation.

## Durable entry point

Dataset `qcz/embedding-optimizer-study-analysis-artifacts`, revision
`3883b677f87b1982f06016e9fadb8bb95e0cfc96`, adds one prefix under
`corrected-dense-correctness-v3/complete-endpoint-evaluations/` with primary
protocol `4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b`
and manifest `bf443d5623c1312f8661a918ab93f9aea5f119058307733d477c8eac0c6da74f`.
Read the [complete restoration guide](../../../docs/evaluation-analysis-restoration.md)
for the exact immutable link and offline byte-verification instructions.

[actual/remote-audit.json](actual/remote-audit.json) and
[actual/download-verified.json](actual/download-verified.json) retain the actual
remote and local recovery proofs. The original [upload receipt](actual/upload.json)
is preserved verbatim, including its pre-audit false durability flag; the separate
actual remote audit establishes completion. The new successful strict-root audit
does not reinterpret the older training-backup root-byte exception.

## Reconstruct scores without a GPU

The standalone [replay_scores.py](source/replay_scores.py) reads only the selected
immutable downloaded root. It checks the complete manifest, reconstructs validation
and loss-only selection before reading BEIR, then verifies all-rate and selected
endpoint tables. Run from the repository root, substituting an absolute recovered
snapshot root and a new output filename:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  reports/engineering-archive/dense-v3-evaluation-artifact-backup-v1/source/replay_scores.py \
  --root /absolute/path/to/recovered/prefix \
  --output /absolute/path/to/new-replay.json
```

Only this replayer is standalone. The archived uploader uses original host paths
and the unchanged recovery hash helper; it is historical execution evidence, not
a new-machine upload instruction. It must not be rerun against the completed
remote prefix. No executable source was published in the HF dataset snapshot.

The original BEIR JSON contains task aggregates, not per-query retrieval rankings.
Offline replay does not recompute model forward passes, raw-text decontamination,
query-level BEIR rankings, uncertainty intervals or a weight-space mechanism.
The validation choices are AdamW 3e-5, Muon 3e-4 and NorMuon 3e-4; BEIR never
selects those rates. Four rate cells are not independent training seeds.

## Preservation and unfinished work

The preceding [validation-selection archive](../dense-v3-validation-selection-readback-v1/README.md)
and its complete endpoint parent remain unchanged. Its former live handoff binding
is now preserved in [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md).
The [existing observer snapshot](observations.json) has primary BEIR **172 / 840**,
eight live exact workers, and functional **0 / 61**, waiting for a leased GPU after
validation completion. No resource handoff, numerical-source or dispatcher change
was made. All twelve primary training runs and sixty checkpoint backups remain
complete; do not retrain them.

Forty-eight intermediate checkpoint evaluations, all functional measurements,
the crossed continuation, complete scientific reconstruction, source availability
and the NAACL paper remain unfinished. Paper/generated findings and frozen
scientific rules were not edited. [verification.json](verification.json) binds
this bounded archive and the two current documentation files, not whole-goal
completion. No historical erasure, GitHub retry or protected-helper access occurred.
