# Accepted pretrained-vector data preservation

On 2026-09-12, the existing accepted pretrained functional-probe state was added
to a **new data-only HF subtree**, with no encoding, analysis or coordinator restart.
The [recovery guide](../../../docs/pretrained-vector-restoration.md) supplies exact
revision/path/hash coordinates and the portable NumPy reader.

The sole upload returned zero and created revision
`37702ccb55820dbf5956a57257c5582d56733da9` at **02:06:43 UTC**, parent
`fdad53c239d9ed59141fcafbd92a0438bb1a657f`. The remote audit verifies all ten files,
twenty unchanged other root entries and nine unchanged corrected-analysis subtrees.
No root attributes changed: only the NPZ used the already existing LFS rule.
All **10 files / 6,439,310 bytes** were anonymously recovered at **02:07:07 UTC**.
The archived standalone reader then passed against those recovered files from
outside the project with `python -I -B`; it does not import model/project code.

The original eight accepted files were preserved byte-for-byte: vectors, vector
manifest, five native receipts and the independent original payload readback.
The ninth payload is a new explanatory README; the tenth file is its encompassing
manifest. This is still **1 / 61 vectors and 0 / 61 features**, not a new scientific
result or full functional-campaign recovery. The failed original record and frozen
numerical/dispatch authority remain unchanged. No source publication was attempted.

## Evidence

- [commands.json](commands.json): actual commands, sixteen passing CPU tests,
  original upload/recovery session results and isolated-reader results. The HF
  anonymous-download warning is retained; both owned sessions exited zero.
- [actual/preflight.json](actual/preflight.json): source identities and exact population.
- [actual/upload-mode-preflight.json](actual/upload-mode-preflight.json): one permitted
  NPZ LFS addition and nine regular additions; no ignored or overwritten paths.
- [actual/upload-started.json](actual/upload-started.json) and
  [actual/upload.json](actual/upload.json): the one-shot attempt and actual commit.
- [actual/remote-audit.json](actual/remote-audit.json): remote content and preservation checks.
- [actual/download-verified.json](actual/download-verified.json): full anonymous
  download and recovered-native array/provenance readback.
- [source/verify_recovered.py](source/verify_recovered.py): standalone reader; copy
  it and the guide off-host separately because the public snapshot contains no code.
- [source/test_backup.py](source/test_backup.py): bounded data-transport tests.
- [observations.json](observations.json): separately timestamped exact primary
  observer/worker reads. The eight original workers remain live at 676 / 840;
  this preservation operation did not change their priority or launch a task.
- [before/CURRENT_EXPERIMENT.md](before/CURRENT_EXPERIMENT.md): previous handoff.

The adapter [source/backup.py](source/backup.py) is retained for provenance only;
its one-shot upload must not be rerun. Portable recovery uses the reader and the
immutable data snapshot instead. This archive, guide and source are local, not
claimed as updated GitHub content or a physical second-host experiment.
