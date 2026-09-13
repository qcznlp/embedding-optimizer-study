# Current v3 recovery and repository entry points

Verified September 10, 2026. This is checkpoint transport and handoff evidence,
not retrieval, useful-dimension, training-resume or publication acceptance.

## Completed deliverable

The top-level README and recovery guide now point to the **current twelve-run,
sixty-stage v3 campaign**, rather than the older held experiment. The new
[CURRENT_EXPERIMENT.md](../../../CURRENT_EXPERIMENT.md) separates completed
training/backups/weight measurements from unfinished evaluations, functional
analysis, crossed continuations and the paper. Original documents are preserved
under `before/`; the first revised entries are under `entry-first/`.

The standalone [recovery script](../../../scripts/restore_primary_v3.py) can list,
download and independently verify one stage, five stages or all sixty. It uses
explicit selections, immutable HF revisions and anonymous reads. It preserves
corrupt, extra and partial files; it does not deserialize models, run pickle,
modify the remote repository or obtain GPU resources.

## Actual evidence, with distinct scopes

| Evidence | Observed result | Original record |
| --- | --- | --- |
| Complete index | 60 immutable revisions, 1,200 files, 89,889,820,336 logical bytes | `actual/primary-v3-checkpoints.json` |
| Fresh anonymous remote metadata read | All 60 revisions / 1,200 exact paths match the original file sizes and LFS/Git digests | `actual/public-metadata-first.json` |
| Real complete checkpoint download | Muon 3e-4 step 2345: 20 files / 1,350,896,259 bytes verified | `actual/download-first.json` |
| Offline verification with final script | All 20 downloaded files verified | `actual/verify-final.json` |
| Offline explicit download resumption | 20 existing verified files reused; zero downloaded | `actual/resume-offline-final.json` |
| Independent original-receipt reader | All 20 full-file hashes, native seal and run identity agree | `actual/independent-read-first.json` |
| Focused tests on both script versions | 23 cases each, zero failures/errors/skips; synthetic payload tests are not model evidence | `actual/tests-first.xml`, `actual/tests-final.xml` |

The public metadata audit completed at **11:31:58 UTC**. It is not a fresh
download/rehash of all 89.9 GB. The independent reader completed its file checks
at **11:34:35 UTC** without importing the recovery implementation, accessing the
network/GPU or executing a model. The one real download is still preserved at
`/tmp/dense-v3-recovery-entry.m29J2t/download`; it is not copied into Git.
This is a fresh-directory recovery on the same physical host, not a second-host
end-to-end experiment or proof of subsequent bitwise distributed resumption.

The exact downloaded checkpoint revision is
`3b20ad40ac7cea5029b3058468381051a91ec7af`. The trusted index has 507,809 bytes and SHA
`76e1253373e24525044ebbb459aadcb56afa76da3db629e5b00a652ed44dbe89`.
The metadata read has SHA
`b324a01f3224021a7e422a0f7681b66d1490f1511dd16b8fa6ca8d7fda977c99`.

## Reconstruct the index from original evidence

`original-launch/` preserves all **180 original receipts**, plus the original
twelve-run completion/durability observation, SHA
`1d5aa49c6ec8c89c4454b4b9b641aa5e1bffc09a9967dbac6320cbe129d505f3`.
The original upload-stage records remain distinct from subsequent successful
remote audits; none was rewritten to make an earlier stage claim durability.

From the current checkout, choose an output that does not already exist:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" python -B \
  reports/engineering-archive/dense-v3-recovery-entry-v1/build_index.py \
  --launch reports/engineering-archive/dense-v3-recovery-entry-v1/original-launch \
  --output /absolute/path/to/new-primary-v3-checkpoints.json
```

The output must match the independently trusted index digest above. Do not
accept a changed index by replacing that digest. Actual download and verification
commands are in [the recovery guide](../../../docs/checkpoint-restoration.md).

## Source versions and observation limits

`source-first/` is the actual initial download version; `source-final/` is the
current script/test pair. Only receipt timing labels changed: the first
`observed_at_utc` was recorded before downloading, whereas the final version
records separate start and completion times. The first receipt is retained
unchanged; its timestamp is not described as download completion.

Actual tool handles 75982 (download), 95647 (offline verify), 58345 (offline
resumption), and 22760 (remote metadata) were observed terminal with exit zero.
The original final-test/list and independent-reader wrapper exit observations
were not retained across context compaction. Their complete XML/JSON records
are present and checked, but their OS exit codes remain **unobserved/null**.
Do not manufacture successful wrapper exits from an output file.

The original 11:28 runtime snapshot remains in `observations.json`. The separate
11:45–11:47 reads in `observations-later.json` show 17/840 primary tasks, 14/14
baseline tasks, eight exact primary workers alive, validation 7/12 and functional
0/61 waiting for validation. No failed/completed primary-campaign receipt was
present. These are dated observations, not permanent heartbeat claims.

The current source/index remains local WIP, not a published GitHub release. No
frozen numerical/dispatch source, training run, original receipt, manuscript,
historical controller or protected helper changed. No HF deletion, GitHub retry,
source publication or resource-priority handoff was attempted. Full scientific
outcomes, portable analysis reconstruction and the NAACL paper remain required.
