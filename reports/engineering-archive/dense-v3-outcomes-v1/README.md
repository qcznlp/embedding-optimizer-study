# V3 outcome integration and independent CPU replay

The v3 primary/validation chain now has a strict outcome adapter. It requires complete authenticated
validation first, without BEIR input, followed by all twelve complete primary runs, sixty stages and
840 distinct task cells. It reuses the unchanged statistical kernel, exporting all raw task scores,
primary/secondary contrasts, stage summaries, observed-range AUC, validation metrics and system
measurements from deep-admitted whole-run metadata.

The active [v2 implementation protocol](protocol-v2.json), SHA-256
`2a12b78a1a309cd6cc4a56593e4b1b1030370fc579045d1708d572eb5577d68e`, remains **preparation-only**.
Its predecessor `914996b3...` and failed source are preserved in [attempt-1](attempt-1/).
Neither the primary `4400f1ce...`, validation `96a8aa2f...`, old statistical protocol
`7f321f44...` nor old statistical kernel `b22f5a9b...` has changed.

## Statistical rules retained

The primary endpoint averages all four declared rates within optimizer at the final checkpoint,
then contrasts optimizers over the same fourteen tasks. The three contrasts use the unchanged
50,000 shared paired-task resamples, seed 20260903, observed fixed standard errors and simultaneous
max-T 95% intervals. Nominal intervals cannot determine support. Validation-selected recipes are
secondary; exact loss ties choose lower learning rate. Learning rates are not independent seeds.

All five checkpoint stages are reported descriptively. AUC covers only observed progress 0.2–1.0;
no initialization score is imputed and no best-stage endpoint is chosen. Missing high-rate runs,
duplicate/unknown cells, wrong stage/recipe identities and non-finite/out-of-range nDCG are rejected.
System throughput uses accepted complete timing, not unverified Trainer throughput or an invented
prior-time correction. System measurements remain descriptive and bound to the observed hardware.

## Actual checks and preserved failure

The first implementation passed 65 and then 67 focused tests, and 1,844 isolated tests, but its
actual independent-process fixture readback **failed**. CSV headers inherited Python dictionary
insertion order; canonical JSON reordered the keys. A separate CSV check confirms all 840 rows'
cell values remained identical. The failure is retained in
[attempt-1/failure.json](attempt-1/failure.json), alongside the exact failed source/protocol.
`/tmp/dense-v3-outcomes.pMqV0Z` preserves its original fixture artifacts.

The separate implementation revision sorts CSV field names. No metric value, statistical kernel,
estimand, tolerance or support rule changed. The new key-order round-trip regression would reject
the old serializer. The revised focused suite passes **69 cases** and the full isolated suite
passes **1,846 cases**, zero failures/errors/skips. These do not clear the different numerical
candidate's eleven old unwaived source-contract failures among 963 cases.

The [actual CPU rehearsal](rehearsal.json), SHA-256
`82aa59dde630634a2224f3f12023646e1886d797ede98b64f3db8f8cd80874f2`, completed at 18:09:01 UTC:

- An explicitly **synthetic** 840-cell fixture produces all nine score/validation tables. Independent
  scalar means, sample variance and interpolated quantiles agree with all 21 checked statistics
  from the locked 50,000-resample kernel. Maximum observed error is `2.220446049250313e-16`, under
  the unchanged predeclared absolute/relative `1e-12` check. The declared NumPy index generator
  is shared; the scalar aggregation/quantile implementation is separate.
- Changing only validation selection leaves both primary tables identical and changes secondary
  task contrasts. These are mathematical fixture checks, not observed optimizer effects.
- A separate CPU process reloads the serialized fixture and recomputes the complete bundle
  successfully. The reader compares stored bytes with fresh expected evidence/statistics, not
  just a self-declared hash.
- Thirteen changed protocols reject. A status-only primary-release projection also rejects:
  the bound draft-parent identity changes, so a release requires a reviewed parent/source transition.
  No real release, commit, source assembly or execution is performed by that projection.
- Four byte-modified fixture bundles reject even after their copied hashes are refreshed.
  Unit tests separately cover a changed statistical support label and changed expected raw evidence.
- The actual public outcome CLI against the live experiment rejects missing v3 full runs before
  reading BEIR or creating output. **No formal outcome or primary recipe selection is produced.**

Preserve `/tmp/dense-v3-outcomes-replay.EEVh60`: 72 bound artifacts / 1,476,182 bytes plus the
receipt. Every positive table here is synthetic and explicitly labelled as such. The standalone
rehearsal does not supply actual twelve-run system measurements; that projection is covered by
isolated full-collection fixtures and strict upstream run admission.

## Completion boundary and next work

The outcome adapter and bounded independent replay are no longer missing implementation. Actual
complete v3 training/validation/BEIR evidence still does not exist. Old twelve executed runs and
sixty preserved checkpoints remain on scientific hold. No optimizer advantage, mechanism finding,
speed claim or manuscript result follows from these tests.

Next integrate weight-space/dimension and publication consumers, separately routed factorial
control, then the reviewed single-source/runtime/parent handoff. All three preparation contracts
bind draft parents; changing a status field is not a complete release mechanism. Do not merely
refresh old source hashes, repeat resolved serializer checks or adopt historical model outputs.

The owner was asked again for explicit WIP-commit and post-acceptance twelve-run rerun direction;
no answer had arrived at this handoff. Existing access/approval gates remain. All owned calls have
exited. Original numerical cores, manuscript, exact stopped BEIR chain, ledger and lease are
unchanged. No GPU worker, training, controller transition, commit/push, HF mutation or protected
helper interaction occurred. This local engineering milestone belongs in no manuscript section.
