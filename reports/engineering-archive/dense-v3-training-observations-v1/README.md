# Complete native training dynamics

Observed and independently replayed **2026-09-10**. These are genuine telemetry
from **all twelve completed v3 primary runs**, not simulated trajectories or a
retrieval/optimizer-quality verdict. The unchanged full retrieval and functional
campaigns remain required. No manuscript or frozen numerical/dispatch source changed.

## Read the actual results

![All twelve training trajectories](actual/outputs/training-trajectories.png)

The [vector figure](actual/outputs/training-trajectories.svg) and
[PDF](actual/outputs/training-trajectories.pdf) preserve the same full curves.
The image was actually viewed: all legends, axes, curves and footer are readable.
All numerical outputs below are unchanged copies from the original producer.

| Output | Complete population |
| --- | --- |
| [Logged training](actual/outputs/logged_training.csv) | 4,692 observations: 391 per run |
| [Five-stage training summaries](actual/outputs/stage_training.csv) | 60 run/checkpoint rows |
| [Accepted timing segments](actual/outputs/timing_segments.csv) | 60 original segments |
| [Systems](actual/outputs/systems.csv) | 12 complete runs |
| [Run summary](actual/outputs/run_summary.csv) | 12 complete runs |
| [Full-precision JSON](actual/outputs/tables.json) | All five tables |
| [Original manifest](actual/outputs/manifest.json) | Nine output bindings and all 132 native JSON inputs |

These runs have one common training seed, not twelve independent random-seed
replicates. The same 500K groups, seven own negatives, no in-batch negatives,
temperature 0.02, context 8192 and global batch 128 apply. Muon/NorMuon's listed
rate is the hidden-matrix rate; their auxiliary AdamW rate is 3e-6.

The terminal native train_loss and trailing-ten-logged-observation mean are
different quantities; neither is a held-out loss. This rounded table is copied
from the independently verified full-precision results.

| Optimizer | Configured LR | Native full-run train_loss | Final trailing-ten loss | Accepted hours |
| --- | ---: | ---: | ---: | ---: |
| AdamW | 1e-6 | 0.673167 | 0.542790 | 7.806 |
| AdamW | 3e-6 | 0.507251 | 0.392961 | 7.832 |
| AdamW | 1e-5 | 0.388173 | 0.301710 | 7.848 |
| AdamW | 3e-5 | 0.324197 | 0.253564 | 7.825 |
| Muon | 1e-4 | 0.404040 | 0.300085 | 7.802 |
| Muon | 3e-4 | 0.326952 | 0.245926 | 7.822 |
| Muon | 1e-3 | 0.328991 | 0.248790 | 7.845 |
| Muon | 3e-3 | 0.506304 | 0.371276 | 7.844 |
| NorMuon | 1e-4 | 0.412519 | 0.305479 | 7.835 |
| NorMuon | 3e-4 | 0.330282 | 0.246948 | 7.791 |
| NorMuon | 1e-3 | 0.321906 | 0.242696 | 7.721 |
| NorMuon | 3e-3 | 0.478063 | 0.353628 | 7.858 |

Within AdamW's tested grid, the final trailing training loss decreases as rate
increases. Muon's 3e-4 and 1e-3 trajectories finish near each other; both Muon-family
3e-3 curves show an early loss decrease, a rebound, and later decline. All recorded
loss/gradient/LR observations are finite. The rebound is not a failed run or proof
of worse retrieval. These are post-observation descriptive readings, not revised
validation selection, uncertainty intervals or an explanation of useful dimensions.

| Optimizer | Accepted-hour range, four rates | Maximum serialized optimizer state, GiB | Peak allocated memory, GiB |
| --- | ---: | ---: | ---: |
| AdamW | 7.806–7.848 | 1.110–1.110 | 35.246–35.246 |
| Muon | 7.802–7.845 | 0.699–0.699 | 34.827–34.827 |
| NorMuon | 7.721–7.858 | 0.700–0.700 | 34.830–34.830 |

Each run used four NVIDIA L20Z devices. Elapsed times do not demonstrate a large
whole-training speed advantage in this campaign; this is not an equivalence test
or an isolated optimizer-kernel benchmark. Smaller serialized optimizer state is
a systems measurement, not a novel embedding mechanism or peak-memory saving of
the same percentage. Hardware scheduling and startup are not experimentally
controlled by these descriptive summaries.

## Measurement definitions and boundaries

- Native logs contain step 1 and every tenth step through **3900**. All runs
  actually finish at **3907**. There is no invented final minibatch-loss row.
  The last native history entry is the separate full-run TrainOutput summary.
- Each stage uses the last ten available logged observations at or before
  checkpoints 782 / 1563 / 2345 / 3126 / 3907. Actual window endpoints are retained.
  Sample standard deviation is descriptive, not a confidence interval. No model
  is re-evaluated on a fixed training set by this operation.
- Raw logged curves are unsmoothed; their x-coordinate is logged step / 3907.
  The logging system itself aggregates loss between logging events. The displayed
  values are those native records, not invented per-update minibatch measurements.
- All 4,692 sampled gradient-norm records exceed the clip threshold of one.
  These are pre-clipping logged norms, **not an every-update clipping count**.
- Accepted elapsed time sums five maximum-rank segment durations. It includes
  checkpoint saves and work within those training segments, including loading
  batches; it excludes pre-on_train_begin model setup and external queue waits.
  No overhead is subtracted and no per-step wall timestamp is interpolated.
- Ten runs retain observed OS exit zero. The final two retain unobserved/null
  exit codes alongside their genuine complete-artifact and termination evidence.
  The original released-source/single-selection flags remain unchanged. This
  metadata reader does not impersonate the complete scientific consumer.

## Independent verification

The [independent first read](actual/independent-first.json) authenticates all
132 original JSON files and copied inputs; it checks all 60 checkpoint log prefixes,
all 4,692 raw records and every field of the five tables.

It executes only hash-authenticated original pure `system_rows` and
`timing_summary` functions, with their small helper dependencies, from the
[retained reference sources](reference-sources/). A metadata adapter supplies the
already accepted twelve identities; no full OutcomeContract or old release guard
is claimed to have passed. No model, optimizer, Torch or GPU is loaded.

Trailing means, sample variances and medians are independently reconstructed using
exact Fraction arithmetic on the original binary floating-point inputs, with a
rounded square root. Maximum standard-deviation difference is **3.469446951953614e-18**.
Across **44,244 checked scalar fields**, all non-standard-deviation fields agree
exactly. All five CSV files faithfully transcribe their JSON tables.

A fresh process reads only the copied input bundle into a new output directory.
**All 143 files are byte-identical**: nine numerical/figure outputs, the manifest
and all 133 input files. A [second independent full numerical read](actual/independent-replay-first.json)
also passes. This is actual same-host fresh-process reconstruction, not a physical
second-host experiment.

The [35 focused tests](actual/focused-first.xml) pass, with no failures/errors/skips.
They use explicitly synthetic small history/file fixtures for missing/reordered
logs, invalid numbers, schedules, terminal-loss distinction, unsafe paths, changed
inputs, incomplete stages and reference-source refusal. The actual full-population
checks are the separate genuine artifact reads above, not these fixtures.

## Source, execution and preservation

- [Plan and original scope](plan.md): post-observation description, no inference change.
- [Producer](source-first/observe_primary_v3_training.py):
  SHA-256 `5424027d6363b1edbccb5c7916f0211bd9f67d355d39aaf1c4a908fe3e8127de`.
- [Independent auditor](source-first/audit_primary_v3_training_observations.py)
  never imports the producer.
- [Actual commands and observed exits](commands.json), [original logs](actual/),
  [figure inspection](figure-review.json), and [dated runtime observations](observations.json).
- Original production is at
  `/root/embedding-optimizer-v3-experiment/analyses/dense-primary-v3-training-observations-v1`.
  Its `inputs/admission.json` and `inputs/native/` contain all authenticated
  reconstruction inputs. They are not newly published data/source.
- Fresh replay remains at
  `/tmp/dense-v3-training-observations.GnXWzfYq/replay-first`.
  Keep both original and replay outputs; do not overwrite them.

The original manifest is 25,227 bytes with SHA-256
`98e3a7cae491d4bb06ae949a909f9466690e8f5d40edb600a3e5ce1e2ba17205`.
Its parent is the existing 19,515,868-byte genuine admission, SHA-256
`be6d92fda2f7f0ba3e86f2c2866fa9dca530f066d29ce92d0887fdad91684067`.
The [preservation verification](verification.json) records this milestone's exact
sources, tables, evidence, handoff before-copy and unchanged primary/live sources.

At **13:43 UTC**, primary BEIR is **32 / 840** with all eight exact workers live/R;
baseline is **14 / 14**. Validation remains **7 / 12**, functional **0 / 61**,
both coordinators live/S. No failure receipt is present. The previously asked
one-GPU priority question remains unanswered. No lease, worker, source, controller,
protected helper or resource policy was changed.

Next complete the existing full retrieval and validation populations, functional
vectors/interventions and held-out-dose inference, and separately admitted real
crossed continuations. Then consume the complete evidence into the NAACL paper
and reproducible release. These training curves do not fill missing retrieval
cells or establish a useful-dimension mechanism. No new blog, HF/GitHub write or
overall scientific completion is claimed.
