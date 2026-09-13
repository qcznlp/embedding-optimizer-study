# First full-suite failure: mixed historical test modules

The 37-case focused suite passed. The first full isolated run had 2,418 cases and **17 failures**,
all raised by `running_sources` before numerical reconstruction. Its exact source snapshot and
JUnit are retained here; no production source or numerical rule was changed to remove the failure.

`tests/test_main_resume_repair.py::load_controller` registers historical snapshots as
`embed_optim._main_resume_before` and `embed_optim._main_resume_after`, without removing those
registrations. When the later reconstruction tests run in the same process, the archived `after`
controller differs from the package source. The new strict gate correctly refuses this mixed
implementation; its standalone, archived-source-only cold process passes input loading.

The new test module now temporarily isolates only those two exact historical test aliases,
checks their expected archival paths and restores them after every test. Frozen historical tests
remain byte-identical. A new explicit injected-foreign-source test requires the production gate
to keep rejecting mixed implementations. No runtime/source admission check or scientific tolerance
was relaxed, and the numerical kernel was not edited.

Original full receipt: `/tmp/dense-v3-vector-reconstruction-tests.zaQFHV/full-initial.xml`,
369.512 seconds, 2,418 tests, 17 failures, zero errors/skips. The retry must be a new receipt.
