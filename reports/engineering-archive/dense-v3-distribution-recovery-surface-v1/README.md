# Current v3 distribution and recovery coverage

Bounded engineering progress, **2026-09-13**. This is not a source release,
scientific admission, GPU-resume result, new experiment or final paper.
All live numerical/document/dispatch sources remain unchanged.

## What changed

The actual declared wheel omitted every current `*v3*.json` configuration,
the three implemented restoration scripts, and the current restoration guides.
Only `pyproject.toml` and a new two-case distribution test changed in task code.
The existing dirty worktree was preserved; the
[task-only diff](actual/pyproject-task-only.diff) compares the actual pre-task
file, not Git HEAD. No existing data-file declaration was removed.

The 40 added data files comprise 17 configuration files, three restore scripts,
18 restoration guides, the primary paper-results reproduction guide, and
`CURRENT_EXPERIMENT.md`. The recursive exact configuration-reference closure
from all fourteen current v3 roots plus the claim-wording amendment contains
47 files. Referenced historical parent configurations remain provenance, not
current scientific results. No configuration contents, optimizer/loss code,
dependency requirements or console-script declarations were changed.

## Actual verification, including failures

| Check | Observed result |
| --- | --- |
| Pre-task build, session 80569 / terminal fea860 | Exit 0; actual wheel and sdist; 158 declared data files |
| Original pre-task full distribution audit, e04283 | **Exit 1**, 11 findings; missing executable configuration plus ten producer-path findings |
| Expanded build, session 49990 / terminal 09c388 | Exit 0; actual wheel and sdist; 198 declared data files |
| Original expanded full distribution audit, b28dae | **Exit 1**, ten producer-path findings remain; missing configuration resolved |
| Focused tests, final 8853a7 | Exit 0; 21 tests, zero failures/errors/skips; two new cases plus 19 existing cases |
| Final Ruff checks, 9c03dc / 3a9175 | Exit 0 for checking and formatting |
| Actual sdist-to-wheel round trip, session 86225 / terminal 05cde0 | Exit 0 at 00:00:33 UTC; all 408 member names and payloads byte-exact |
| Three restore scripts extracted from the rebuilt wheel | Each `--help` exits 0 from a new directory with empty `PYTHONPATH` and hidden CUDA |

Both full original audits are preserved unchanged:
[before](actual/distribution-audit-baseline.json) and
[after](actual/distribution-audit-expanded.json). **The full distribution gate
has not passed.** The remaining ten findings span seven unique source/document/
test paths: a frozen factorial input adapter, two historical handoff documents,
and four tests containing provenance or negative-control paths. They require
reviewed separation of runtime portability from preserved historical evidence;
do not hide strings, weaken the old scanner, discard controls, or edit a live
source binding to make this gate green. The initial format check (663fd2,
exit 1) concerned the new test's line wrapping; formatter 6676af changed only
that test before the final checks. No scientific validation was altered.

The expanded wheel SHA is
`51a65e250a75b5e54f459c12cd930b7bf8cd87dc2c0928889091928be4e37f6e`;
the sdist SHA is
`4c246130848502e29aeb414b94e79921d1bbaaabdb276b1c34454c64f9b10a77`.
The rebuilt wheel SHA is
`2d5e7a37cd92693c254db9f0ef011242097004c99a34224199789c97e93a6ec7`.
ZIP metadata differs; every actual member payload agrees. See the
[round-trip receipt](actual/sdist-roundtrip.json). This is same-host source
relocation, not a physical second machine, fresh environment, model download,
native experiment admission, all-CLI execution, or an install-and-train claim.
The existing interpreter/build stack was used without installing or upgrading
anything. Corrected training still has its original draft/released-source gate.

The archive preserves both pairs of actual built artifacts, the rebuilt wheel,
723/761-file input inventories, original audit source and failures, both test
XMLs, exact build/round-trip drivers, task-only source delta, added data, and
before/after operational handoffs. Built documents are the pre-handoff-update
snapshots recorded by their original inventories, not perpetual current status.
They are local evidence, not GitHub/HF/source publication. Builders and completed
checks must not be rerun as missing scientific work.

## Existing experiment remains active

At **00:06:57 UTC**, the unchanged continuation BEIR has **10/168** completed
task units. Both coordinators and all eight exact registered workers are live;
workers are R, and neither pool has a terminal failure receipt. All six
downstream CPU waiters are live/S, without actual collectors, complete paper,
combined replay, outcome upload or GPU-resume ranks started. Exact observations
are in `actual/*-closeout.json`; all are read-only and exit zero.

The **23:46:42 UTC** nonblocking stack samples are also preserved: all eight
exact workers are in corpus encoding (tokenization/tensor construction or
model forward). The four ClimateFEVER logs show the original built-in OOM
fallback from 3,000,000 to 1,500,000 characters. That bounded log filter is
ClimateFEVER-specific; empty FEVER matches do not establish absence of fallback.
No source/context/corpus changed and no agent restart occurred. A stack sample
is not continuous throughput, instantaneous GPU utilization, a complete task,
or an extrapolatable fraction of wall-clock work. The copied
`evaluation-final.json` is the original 23:44 snapshot used as sampler input;
the actual later observation is `evaluation-closeout.json`.

All 24 scientific training runs / 120 checkpoints, primary 840-task results,
complete probes, tracking and prior numerical reconstructions remain complete.
Finish the unchanged actual full-corpus queue, genuine native statistics,
strict document/replay and its review, actual GPU resumes, outcome durability,
then reviewed source release. The separate five prose corrections remain
uninstalled until the native document and combined replay finish unchanged.
No engineering incident from this archive belongs in the manuscript.

This turn is **PROGRESS**, with continued verified waiting for real BEIR.
The overall scientific/paper/reproducibility goal remains active.
