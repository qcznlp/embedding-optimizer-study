# Exact owned calls

Observed 2026-09-07 09:28 UTC. **44536 is terminal, exit 0**: the whole-development
pytest run passed all 2,636 cases, with zero failures/errors/skips. Its 1,025.358-second
JUnit record is `/tmp/dense-v3-primary-publication-tests.EAQrsP/full.xml`, SHA-256
`8c46fda787ba7417fcf9954ba62deb9fdc5a664bff70c5671831e2e2fbd70ee5`.
No numerical/test handle listed here remains live. Do not poll or restart them.
This whole-development result does not accept the separate 963-case training candidate.

Terminal calls — do not poll again:

- Joint replay 74613: exit 0; joint validator 26171: exit 0, acceptance
  `e6fd1dd99d14d8eec86937f63352a7e5b75083c4ef519d2b07fd56e8b8e85073`.
- 23057: first 50 focused tests pass.
- 71600: first complete synthetic generation passes; result SHA-256
  `878940c7a59f4418022f01af80b3365be8ef751e2cfd0580b69048917c024497`.
- 98564: first CSV roundtrip test exits 1, three failures among four controls;
  preserved in exact-csv-attempt-1/, with its package source in source-first/.
- 57786: 57-case retry passes. 50104: complete generation after the type fix
  passes with six identical outputs, result SHA-256
  `ca672b515fa520aa132f5bab0faf826029e2684e2690426b11cb09c536fa2cde`.
- 49312: first full-layout command exits 1 because the dimension table label is
  absent. PDF/logs/source remain preserved; the main endpoint was already page 7.
- 87580: final 63 focused tests pass, zero failures/errors/skips.
- 81259: final complete generation passes, result SHA-256
  `b4b09cd1af09772636c88cc47ab8da3c0551a00031a99a89f1a85a4dbf89b1d0`.
- 96597: full synthetic layout passes, result SHA-256
  `20b8dabeede997426e0a90b336d215da7f300894718e62650df3c03e226da6de`.
- 83522: canonical draft preparation exits zero, SHA-256
  `39c8740962fbfa57770bc21058bfc18b3f862cf97f7b12f48ad29dcd04b43f81`.
- 13529: preservation precheck exits zero at 09:14 UTC. All 54 direct joint-parent
  bindings are preserved (root handoff docs via before/), 349 generation/layout
  bindings pass, all three generated attempts and failed tests remain intact,
  the canonical draft loads, both six-file numerical cores and the manuscript
  are unchanged, and all three exact owned dispatcher states remain T. This
  precheck does not supply a terminal whole-suite or publication acceptance.

No positive actual checkpoint-backed publication, complete portable primary
publication, strict manuscript integration, formal training or remote action is
verified here. All AGENTS.md restrictions remain in force.
