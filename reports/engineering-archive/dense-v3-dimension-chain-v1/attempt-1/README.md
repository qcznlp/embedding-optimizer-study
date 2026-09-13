# Preserved first real-loading refusal

The first 53 focused synthetic interface tests passed under the retained initial source.
The first real CPU loading rehearsal failed before any synthetic export/feature work:

```text
output root: /tmp/dense-v3-dimension-chain-audit.JzeFAi
owned command session: 85391, terminal exit 1
verify_cpu_loading -> verify_loaded -> require_same(actual, expected)
ValueError: Artifact belongs to a different primary identity
```

No `result.json` was produced, no forward/model update/GPU worker ran, and the copied actual
probe files remain in that directory. The exact failed source and source-bound draft
`42ee1128fe376ead2ecccdcb387ca2bee3f62dedb57b401cf17d85d435843444` are preserved here.

Read-only localization found 134 actual and 134 saved tensors, but disjoint names:
the installed SentenceTransformer uses `0.model.<bare tensor>`, while TensorStore's ordinary
SentenceTransformer-directory reader already gives `0.<bare tensor>`. The new guard had
incorrectly prepended `0.auto_model.` to the latter. Its tiny synthetic fixture omitted the
module-directory metadata and shared the wrong assumed prefix, so that fixture did not detect it.

The proposed narrow correction explicitly maps canonical `0.` to runtime `0.model.`, requires
the exact complete key set and checks each saved tensor after BF16 conversion. The corrected
test fixture must include actual module-layout metadata. Do not broaden to arbitrary prefix
stripping, change model weights, relax numerical equality or re-label this failed attempt as passed.
The full first-suite receipt `/tmp/dense-v3-dimension-chain-tests.MSJXgc/full.xml` has
2,268 cases: three failures and one error (2,264 passing). Its failures are not waived:

- Two documentation tests expose missing hardware/provenance-ordering text after the preceding
  README consolidation. Restore those declarations without turning old operational commands into
  default authorization.
- A non-strict paper reader raises IndexError because the consolidated README omitted its
  `FINAL-CONCLUSION` section markers. Restore the markers around an explicitly pending notice,
  not the held historical conclusion; do not relax or rebind the frozen paper reader.
- The normalization stack guard correctly refuses the accidentally selected old live `.venv`:
  Torch 2.9.0 / Accelerate 1.14.0 / SciPy 1.18.0. The accepted `/usr/bin/python` has
  Torch 2.9.1+cu129 / Accelerate 1.13.0 / SciPy 1.17.1. Both use SentenceTransformers 5.7.0,
  but that shared version does not authorize substituting the whole environment. Re-run under
  the pinned interpreter; do not change the normalization guard or install over either environment.

Read-only equality localization (owned session 93166, exit 0) confirms the corrected mapping
gives exactly 134 keys and 134 tensor equalities after BF16 conversion on the real reference.
The complete retry still requires a new source-bound protocol and real/synthetic fresh replay.

The first four-case regression retry passed hardware, tracking-order and the actual pinned stack,
but revealed one stale positive test expectation after the missing-marker crash was removed:
it required the two historical outcome headlines to have complete evidence. The new pending README
correctly makes `ConfirmationHeadline` and `InterventionHeadline` incomplete through their old
outcome manifest. The preserved test predecessor is under `compatibility-before/tests/`.
The revised test explicitly requires both to be incomplete and their outcome manifests rejected;
the production paper reader, historical manifest, manuscript and release threshold are unchanged.
Do not restore held historical conclusions or refresh their old hashes merely to make this test green.

This is engineering provenance only, excluded from every manuscript section.
