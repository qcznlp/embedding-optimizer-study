# Factorial claim calibration — scientific review, not experiment results

Status: mathematical counterexamples reproduced; a publication-only candidate is prepared and
tested under `after/`. **Neither the canonical manuscript/renderer nor a runtime contract has been
changed.** This is not another optimizer experiment and supplies no AdamW/Muon outcome.

## What the current evidence actually supports

The frozen summary computes final nDCG@10 after a 50K-query continuation. Writing its four cell
means as AA, AM, MA and MM (source state first), the three contrasts are:

```text
state       = ((MA - AA) + (MM - AM)) / 2
operator    = ((AM - AA) + (MM - MA)) / 2
interaction = (MM - MA) - (AM - AA)
```

The endpoint diagonal difference is exactly `MM - AA = state + operator`. The interaction is not
a third additive component. None of these contrasts decomposes the primary 500K-training outcome:
the endpoint, source pair, learning rates and reset histories differ. See
[the exact rational check](exact-algebra.json).

[The counterexamples](counterexamples.json) use synthetic bounded cell scores and the unchanged
production contrast/100,000-resample bootstrap functions. All three have the full synthetic
3-order × 14-task × 2-state × 2-operator grid. They are not measurements of trained models.

| Counterexample | Mathematical observation | Problem with the current generated interpretation |
| --- | --- | --- |
| Muon continuation harms both source states | Simple effects -0.25 and -0.125; interaction +0.125 | Increasing differences can be called complementarity mathematically, but do not establish beneficial continuation or full-trajectory co-adaptation |
| Both average main effects positive, interaction negative | State +0.3125, operator +0.0625, interaction -0.375 with an entirely negative interval | The renderer nevertheless says the benefits are additive |
| State interval positive; larger operator estimate uncertain | State +0.015625; operator +0.041667 with interval spanning zero | Significance of one contrast and not another does not establish state-effect dominance or equivalence of the other effect to zero |

Two further wording corrections are required:

- The abstract must not describe the three contrasts as a decomposition of an unspecified gain.
- Scale matching applies to the fixed calibration-probe gradient history and hidden update/weight
  ratio, not to the first realized update on the shuffled training stream. The existing scientific
  protocol explicitly records this clarification. Its numerical calibration rule is unchanged.

The factorial uses separate marginal 95% intervals, unlike the simultaneous max-T families in the
primary retrieval/dimension analyses. The manuscript must not imply family-wise coverage here.
Only three data-order seeds are resampled; they are not independent source-model training seeds.

## Coordinate utility and the cited reference

[Takeshita et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.1410.pdf) documents robust
performance under random dimension removal and coordinates whose removal improves performance.
Our interpretation is deliberately narrower: effective rank is not useful dimensionality, and
leave-one-coordinate-out score changes are context-dependent sensitivities, not additive shares of
a total score. Cosine renormalization and changing ranks couple the effects. The existing native-
basis and shared-rotation boundaries remain necessary. Only this specific citation was checked in
this review; it is not a complete bibliography audit.

## Prepared correction

`after/state_operator_factorial_publication.py` changes only `_interpretation`, `_render_latex`
and the descriptive claim boundary in `_expected_manifest`. All loading/validation functions,
metrics, point estimates, intervals, decision labels, thresholds and table rows are unchanged.
The existing conditional branch selection is retained; its prose is made weaker and precise.
All 27 positive/negative/inconclusive decision combinations retain their numeric output.

`after/main.tex` defines the actual endpoint contrasts, names the fixed source rates, explains
calibration-probe matching, states the marginal-interval boundary, and distinguishes averaged
continuation responses from feature mediation. The comparison of reached weights and functional
dimension use remains the paper's scientific focus. No implementation-incident narrative was added.

`before/` preserves the original renderer/manuscript and nine dependency/documentation files
byte-for-byte. The canonical files still have their prior hashes. The candidate has **not** been
integrated into the three-layer successor, source-published or deployed. The existing figure's
continuation-panel wording and development placeholder still need alignment during integration;
the successful candidate layout test alone does not close that work.

## Verification

Run from the isolated checkout with its own source and test directories explicitly selected:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor/tests \
python -B -m pytest -q reports/paper-review/factorial-claims-v1/test_candidate_wording.py
```

The latest successful receipt is `candidate-tests-v4.xml`: **34 passed**, zero failures/errors/skips.
It includes counterexamples, unchanged-
AST checks outside the three allowed functions, all 27 decision combinations, the existing abstract
word reserve and a complete synthetic-results paper build under the existing layout gate. The
synthetic PDF is explicitly labelled and is never a primary manuscript output. No gate was relaxed.
Its main text ends on page 6 under the existing eight-page layout gate. The durable
`candidate-layout-audit.json` preserves that synthetic-only receipt; `validation.json` binds the
exact candidate, counterexamples, test receipt and unchanged numerical/canonical source identities.

`candidate-tests-v2.xml` preserves an unsuccessful diagnostic attempt: its newly added word-budget
check used the macro name instead of the existing `state_operator_finding` dictionary key. The key
was corrected without enlarging the word reserve. Earlier passing receipts cover earlier, smaller
test selections and must not be relabelled as the final check.

## Next integration, with numerical decisions unchanged

1. Recheck the archived/current source identities and review the candidate against the frozen
   factorial estimands. Keep the original scientific/implementation protocols unchanged.
2. Align the conceptual figure and regenerate only a clearly marked development placeholder;
   never write synthetic scores into the primary generated include.
3. Add an exact publication-wording clarification binding. Refresh only the publication/completion
   identities it actually affects, preserving and verifying the previous three preparation layers.
   Main target `4380a363...` and all 47 successor commands remain required and unchanged.
4. Update the successor auditor and tests to validate the new exact layer without accepting
   arbitrary source drift or weakening any completion gate. Re-run actual consumer loaders,
   projection, complete synthetic layout and full tests. Until then, the candidate is not the
   active prepared successor contract.
5. Deployment still follows the separate owner-approved controller handoff and zero-step lease
   boundary. This review does not authorize a runtime takeover, WIP source publication or HF
   withdrawal. Training, original checkpoints and `gpu.py` remain untouched.
