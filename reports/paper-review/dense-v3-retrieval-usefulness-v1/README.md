# Retrieval usefulness: manuscript framing and method definitions

This is a **draft prose revision**, not a new experiment or completed scientific paper.
The authoritative [manuscript](../../../paper/main.tex) now follows one distinction:
does the optimizer merely distribute weight-space motion, or does the reached retriever
distribute useful ranking evidence more effectively?

## What changed

- The abstract and introduction move from a list of verification requirements to a scientific
  question connecting reached weights, coordinate utility, and full-corpus retrieval. The opening
  contrasts Muon pretraining/transfer evidence with optimizer-switching risks during adaptation.
- A cosine-margin identity motivates the distinction between distributed coordinates and helpful
  ranking evidence. It is not a new theorem, one-step optimization proxy, or additive interpretation
  of deletion-and-renormalization effects.
- Methods now specify the original task-first margin attribution, helpful **mass share**, and
  participation normalized by 768. Analogous nDCG attributions remain descriptive. The original
  nine-contrast simultaneous inference and four held-out-dose folds are unchanged.
- Weight magnitude is a ratio of joint Frobenius norms. Full-spectrum entropy and renormalized
  truncated-spectrum entropy are explicitly different measurements; nonzero parameter weighting
  in the full-spectrum rank summaries is disclosed. Neither branch replaces the other.
- Discussion and the authorial plan distinguish unsupported explanations from falsified mechanisms,
  and inconclusive optimizer evidence from an established negative result. Prediction and the
  crossed whole-state continuation do not identify feature mediation.
- HIL is described as balancing isotropic and anisotropic representations, not cited as if it alone
  proved that rank cannot predict retrieval. No new architecture or experiment is introduced.

The [paper guide](../../../paper/README.md) and [authorial plan](../../../docs/naacl-dense-paper-plan.md)
carry the same definitions. No historical performance number was inserted into an active finding.

## Checks and limitations

The ordinary draft build exits zero. The existing layout checker puts the main-text endpoint on
page **6**, with all three main floats before that endpoint and all four appendix floats after it.
The PDF has 11 pages including references and appendices. The original abstract checker reserves
84 words for future findings: 89 fixed words plus that reserve gives a conservative maximum of
**173/200 words**. This is not a promise that arbitrary later result text will fit.

The final log has no overfull boxes or unresolved-reference/citation warnings; all fonts are
embedded and none is Type 3. Pages 1, 2 and 4 were rendered and visually inspected, including both
equations and the unchanged conceptual figure. The first successful draft build had a 6.80-point
overfull equation; its source/PDF/log/auxiliary file are retained in [draft-attempt-1/](draft-attempt-1/).
Introducing a document-difference symbol fixed the layout without changing the identity or font size.

The existing language helper reports no document-language problems. All **12 pending marker
definitions** remain in the three active generated includes (5 primary, 4 dimension, 3 factorial).
The result files, bibliography, conceptual figure, style files, build rules, numerical sources,
scientific protocols, and frozen dispatch/authorization sources were not edited. No strict release
build, training launch, evaluation restart, remote write, or source publication was performed.

One inspection initially requested a nonexistent layout JSON, and one helper import used an
incorrect function name and exited 1. These were inspection mistakes, not manuscript build or
experiment failures; actual outcomes are retained in [commands.json](commands.json). A previous
multi-hunk prose patch was rejected without applying any hunk; the correctly ordered patch was
then applied. No failed numerical evidence or historical gate was reclassified.

## Evidence and continuation

[Before-bindings](before-bindings.json), [verification](verification.json), and exact [before-copies](before/)
preserve the preceding manuscript and handoff. The earlier six-complete-evaluation archive remains
unchanged; its old manuscript and CURRENT_EXPERIMENT bindings resolve to these before-copies.
The final reviewed source and PDF are retained in [draft/](draft/).

At **2026-09-10 15:46 UTC**, the original observer reports **107/840** primary BEIR task cells and
eight live exact workers. Validation is **7/12** with its coordinator live; functional analysis is
**0/61**, with its coordinator live and waiting for validation. See the separately timestamped
[observations](observations.json). There is no new optimizer ranking or dimension-use result here.

Continue the existing full evaluation and functional pipeline. Do not retrain the twelve completed
primary runs, replace the fixed features or inference rules, install partial findings, or infer
approval for the unanswered one-GPU priority request. The [current handoff](../../../CURRENT_EXPERIMENT.md)
and unchanged [AGENTS.md](../../../AGENTS.md) govern further work.

From the repository root, the documented draft command is:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 make -C paper PYTHON=/usr/bin/python
```

This compiles existing includes. It is not the strict final-publication audit and does not generate
or validate the unfinished retrieval, dimension, or continuation findings.
