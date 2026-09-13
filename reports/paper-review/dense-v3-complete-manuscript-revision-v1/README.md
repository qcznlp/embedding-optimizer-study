# Complete-result manuscript revision

Reviewed 2026-09-13. Read the [complete PDF](actual/paper/build/main.pdf)
or [LaTeX source](actual/paper/main.tex). This is a reviewed local manuscript
candidate, not an installed authoritative paper, source release or submission.

The narrative now distinguishes **the weight state an optimizer reaches from
the update rule used to continue from it**. The introduction, continuation
interpretation, discussion and conclusion connect the completed results:

1. The tested rate-grid average is inconclusive; validation-selected recipes
   answer a separate question and give higher Muon-family point estimates.
2. Weight-space separation and native-coordinate participation do not establish
   a recipe-robust geometric explanation of retrieval.
3. In the fixed crossed continuation, the averaged Muon-source state contrast
   is positive while the reset-Muon operator contrast is negative. Interaction
   is inconclusive. This is a state-level result, not mediation or a claim
   that one AdamW step explains the primary Muon result.

The revision incorporates all five deferred corrections: the DenseOn-specific
AdamW attribution; distinction between grid-average and selected-recipe
estimands; AdamW's upper-grid selection boundary; complete primary recipes
with different auxiliary rates; and Jiacheng You's bibliography name order.
It integrates the six previously reviewed closest references, without a new
priority claim. The continuation paragraph explicitly retains the primary
source-history/auxiliary-rate difference after continuation uses common routing.

## Verification

- The twelve original document inputs were authenticated against the completed
  native document receipt before copying. Only `main.tex` and `references.bib`
  changed. All four generated includes, three external figures, constants and
  vendor files are byte-identical to the completed numerical/PDF-replayed paper.
- The **unchanged** complete-document checker performs the new fresh build.
  Actual session 28526 / terminal 4ef483 exits zero: abstract **158 words**,
  main ends **page 8**, **13 total pages**, **25 embedded non-Type-3 fonts**,
  no overflow, unresolved references or active pending results. All four
  figures and nine tables remain present in their required document regions.
- All thirteen newly rendered pages were visually inspected, including every
  plot, table, reference page and the main/limitations boundary. No clipping,
  overlap or unreadable glyphs were observed. Appendix floats leave a sparse
  introductory page and a standalone final trajectory figure; these are not
  hidden missing results or evidence of submission acceptance.
- No training, retrieval, statistics, SVD, coordinate attribution or completed
  numerical replay was repeated. No engineering incident enters the manuscript.

The original native paper, its successful complete numerical replay, old
authoritative manuscript and every prior failed check remain unchanged.
The final source-level consumer/release transition and authoritative install
remain separate work; this report does not waive them or GPU recovery equality.

## Read and build

The `actual/paper` folder contains a self-contained LaTeX document and pinned
style files. A normal document-only rebuild uses a **new output directory**:

```bash
cd actual/paper
TEXINPUTS="$PWD/vendor//:" BSTINPUTS="$PWD/vendor//:" \
  latexmk -pdf -bibtex -no-shell-escape -interaction=nonstopmode \
  -halt-on-error -recorder -outdir=build-local main.tex
```

This compiles the retained generated findings; it is not a new upstream
scientific audit. `actual/document.json` records the actual strict build.
The original `prepare.py`, `build.py`, input bindings and task-only source
diffs preserve how this candidate was produced. Their producer paths are
engineering provenance, not promised portable training entry points.
