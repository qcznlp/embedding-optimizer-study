# Current complete-result paper

This directory is the stable local source for the complete-result manuscript,
including all four generated result includes and all three external figures.
It reproduces the reviewed 2026-09-13 revision. The parent `paper/main.tex`
and its release target retain the historical source-transition gate; they are
not this current manuscript. Local integration is not remote source publication.

From a repository checkout with the study environment and TeX Live available:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -m embed_optim.current_paper \
  --paper-dir paper/current --output /absolute/path/to/new-paper-build
```

Or use `make -C paper current CURRENT_OUTPUT=/absolute/path/to/new-paper-build`.
The output must not already exist, and its parent must exist. Failed or older
builds are retained; choose a new directory for the next attempt. The resulting
PDF is `new-paper-build/paper/build/main.pdf`.

The command authenticates every input against the reviewed snapshot, compiles
with shell escape disabled, and checks actual abstract expansion, the eight-page
main limit, every main/appendix float, compiler input identities, bibliography,
unresolved references and embedded fonts. It uses no GPU, network, model/data
download or producer-directory fallback. Its success is **document reproduction**,
not a rerun of training/statistical inference or a full source-release verdict.

The snapshot is fixed deliberately: editing a result, figure, manuscript or
bibliography invalidates this reproduction command. For prose development,
copy this folder elsewhere and run ordinary LaTeX there; a new reviewed snapshot
is required before a changed manuscript can pass this reproduction entry.

For provenance and the previously inspected PDF, see
`reports/paper-review/dense-v3-complete-manuscript-revision-v1` in the repository.
Complete experimental results and HF restoration instructions are linked from
the repository README and `docs/continuation-outcomes-restoration.md`.
