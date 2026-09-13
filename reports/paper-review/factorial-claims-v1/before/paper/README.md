# NAACL manuscript

This directory contains the authoritative DenseOn paper for the AdamW/Muon/NorMuon optimizer
study. The paper asks one question: starting from the same pretrained retriever, how do the
optimizers change weight space, and which of those changes explain full-corpus retrieval?

The manuscript intentionally excludes implementation-debugging narrative and invalid exploratory
runs. Those records remain available in the repository for engineering audit, but they are not
scientific evidence and are never rendered into the paper.

## Story and evidence chain

The paper follows one sequence:

1. compare the complete four-rate surface for each optimizer over five checkpoints and 14
   decontaminated BEIR tasks;
2. measure the weight states reached by each trajectory through displacement, rank, row allocation,
   and subspace overlap;
3. test whether the reached representations use their 768 coordinates differently through random
   removal, helpful/degrading attribution, participation, and shared rotations;
4. require candidate state measurements to predict retrieval when a learning-rate dose is held out;
5. cross AdamW- and Muon-reached states with reset AdamW and Muon continuations to separate reached
   state, next-operator, and state-by-operator interaction effects.

Muon's orthogonalized update spectrum and NorMuon's row rescaling are operator fingerprints, not
standalone contributions. A one-step proxy cannot establish which optimizer is better. The paper's
explanatory claims require trajectory-level evidence, out-of-dose retrieval prediction, and the
crossed continuation intervention.

## Build and result safety

The LaTeX environment needs PGFPlots and its `groupplots` library (provided by
`texlive-pictures` in the TeX Live distribution). Dimension plots are native vector coordinates in
the generated include, so the publication audit recomputes the plotted values and the exact figure
source along with the tables; it does not trust a separately edited image.

Build the review-format PDF with:

```bash
cd paper
make
```

`\ResultPending{...}` markers are visible development placeholders. The paper is not
submission-ready while any remain. The final result sources are generated, not hand edited:

- `generated/optimizer-primary.tex` supplies the primary optimizer contrasts, weight-state
  summary, and complete geometry bridge;
- `generated/dimension-utilization.tex` supplies the all-rate utility figure, dimension-use contrasts, rotation control, and
  dimension-to-retrieval bridge;
- `generated/state-operator-factorial.tex` supplies the state, continuation-operator, and
  interaction effects.

Every generator validates source hashes and expected cardinalities before replacing its placeholder.
The ordinary build compiles those existing includes; it does not regenerate historical findings or
require the withdrawn experiment's model/analysis tree. Generate real primary results through the
declared analysis controller, not by editing the includes or invoking the historical headline writer.

The final gate independently reloads the primary publication's bound upstream tables and reconstructs
the exact manuscript text, standalone engineering report and manifest. Updating an edited output's
digest cannot bypass this comparison. This publication check does not repeat training or retrieval.

Use the ordinary build while evidence is incomplete. The release build requires complete generated
results and runs the strict manuscript gates:

```bash
make -C paper release
python -m embed_optim.paper_audit \
  --strict \
  --families dense \
  --scope-amendment configs/dense_scope_amendment.json
```

The final audit must establish all of the following:

- exactly 12 primary runs, 60 checkpoints, and 840 checkpoint-task retrieval units;
- no unresolved result marker;
- a maximum eight-page main paper before Limitations and Ethical Considerations;
- an abstract of at most 200 words after all generated findings expand;
- all main claims trace to content-addressed manifests;
- the dimension publication's portable closure passes source checks and recomputation of all
  statistics, decisions and exact generated LaTeX in a clean clone;
- no hidden or best-cell-only optimizer selection;
- no unsupported causal language for observational geometry or predictive bridges.

The portable dimension gate does not rerun encoding or coordinate ablations and does not validate
model payloads. The closure is created only after the full checkpoint-backed authoring audit; the
larger model and embedding archive is needed to repeat those upstream computations. Pending primary
results therefore cannot pass merely by removing the visible development markers.

## Claim boundaries

The primary optimizer comparison averages all four predeclared rates within optimizer. A
validation-selected recipe is secondary. Task-level common resampling yields simultaneous max-T
intervals for the three pairwise optimizer contrasts.

Geometry is explanatory only when it adds held-out predictive value beyond optimizer identity,
checkpoint stage, and within-optimizer dose. Coordinate attribution is basis dependent, so native
coordinate findings are checked under three shared orthogonal rotations. Predictive support is not
causal mediation.

Agreement under three sampled rotations does not establish invariance under arbitrary bases.
The crossed reset experiment does not directly manipulate a measured geometric feature, so its
effects cannot by themselves establish that feature as a causal mediator.

The crossed continuation experiment is the mechanism test. It uses one AdamW-reached and one
Muon-reached 60% checkpoint, resets optimizer history, calibrates the initial update scale, and
repeats all four state-by-operator cells under three data-order seeds. Its conclusions are bounded
to that source-state pair and 50,000-query continuation horizon.

## Scope

The manuscript is DenseOn-only. Other model-family artifacts remain outside the paper and do not
enter its intervals, tables, or claims. See `../PROJECT_STATUS.md` for live execution state,
`../AGENTS.md` for safe continuation instructions, and `../docs/naacl-dense-paper-plan.md` for the
result-contingent narrative map.
