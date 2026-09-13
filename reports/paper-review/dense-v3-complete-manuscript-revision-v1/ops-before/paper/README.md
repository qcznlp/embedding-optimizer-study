# NAACL manuscript

This directory contains the authoritative DenseOn paper for the AdamW/Muon/NorMuon optimizer
study. The paper asks one question: starting from the same pretrained retriever, how do the
optimizers change weight space, and which changes are associated with more useful ranking
representations? The distinction is between spreading weight motion and distributing helpful
retrieval evidence, not whether Muon has its characteristic update spectrum.

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
   removal, helpful mass share, normalized helpful participation, degrading mass, and shared rotations;
4. require candidate state measurements to predict retrieval when a learning-rate dose is held out;
5. cross two fixed AdamW- and Muon-reached states with reset continuations to measure averaged
   endpoint contrasts and state-dependent continuation responses.

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

The three co-primary dimension features use positive--hardest-negative margin. First average each
coordinate's deletion-and-renormalization effect over queries within a task; only then separate
positive and negative effects. Helpful share is H/(H+B), helpful participation is
H²/(768 × sum of squared helpful effects), and degrading mass is B. Zero denominators map to zero
under the unchanged definition. Average task-level summaries, not individual-query summaries.
The nine final-stage feature-by-optimizer contrasts share one fixed-SE max-T family. Analogous
shortlist-nDCG attributions remain descriptive, not substitute primary endpoints.

Weight-displacement magnitude uses a ratio of joint Frobenius norms, not a mean of layer-wise
ratios. The original approximate geometry and full-spectrum sensitivity are separate measurements:
full-spectrum entropy is not renormalized truncated-spectrum entropy, and full-spectrum rank
summaries explicitly use nonzero parameter weights. Retain both branches and all their candidates.
The margin decomposition in the introduction is a score identity, not a novel theorem or an
additive interpretation of deletion sensitivities. Held-out dose is not held-out task.

Agreement under three sampled rotations does not establish invariance under arbitrary bases.
The crossed reset experiment does not directly manipulate a measured geometric feature, so its
effects cannot by themselves establish that feature as a causal mediator.

The crossed continuation uses fixed AdamW 3e-5 and Muon 3e-4 states at 60% training progress,
resets optimizer history, matches hidden update scale on a fixed calibration probe, and repeats
the four cells under three data-order seeds. It does not match the first realized shuffled-batch
update. The measured quantities are final 50,000-query continuation endpoints, not gains relative
to the starting checkpoints or a decomposition of the original training result.

An averaged benefit need not hold within both states or both continuation rules. Positive
interaction may mean less harm rather than beneficial continuation. The three marginal intervals
are not simultaneous coverage, and an inconclusive interval establishes neither equivalence nor
the dominance of another effect. The source pair has one primary training seed; the three
continuation orders are not independently retrained source models.

The [claim-wording amendment](../configs/dense_no_packing_state_operator_claim_wording_amendment.json)
narrows those prose implications while preserving numerical decisions and the original scientific
protocol bytes. The manuscript, result renderer and conceptual figure use this interpretation in
the isolated checkout. This is not a deployed runtime or a completed scientific paper.

If no summary or publication exists, regenerate the development-only factorial placeholder with:

```bash
python -B scripts/render_state_operator_pending.py --repository /absolute/path/to/checkout
```

This helper contains no invented scores and refuses to overwrite a non-development include or
operate after a scientific summary/publication appears. It is not part of the runtime pipeline and
cannot substitute for a completed result. The conceptual map is reproducible through
`scripts/plot_weight_space_dimension_map.py`; its PDF uses vector text without a creation-time stamp.

## Scope

The manuscript is DenseOn-only. Other model-family artifacts remain outside the paper and do not
enter its intervals, tables, or claims. See `../CURRENT_EXPERIMENT.md` for dated execution state,
`../AGENTS.md` for safe continuation instructions, and `../docs/naacl-dense-paper-plan.md` for the
result-contingent narrative map.
