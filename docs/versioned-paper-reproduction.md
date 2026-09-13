# Complete results to the reviewed paper, with isolated source versions

The current training package and the original numerical consumers bind different
versions of `config.py` and `optimizers.py`. Neither version is silently replaced.
`embed-optim-reproduce-paper` runs the original authenticated analytical closure
in a fresh process, then joins its complete result inputs to the reviewed paper
and builds that paper using the installed current document component.

This is an executable source-version composition, not an exception to an old
source lock. The old preparation/publication contracts retain their original
identities and refusal behavior. The factorial statistical functions execute
from their original closure; this does not grant fresh-training admission to a
changed factorial worker or rewrite saved model provenance.

## Run

Obtain the entire accepted numerical bundle as described in
[complete numerical reproduction](paper-results-reproduction.md#complete-paper-replay).
Its manifest SHA is
`746a545d3e59d86aab8d4ff6579abca3be9884f1ea824a702e4fa28bc85956b7`.
The command requires that exact 189-file bundle, not an edited manifest or
arbitrary external Python code. It also requires the twelve-file reviewed
`paper/current` snapshot. At present the bundle is local delivery material;
remote source publication is tracked separately in the current handoff.

From a checkout with the recorded CPU numerical runtime and TeX Live:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /usr/bin/python -B \
  -m embed_optim.paper_reproduction \
  --numerical-bundle /absolute/path/to/accepted-closed-bundle \
  --paper-dir /absolute/path/to/checkout/paper/current \
  --output /absolute/path/to/new-output
```

An installed wheel supplies the equivalent `embed-optim-reproduce-paper`
command. Inputs and output must be ordinary local paths, and the output must
not exist; its parent must exist. No output or failed attempt is overwritten.
Use the recorded separate environment; do not upgrade the experiment runtime.

The package invokes the original numerical entry from its required component
working directory with an empty `PYTHONPATH`, no visible GPUs and one numerical
thread. It retains original producer-directory and Python network refusals,
source inventories and exact output comparisons. The first phase reconstructs
all primary outcomes, weight/functional predictors and exploratory controls,
six factorial tables, independent arithmetic/bootstrap checks, result includes,
empirical maps and the original complete-result PDF.

The second phase verifies the eight shared result/figure inputs against the
reviewed snapshot before compiling it. Seven inputs are numerical outputs;
the conceptual map is a preserved design illustration, not a newly measured
finding. The reviewed prose and bibliography are authenticated separately.
All original strict document checks apply: expanded abstract, main-page limit,
references, layout, fonts and exact compiler input identities.

## Outputs and scope

- `numerical/`: original complete reconstruction and its unchanged receipts;
- `reviewed/paper/build/main.pdf`: reviewed complete-result manuscript;
- `numerical-exit.json`, `numerical.log`: actual original child execution;
- `complete.json`: explicit source roles, both phase bindings and eight-input join;
- `failed.json`: retained failure, if any; no automatic retry.

Completion means full numerical-to-reviewed-document reproduction. It does
not rerun training, model encoding, coordinate deletion or native model
admission; their complete genuine upstream evidence remains preserved. It does
not claim a physical second-host run, GPU recovery, remote publication or
automatic manuscript installation. The parent `make release` now invokes this
complete numerical/document command, with explicit `NUMERICAL_BUNDLE` and new
`RELEASE_OUTPUT` arguments; its actual execution has passed. The original
historical Make source is preserved in `paper/legacy.Makefile`, and its guards
remain accessible through `make legacy-release`. Neither target publishes
remotely or substitutes for the remaining complete source-version test audit.
