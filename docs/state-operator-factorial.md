# Corrected DenseOn state-by-operator factorial

This follow-up asks why the historically strong Muon trajectory can finish better even when an
isolated, norm-matched local step is not universally better than AdamW. It separates two possible
causes in a crossed experiment:

- the **weight state** reached by the source trajectory: AdamW 3e-5 or Muon 3e-4 at 60%; and
- the **continuation operator** applied after resetting optimizer state: AdamW or Muon.

The fixed design is 2 states × 2 operators × 3 data-order seeds = 12 short DenseOn runs. Every run
uses the same deterministic 50K intact-group branch view, seven explicit negatives, no in-batch
negatives, 8,192-token context, global batch 128, independent padding, and five checkpoints. The
final checkpoint is evaluated on all 14 full-corpus decontaminated BEIR tasks. All five checkpoints
are evaluated on the balanced 224-query unseen probe.

The scientific decisions are frozen in
[`dense_no_packing_state_operator_factorial_protocol.json`](../configs/dense_no_packing_state_operator_factorial_protocol.json).
The executable source hashes, cardinalities, output namespaces, and commands are frozen separately
in
[`dense_no_packing_state_operator_factorial_implementation_protocol.json`](../configs/dense_no_packing_state_operator_factorial_implementation_protocol.json).
Do not execute a changed implementation against this design.

The paper-only decision renderer is frozen in
[`dense_no_packing_state_operator_factorial_publication_protocol.json`](../configs/dense_no_packing_state_operator_factorial_publication_protocol.json),
and the resumable operational handoff is frozen in
[`dense_no_packing_state_operator_factorial_completion_protocol.json`](../configs/dense_no_packing_state_operator_factorial_completion_protocol.json).
The latter waits for the exact main corrected completion ledger before any factorial GPU work,
backs up both five-checkpoint runs after every training wave, evaluates the three declared BEIR
pairs concurrently, and reruns the manuscript release gates.

## Readiness gate

Do not begin until the requested source checkpoint is both deeply resumable and remotely verified
by its sealed-checkpoint receipt. The two source checkpoints are:

- `outputs/dense-no-packing-v1/dense/padded-adamw-3e-5/checkpoint-2345`
- `outputs/dense-no-packing-v1/dense/padded-muon-3e-4/checkpoint-2345`

The gradient and direction commands acquire the shared GPU lease. They record every source
checkpoint input by byte count and SHA-256, require `can_flatten_inputs=false` to be observed by the
loaded model, and refuse pre-existing untagged artifacts.

## Execution

For unattended execution, use the source-bound controller from the repository root:

```bash
python -m embed_optim.state_operator_factorial_completion --resume
```

It is safe to start this command while the main corrected study is still running: it holds only its
own control-plane lease while waiting and requests no GPU lease until the main 12-run evaluation,
analysis, publication, and release ledger is complete. The commands below document the frozen
manual sequence and are not a second controller.

Audit the frozen contracts and portable data receipt:

```bash
uv run embed-optim-state-operator-factorial audit-protocol --portable-data-audit
```

For each ready source state, export the fixed 32-example gradient history and compute only the two
global direction norms needed for scale matching:

```bash
uv run embed-optim-state-operator-factorial calibrate-gradients \
  --state adamw_state --gpus 0 --device cuda:0
uv run embed-optim-state-operator-factorial calibrate-directions \
  --state adamw_state --gpus 0 --operator-device cuda:0

uv run embed-optim-state-operator-factorial calibrate-gradients \
  --state muon_state --gpus 0 --device cuda:0
uv run embed-optim-state-operator-factorial calibrate-directions \
  --state muon_state --gpus 0 --operator-device cuda:0
```

Generate and audit the six two-run matrices:

```bash
uv run embed-optim-state-operator-factorial generate
uv run embed-optim-state-operator-factorial audit
```

Run the matrices in the `training_order` stored in the implementation protocol. One command uses
both four-GPU pools and trains the reset AdamW and reset Muon continuation concurrently:

```bash
uv run embed-optim-train-state-operator-factorial \
  --state adamw_state --seed 314159 \
  --gpus-a 0,1,2,3 --gpus-b 4,5,6,7 \
  --port-a 29710 --port-b 29720
```

Repeat with the other five exact state/seed pairs from the implementation protocol. The dedicated
wrapper rechecks the implementation lock and source-checkpoint content, holds both GPU leases, and
does not return complete unless both runs and all ten scheduled checkpoints pass the specialized
deep audit.

Evaluate all five stages on the unseen probe:

```bash
uv run embed-optim-probe-state-operator-factorial
```

Evaluate final full-corpus BEIR. Two state/seed commands may run concurrently on disjoint four-GPU
pools; the result roots and leases are disjoint:

```bash
uv run embed-optim-evaluate-state-operator-factorial \
  --state adamw_state --seed 314159 --gpus 0,1,2,3
uv run embed-optim-evaluate-state-operator-factorial \
  --state muon_state --seed 314159 --gpus 4,5,6,7
```

After all six evaluation cells and the 61 probe jobs pass their audits:

```bash
uv run embed-optim-summarize-state-operator-factorial
```

The completion controller then runs the source-bound paper renderer, rebuilds the ACL PDF, refreshes
the portable evidence closure, and repeats strict paper, test, and style audits. The rendered result
enters the abstract, mechanism section, main Conclusion, and an appendix table. The ACL manuscript
is the only publication artifact.

## Interpretation

For every seed–task cell, the summary reports:

- the weight-state main effect, averaged over continuation operators;
- the continuation-operator main effect, averaged over source states; and
- their interaction.

Each of the three co-primary estimates uses the frozen 100,000-replicate two-way seed/task cluster
bootstrap. A positive or negative effect is supported only if its 95% interval lies wholly on the
corresponding side of zero; otherwise it is inconclusive. The exact 126 seed–task contrasts remain
available regardless of the interval result.

This experiment can support a carried-state, continuation-operator, or state-feedback interaction
account for this checkpoint pair and branch horizon. It cannot prove a universal Muon mechanism or
retroactively turn the historical crossover into confirmatory evidence.
