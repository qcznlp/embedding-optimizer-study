# Current DenseOn training source

The current checkout now contains the exact primary numerical implementation
used by all twelve completed primary runs. All **33 primary package modules**
and **56 source/configuration/test/parent bindings** match the retained training
snapshot. No optimizer or training calculation was rewritten during integration.
The ordinary checkout can now load the original primary contract with its own
directory as both repository and training-source root; all twelve source
identities match. Previously that required a separate numerical-source tree.

## Inspect a primary recipe

From this checkout, with the pinned runtime already installed:

```bash
study_checkout="$PWD"
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$study_checkout/src:$study_checkout" \
/usr/bin/python -B -m embed_optim.primary_v3_training \
  --protocol "$study_checkout/configs/dense_primary_v3_protocol.json" \
  --repository "$study_checkout" \
  --training-root "$study_checkout" \
  --experiment-root /absolute/path/to/your/experiment \
  --run-id verified-v3-adamw-3e-5 \
  --inspect
```

Inspection does not need a GPU or model/data loading and creates no experiment
directory. The other eleven identifiers are recorded in the primary protocol's
bound input inventory and the checkpoint restoration index. The source-code
comparison and actual one-tree inspection are recorded in the
[integration evidence](../reports/engineering-archive/dense-v3-numerical-source-integration-v1/README.md).

The preserved `dense_correctness_candidate.yaml` is a **historical preparation
fixture**, not the current v3 training matrix. Do not use its older data/output
locations to reproduce the completed study. The v3 contract determines the
current run identities and complete recipes.

## Important remaining limitations

This integration does not authorize rerunning completed experiments and does not
change the old draft/release checks. `--execute` still rejects that historical
draft; the separately authorized scientific launches remain historical evidence,
not a switch silently applied to the generic entry. Final source publication
requires the reviewed release transition.

The factorial modules import the correct primary numerics from this checkout,
but their historical 66-file source contract still binds the previous input
location module. The portable input reader gives exactly the same genuine
readback, yet its source bytes differ. That old contract correctly refuses the
mixed assembly. A separately explicit compatibility/version transition is
required; neither historical source hashes nor checkpoint identities have been
rewritten to force admission. The full factorial source-bound suite is therefore
not claimed green.

Both actual GPU endpoint-resume comparisons remain failed and unresolved.
Fresh primary-source equality and CPU tests do not establish bitwise GPU resume.
Use [the restoration guides](checkpoint-restoration.md) for immutable artifacts,
and preserve original checkpoints instead of replacing them with resumed
diagnostic states. No environment upgrade or new training is needed to inspect
the current source.
