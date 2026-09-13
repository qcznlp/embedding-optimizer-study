# Scoped native training restoration

The `embed_optim_restore` add-on contains the counter-placement and reducer
restoration procedure used in the two successful exact endpoint checks. It is
a separate import namespace so it can coexist with the original, separately
versioned `embed_optim` training assembly without replacing its numerical code.
It has no producer-directory fallback, scheduler, GPU acquisition or network call.

## Verified scope

Only these genuine continuation saves are supported:

| Case | Original run | Restore → endpoint |
| --- | --- | --- |
| `adamw` | `factorial-v3-adamw_state-adamw-seed314159` | 313 → 391 |
| `muon` | `factorial-v3-adamw_state-muon-seed314159` | 313 → 391 |

The original four-rank tests compare all 134 model tensors, complete optimizer
and scheduler states, all four saved rank RNGs and the selected Trainer counters
bit-for-bit after 78 resumed updates. Both pass; their earlier failed attempts
remain in [the recovery evidence](../reports/engineering-archive/dense-v3-warm-reducer-recovery-v1/README.md).
This is not NorMuon, all-checkpoint or physical-second-host verification.

## Inspect the packaged references

From an installed wheel or checkout:

```bash
CUDA_VISIBLE_DEVICES='' python -m embed_optim_restore --case adamw --rank 0
```

This authenticates the bundled cold/warm reference bytes and prints their digests
and the supported checkpoint identity. It does not import the study training
package or initialize Torch/CUDA. Reference JSON files contain fingerprints and
the exact next four local batches' row indices, not large gradient arrays.

## Use inside an admitted four-rank worker

First restore the actual model/data/calibration/runtime artifacts following the
[checkpoint](checkpoint-restoration.md) and [continuation](continuation-probe-restoration.md)
guides. Construct the original source-bound `RunBoundFactorialTrainer` with its
native `resume_binding` and the original unchanged full-horizon arguments.
The caller must already own the required GPU leases and the four-rank process
group; this add-on does not replace native source or resource admission.

Replace the worker's direct `trainer.train(resume_from_checkpoint=...)` call with:

```python
from embed_optim.factorial_v3_bound_trainer import collective_phase
from embed_optim_restore import resume

train_output, restoration_report = resume(
    trainer,
    checkpoint="/absolute/local/path/to/checkpoint-313",
    case="adamw",  # or "muon" for the other declared save
    collective=collective_phase,
)
```

Each rank authenticates the declared checkpoint identity, collates the recorded
next four batches through its native dataset/collator, and checks their token
fingerprints. After the native checkpoint loaders run, the adapter checks actual
loaded state, places named Adam counters with their parameters, and restores the
warm reducer at the first clipping boundary. Exact local/post-DDP reference and
RNG checks must pass before native clipping and updates can proceed. A failure
stops the call; it is not retried or converted into fresh training. Handlers are
removed on exit, including exceptional exit.

The returned report is **not** an independently verified endpoint comparison:
`endpoint_compared` remains false. Compare the resulting saved states separately
with the original complete endpoint and keep the diagnostic output distinct.
Do not overwrite or upload it over the scientific checkpoint. The supported
reference is deliberately immutable; do not regenerate it from a failed attempt
or add a tolerance to make another environment pass.

## Source and portability boundaries

The four helper computations and exact recursive comparator retain their original
executed computation syntax trees (the trace import is package-relative;
standard-library import ordering and docstring indentation are normalized explicitly). New integration
tests exercise handler cleanup, failure refusal and actual CPU autograd through
the pinned native `Trainer.training_step`. Those are not new GPU equivalence runs.

The original numerical training assembly and the current analytical consumers
still have explicit source-version integration requirements. Installing this
add-on does not make an incompatible `embed_optim` assembly satisfy an old source
contract, erase producer paths from saved provenance, or provide a complete new
host training launcher. Use the original admitted trainer; never rewrite saved
identities or bypass its refusal. Current source-release status remains in
[the handoff](../CURRENT_EXPERIMENT.md).
