# Remaining predeployment checks: fresh-process replay and complete run identity

The previous goal turn made concrete progress: the numerical candidate was implemented and
passed optimizer, gradient, checkpoint and real-entrypoint diagnostics. This turn completes its
separate-process replay and establishes the remaining full-recipe admission boundary with actual
counterexamples. No formal training, deployment, publication or controller transition is implied.

## Results and interpretation

| Surface | Evidence | What is not claimed |
| --- | --- | --- |
| New candidate, new process group | **24/24 exact** complete continuation fingerprints; no new baseline | Default DDP bitwise guarantees, a new host, natural-data horizon or retrieval evidence |
| Current actual resume admission | **30 changed-recipe/data/runtime cases accepted** before setup; 6 negative controls correctly rejected | That any historical run actually used the wrong data/configuration |
| Proposed full identity comparator | **42/42 comparisons**: six original/relocation positives and 36 changed-identity negatives | Integration into Trainer, a persisted full-run contract, or formal readiness |
| Isolated regression suite | **1,415 tests**, zero failures/errors/skips, no exclusions | Acceptance of the separate candidate's 923-case suite with its 11 unwaived protocol failures |

These are engineering-only results. They do not enter the manuscript or support optimizer-quality
claims. The 12 completed primary runs and 60 preserved checkpoints remain on scientific hold.

## Fresh process, same sealed candidate

[The fresh-process result](fresh-process/result.json) authenticates the preceding candidate's
[source manifest](../dense-correction-candidate-v1/prepared-source-v2.json) and baseline receipt
before reading checkpoint payloads. Its source is unchanged in
`/tmp/dense-correction-source.0YHywN`; neither original source checkout was numerically modified.

The baseline process group is `9174d58f-4230-4e1b-add2-3a9f6f76ebcc`. The newly launched group is
`9bb8a85c-53de-442b-9e96-d8f073435971`. All four rank identities are recorded. It resumes the
three algorithms from checkpoints 1 and 2, without training another baseline. All serialized
entry/intermediate/final states, gradients, row order and Python/NumPy/CPU/CUDA RNG fingerprints
match the original complete replay fields exactly. Eight persistent workers per rank are used.
The final report was sealed at **2026-09-06 09:55:29 UTC**; the launcher exited zero.

Deterministic owned attention backward and one lexical-vector reduction are explicit diagnostic
controls. No old source guard, comparison field or tolerance was weakened. The previously observed
default-DDP reduction-layout differences remain preserved and are not reclassified as fixed.
The new numerical candidate now has both same-group and independent-group controlled replay
evidence, but this is not a performance recommendation or universal replay guarantee.

## The confirmed missing admission contract

[The read-only admission probe](admission/result.json) starts with actual saved metadata from
the preceding unpatched entrypoint runs. Nine configuration/numerical-contract/training-argument
files are authenticated against the prior evidence receipt before the training-argument pickle
is deserialized. No model/optimizer weights are loaded by this probe.

Each call goes through the real candidate `run_training` admission code, but is intercepted at
the following `_training_arguments` boundary. Random seeding, shared output writes, dataset/model
loading and all actual updates are prohibited. Thus a changed configuration can be tested without
ever executing it or overwriting a source checkpoint.

For each of AdamW, Muon and NorMuon, the current numerical-only gate accepts all ten changes:

- seed, dataset contents, epochs, maximum context and temperature;
- clipping, global/micro batch together while accumulation stays four, and warmup;
- declared base-model revision and the runtime maximum-step override.

Changing the optimizer learning rate or the accumulation divisor correctly rejects, confirming
that the existing numerical contract is active. The diagnosis is **a missing full-run identity
gate**, not another gradient/optimizer defect and not proof of historical configuration drift.
The actual implementation was deliberately a bounded numerical candidate; it was never declared
whole-stack ready. Both admission campaigns exit **1**, retaining that failed readiness decision.

## Tested proposal, still not integrated

The [identity proposal source](sources/dense_run_identity_proposal.py) compares the full resolved
scientific recipe, observed dataset file digests/row count/selected columns, effective training
arguments, numerical policy and source identity. It preserves every supplied scientific field,
instead of using a hand-selected subset. Canonical JSON rejects nonfinite values and distinguishes
booleans from numbers. Dataset files are hashed with a before/after stability check; symlinks and
empty data inventories are rejected.

Only output/data directory locations and W&B project/entity are excluded from recipe equality;
data **contents** remain bound separately. Byte-identical relocation therefore passes, while a
same-length content change cannot hide behind the same row count. This addresses the intended
other-machine reconstruction workflow without treating path identity as data identity.

All 42 proposed comparison inputs are preserved in the second admission receipt and recomputed
by the independent evidence validator. It includes six positive controls and all 36 negative
comparisons. These identities were assembled for this diagnostic from authenticated metadata
and the presently observed synthetic dataset files. No full-run receipt was retrofitted into an
old checkpoint, and no retrospective primary-data digest claim is made.

**This is not yet a production guard.** The caller must still authenticate metadata and obtain
identities from actual inputs. The next integration must seal a complete identity at run creation,
bind each checkpoint to it, and reject mismatches before any resume side effect. It must enforce
the new source-bound primary protocol, not merely trust a caller's identity dictionary. Its actual
save/resume integration must be tested before formal deployment.

The first [admission receipt](admission-first/result.json) and exact auditor source are retained.
The second run adds the replayable comparator inputs and data inventories; it does not change the
30/6 admission result or relabel the failed gate. Combined direct-tool output is explicitly
labeled as such, rather than represented as a separately captured original stderr file.

## Preservation and continuation

[commands.json](commands.json) records exact launch commands and observed exit codes. All eight
fresh-process rank logs, four control observations, source files and test receipts are preserved.
The previous candidate archive and its 93 bound artifacts remain intact; superseded top-level
documents are copied under [before/](before/) and [before-live/](before-live/).

Large diagnostic checkpoint and fingerprint payloads remain in their named temporary roots and
are not uploaded/deleted. The evidence validator rechecks the actual full fingerprint equality,
source and checkpoint digests, original core files, exact stopped dispatcher handles and unchanged
ledger. It refuses optimized Python and does not convert an admission failure into acceptance:

```bash
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
python -m scripts.validate_dense_predeployment_checks \
  --repository /root/embedding-optimizer-story-refactor --output /tmp/new-predeployment-validation.json
```

Next integrate the complete run/checkpoint identity into a newly reviewed execution contract,
exercise its real persistence/admission path, and update downstream source bindings without
changing scientific estimands. The old eleven protocol gates remain unwaived; the routed
factorial operator needs separate integration. Do not repeat the now-completed fresh-process
diagnostic or the already localized old reduction-origin investigation.

The owner has not yet answered the request to permit development-stage commits before all paper
results exist and to authorize the reviewed formal replication. The earlier GitHub integration
403 is not bypassed or retried; this milestone remains local. BEIR's same three dispatchers remain
paused, with the original lease and four-step completed prefix. No old zero-step migration, HF
deletion, protected GPU-helper interaction or formal training was performed. No new scientific
finding was added to the paper.
