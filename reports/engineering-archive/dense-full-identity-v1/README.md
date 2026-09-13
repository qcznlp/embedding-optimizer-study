# Complete run identity integrated into actual Dense checkpoint save/resume

The previous turn identified a missing complete-run admission gate. This turn **implements it
in a distinct prepared source tree**, then exercises the actual four-GPU entrypoint and its
checkpoint persistence/restore. It does not deploy code, launch a primary run, accept old results
or authorize publication. No engineering detail here belongs anywhere in the manuscript.

## What passed

- **Six actual full-model `run_training` calls:** baseline plus checkpoint-2 continuation for
  AdamW, Muon and NorMuon. Data and initial model are copied byte-for-byte to different directories
  for each continuation; the output namespace is also different.
- **Twelve complete newly produced diagnostic checkpoints:** each has 20 files, including the
  model, optimizer, scheduler, four rank RNG files, numerical contract, complete run identity and
  payload seal. All seals pass; each final inference model equals its final scheduled checkpoint.
- **Forty-five actual admission decisions on authenticated producer files:** six original or
  byte-identical relocation controls accepted; all 36 changed recipe/data/runtime controls rejected;
  three separately copied real optimizer payloads with a one-byte corruption rejected before
  deserialization. No update, model load or pickle load occurs in this CPU admission probe.
- **79 targeted candidate tests pass.** Its complete suite has 963 cases, 952 passing and the
  same **11 unwaived old source-contract failures**, zero errors/skips. This is not a green
  whole-repository candidate. Neither source locks nor scientific criteria were relaxed.
- The different isolated paper/audit checkout passes **all 1,424 regression cases**, zero
  failures/errors/skips. Its receipt must not be presented as acceptance of the candidate checkout.

The entrypoint launcher exited zero at **2026-09-06 10:47:01 UTC**. It uses a new torchrun group
`d124cb11-11c5-485f-9384-f04029a52604`, ordinary DDP communication, the actual BF16/FA2 DenseOn
model and eight persistent workers per rank. It starts from an independently digest-authenticated
trained checkpoint **only as a diagnostic fixture**, fresh optimizers, 288 predetermined synthetic
rows and a three-step horizon. This is not a primary run from the untrained base, an 8192-token
natural-data preflight, a different physical machine, or a bitwise trajectory-replay claim.

## Implementation and admission boundary

The complete source is copied under [candidate-source/](candidate-source/); its live preparation
root is `/tmp/dense-identity-source.8rUgLF`. [prepared-source.json](prepared-source.json) binds all
14 selected files and verifies that the older `/tmp/dense-correction-source.0YHywN` source remains
unchanged. Optimizer algorithms, numerical policy, dataset recipe and 12 candidate configurations
are byte-identical to that parent. Only the training integration, new contract module and its
tests change. The original two project checkouts' six numerical files remain unchanged.

The contract is derived inside the real entrypoint from the resolved configuration, pure requested
training-argument values, actual dataset file digests/selected columns/row count, initial-model
file digests, declared world size and selected precision environment settings, nine local source
files, ten installed package versions and the already pinned numerical upstream sources.
It is not a caller-supplied dictionary accepted as a formal source lock. The later primary
execution protocol must still authenticate and constrain the allowed scientific recipes/source.

Data/output/W&B destination paths may change; a local initial-model directory may relocate as
well. Contents, scientific settings and numerical/runtime/source identities must agree. Remote
initial models require a full immutable revision and its already populated local HF snapshot;
on another machine that pinned cache must be downloaded and verified before read-only admission.
This preserves a no-network-write/no-device-setup rejection boundary. Cache blob links are accepted
only inside the named model cache; dataset/checkpoint symlinks are rejected.

All identity comparison and checkpoint payload verification happen before TrainingArguments
initialization, seeding, W&B setup, output writes or model loading. After arguments initialize,
their actual values/world size must still match. All ranks compare the same complete identity
before shared artifacts are created. The old double-normalization fix remains unchanged.

Checkpoint save requires complete admission; a bare numerical-only Trainer cannot save one.
It checks the intended output is absent, synchronizes all ranks before the parent save, then
synchronizes again after every rank's RNG write. Rank zero writes the complete identity and a
content inventory, followed by a final barrier. A missing rank payload cannot be sealed. Actual
model and optimizer restore both recheck identity and every payload digest before their parent
load functions. Existing output is not silently adopted; completed/later checkpoint evidence may
not be overwritten. In-place continuation of an incomplete compatible run preserves original
metadata. A partial or conflicting namespace must be handled through a reviewed preservation/
recovery workflow, not by overwriting it.

These digests detect accidental substitution and drift; **they are not digital signatures**.
Untrusted remote downloads still require the independent trusted-digest backup audit. No complete
identity or new seal was retrofitted into an old primary or predecessor diagnostic checkpoint.
The old 24/24 independent-process bitwise receipt belongs to the older source, not this new
identity integration. The present evidence directly covers real save/load and admission, not a
new cross-process bitwise claim. All earlier failed/control receipts remain preserved.

## Evidence and known boundaries

- [entrypoint/result.json](entrypoint/result.json): full producer observations and exact input/
  output identities; all eight rank logs are archived alongside it.
- [admission/result.json](admission/result.json): all 45 exact decisions and 258 authenticated
  producer files. Original files remain unchanged; three corrupted *copies* remain only in
  `/tmp/dense-identity-admission.xjs6n5` and are not valid continuation inputs.
- [commands.json](commands.json): exact commands and observed exits, including failed checks.
- [tests/](tests/): targeted/full receipts and the original distributed-barrier fixture. Its first
  run failed one of 39 tests because the old fixture supplied no actual data for the new admission
  stage. The unit now explicitly stubs that separately tested stage and still tests the exact
  distributed-before-write boundary. The 11 distinct frozen-source failures remain failures.

Large new diagnostic outputs remain in `/tmp/dense-identity-entry.rCO4Oz`; the copied initial model,
original and relocated data, baseline/continuation checkpoints and finals must be preserved.
No new payload was uploaded or deleted. The read-only evidence validator independently recomputes
payload inventories, identity/step links, the three one-byte corruptions, exact source files and
the failure set, and preserves the preceding archive's 42 bindings via [before/](before/).
Its command is:

```bash
PYTHONPATH=/root/embedding-optimizer-story-refactor/src:/root/embedding-optimizer-story-refactor \
python -B -m scripts.validate_dense_full_identity \
  --repository /root/embedding-optimizer-story-refactor --output /tmp/new-full-identity-validation.json
```

## Next scientific execution work

This closes the bounded full-run identity **implementation and real persistence/admission** task.
Do not keep treating it as an unintegrated comparator or repeat the earlier missing-gate probe.
Next prepare/review the new primary execution, reload, backup and downstream source contracts,
including natural-data readiness and a state-preserving runtime handoff. Do not merely refresh
eleven hashes, adopt old scientific outputs, or bypass the routed factorial control's separate
integration. Scientific estimands, fixed data/base/grid, validation selection and claim boundaries
remain unchanged. Formal launch requires the accepted committed source and owner deployment/
development-publication direction already requested.

All owned diagnostics exited. The same original three BEIR dispatchers remain stopped with their
ledger, lease and four-step completed prefix. The 12 old primary runs and 60 preserved checkpoints
remain on scientific hold. GitHub write access still has the previously recorded 403; this local
milestone was not posted or pushed, and no alternate identity/credential was tried. The pre-commit
pending-results rule and HF deletion rejection remain unresolved. No formal training, controller
transition, evaluation resume, HF deletion or protected GPU-helper interaction occurred.
