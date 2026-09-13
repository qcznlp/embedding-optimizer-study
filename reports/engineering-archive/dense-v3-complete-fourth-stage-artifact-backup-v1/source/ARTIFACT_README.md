# Complete corrected DenseOn fourth-stage BEIR results

This immutable, data-only snapshot preserves all twelve declared optimizer/rate
configurations at step 3126 (the 80% checkpoint), with fourteen tasks each.
It contains 168 native task scores, twelve exact macro means, 216 native result/
metadata/acceptance files and 336 original worker start/exit records.

All states belong to the corrected primary v3 protocol
`4400f1ce26ef423873a4eb509db48e046a07b90159f6d5ef10da084f5171107b`.
The complete parent readback covers all sixty checkpoints / 840 task scores;
this snapshot deliberately selects only the twelve fourth-stage states.

Six original pool-B states overlap the previous partial-fourth-stage snapshot.
Only six original pool-A states / 84 scores are newly backed up. Overlapping
states are not new runs, independent replications or additional statistical evidence.
The previous snapshots are unchanged. Together they preserve all sixty checkpoint
outcomes, while the separate model repository preserves the model/training states.

Native result and worker bytes are copied unchanged. The CSVs are reconstructed
from the accepted original results, retaining every declared learning rate and task.
The means use an exact rational average of the original binary64 task scores.
Any producer-machine paths inside receipts are provenance, not recovery locations.

This snapshot does not include source code, training text, model weights, embeddings,
the aggregate bundle containing embedded source, or a new statistical conclusion.
It is not functional-analysis completion, a crossed continuation, manuscript
admission or a complete reproducible source release. Full-trajectory tables and
figures have a separate pending backup at this milestone.

Download this exact content-addressed prefix at its recorded immutable revision;
authenticate `artifact_manifest.json` against the external SHA-256 from the trusted
handoff. Verify every payload hash, then reconstruct native raw scores and exact
means with the separately supplied local `verify_recovered.py`. That reader needs
only the Python standard library and the supplied snapshot; it does not execute
producer paths, load a model or access a GPU. Code is not bundled in this data-only
addition. Anonymous same-host recovery is not a physical second-host model rerun.
