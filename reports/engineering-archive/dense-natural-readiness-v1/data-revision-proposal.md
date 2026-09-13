# Proposed common-data revision — not executed or released

The existing training/validation snapshots and all their protocols remain immutable.
This proposal addresses only the five empty training groups and one empty validation
group found by outcome-independent text inspection. It does not alter the optimizer
grid, objective, temperature, negatives per query, context limit or BEIR estimand.

Proposed deterministic policy:

1. Keep all **499,995 eligible training rows** and **4,095 eligible validation rows**
   identical in every original value and ID. Record per-row canonical digests. Changed
   Arrow serialization is a new dataset, not evidence of changed retained row contents.
2. Replace each of the six ineligible *whole query groups* from the same source split,
   at its original sample position. Never invent text, promote a negative to positive,
   or choose an alternative labeled positive solely to keep an old query ID.
3. Use the original source revision and original seeded query-priority permutation:
   training seed 42; validation seed 20260826. Take the next unused eligible query in
   that deterministic order, preserving source quotas and totals 500,000 / 4,096.
   Keep the existing 0.95 negative threshold, ten-candidate pool, seven-of-ten sampling
   and per-query seed. Accept only complete nonempty nine-text groups with seven unique
   negative IDs excluding the positive; record exact-text eligibility as well.
4. Reserve **all original training and validation query IDs**, including removed ones,
   before either replacement pass. Prepare training replacements first, then reserve
   their new IDs for the validation pass. Check every declared held-out/analysis partition
   before adopting new training IDs. Recheck full training/validation disjointness, not
   only the changed rows. Never select replacements using loss, BEIR or mechanism outputs.
5. Write separate training and validation roots, parent identities, six exact before/after
   records, skipped-candidate reasons and new manifests. Repeat full row/content/source
   checks; reproduce the materialization independently. Preserve every old file/receipt.
6. Review the explicit protocol amendment and all consumers together. The frozen v2
   primary proposal and existing validation selector intentionally reject changed data;
   do not replace their expected hashes or pass new data under an old identity.
7. Only then execute a bounded, declared **seven-source** real-data diagnostic from the
   immutable untrained DenseOn base, covering complete and final partial accumulations.
   The preserved failed FIQA-prefix attempt is not erased or retroactively made a success.

The candidate ordering, source eligibility and protected-partition checks above still
need implementation/verification; no replacement query has been selected in this turn.
If a deterministic candidate pool is insufficient or another partition conflicts, retain
the failure and review the next action rather than silently changing the seed or quota.

This amendment is preprocessing governance, not a scientific optimizer finding. Its
effect on optimizer results is unknown. The numerical correction and formal source/
runtime handoff remain separate prerequisites; resolving six text records does not
clear those gates or establish a complete NAACL experiment.
