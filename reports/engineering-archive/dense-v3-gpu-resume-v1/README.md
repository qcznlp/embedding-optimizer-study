# Genuine checkpoint recovery verification: inputs complete, GPU checks queued

At **2026-09-12 18:47:32 UTC**, both new recovery waiters are live/S, without
GPU admission or children. These are two engineering verification jobs, not
additional scientific training cells. No live scientific source or output was
changed. The original 12-cell training, 168-task BEIR and 60-probe queues retain
priority. Read the [fixed verification plan](PLAN.md) and [dated handoff](handoff.json).

The original complete-run reader freshly accepted both uninterrupted seed-314159
AdamW-source branches, one using AdamW and one using Muon. Their step-313 native
checkpoints were then **actually downloaded anonymously** at the original fixed
HF revisions: all **36 files / 3,142,951,086 bytes** matched their original hashes,
and both original deep CPU checkpoint readers passed. See [downloaded.json](downloaded.json).
This is stronger than remote metadata inspection but is not GPU resume equivalence.
It is not a physical second-host experiment or a check of the primary NorMuon path.

The real input preparation, download and 29 synthetic operational/comparison
controls all exited zero. Original tool results are in [commands.json](commands.json).
The first archive binds 30 copied source/evidence/preimage files in
[verification.json](verification.json); the final observation handoff adds six
unchanged copies. The large downloaded checkpoint payloads remain in the live
work directory and at their immutable public HF revisions; they were not duplicated
into the repository. No GPU success, scientific admission or source release is claimed.

## Live handles and next action

| Pool | Verification | Registered coordinator | Tool session |
| --- | --- | --- | --- |
| a, tokens 4–7 | AdamW checkpoint 313 → 391 | 824762 / start 321051396 | 6666 |
| b, tokens 0–3 | Muon checkpoint 313 → 391 | 824763 / start 321051398 | 9153 |

Live entry: `/tmp/dense-v3-resume-verification.wcsmQnDm/resume.py`.
Source SHA `1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2`;
authority SHA `51e95a2adf6cf5acf151a18976c476e275307262b6f94528b313b964e2baf7d0`;
download receipt SHA `3c48d5c7d10434fe48925ec223e55c85df59e23ee6767a9d4bffbf1f5901f332`.
**Do not edit or restart the live entry.** Archived entry files preserve the exact
source; their copied directories are not an alternative launcher or live observer.

Use this narrow live observer with a new absolute output path:

```bash
CUDA_VISIBLE_DEVICES='' /usr/bin/python -B \
  /tmp/dense-v3-resume-verification.wcsmQnDm/observe.py \
  --source-sha 1746b8a382bb45fefc2ab17beaf7706984fe91802c528bec48c88a2daa038bb2 \
  --authorization-sha 51e95a2adf6cf5acf151a18976c476e275307262b6f94528b313b964e2baf7d0 \
  --output /tmp/new-resume-observation.json
```

Each waiter requires its entire six-run training pool, all 84 final BEIR units
and all 30 stage probes before requesting all four GPUs in both lease namespaces.
The worker calls the existing native bound Trainer resume hooks with the exact
downloaded seal, retains the full 391-step horizon and complete original data
order, and writes a separate new output. W&B reporting is disabled. No fresh-run
or five-save completion record is synthesized for this resumed suffix.

After all four ranks actually exit zero, a fresh CPU reader must compare all
134 model tensors, every named optimizer/scheduler state, all four rank RNG states
and numerical Trainer counters against the uninterrupted final checkpoint.
There is no tolerance or automatic retry. A failure preserves all artifacts;
actual success must be observed before updating the still-pending GPU equivalence.
This engineering provenance stays outside the manuscript and appendix.

The same dated snapshot has **8/12 scientific continuations complete**, the fifth
pair at **245/391 and 239/391**, and **40/40 completed checkpoints HF-verified**.
The newest backup revisions are `02ee46dd378be7652315faa43d4de920ee06fc82`
and `5ff6dd7c552edaa4bb00a9b6fe7e30843c1232b0`. Their unchanged receipts are
copied in `backup-receipts/`. BEIR/probe/summary remain queued without failures.
