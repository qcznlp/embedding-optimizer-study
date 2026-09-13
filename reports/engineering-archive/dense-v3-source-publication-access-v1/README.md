# Source-publication access clarification — 2026-09-13

This is operational publication evidence, not a new experiment, a release, a
permission change, or permission to retry a rejected operation.

## What the original denial actually covers

The original `prior-github-403.json`, dated **2026-09-06 08:43:33 UTC**, records
two rejected writes to issue **#41**: archiving its previous body as a comment,
and updating its body. Both return `Resource not accessible by integration`.
The receipt explicitly records **no branch push or pull-request change attempted**.
Its SHA is `5bdc8f9cf376be94fe76b3af33dcf6d31fed2080d42e613d4eb12d8d5367a859`.
Do not describe this as an observed failed Git push, nor treat it as proof that
all distinct GitHub write surfaces are denied. The original no-retry/no-alternate-
credential boundary and all separate publication gates remain unchanged.

## Actual current read-only checks

`actual-readbacks.json`, captured at **01:05:05 UTC**, preserves seven read
operations through the existing connected GitHub service. No write was attempted.

- Repository and current-identity reads succeed. The returned login is `qcznlp`;
  repository `qcznlp/embedding-optimizer-study` is reported **public**, not archived,
  with default branch `main` and account-level admin/push/read permissions true.
  This turn did not change visibility or permissions.
- A branch-name search returns `main` without a head SHA. Do not claim it proves
  a particular branch tip. The separate recent-repository-commit query returns
  `f231a6430712388778f32ad1736a4cb6de3bec3e`, dated **2026-09-04 17:37:23 UTC**;
  it also matches the actual local Git base (tool **4f5d58**, exit zero).
- A separately fetched README excerpt is pinned to that returned commit. It is
  an **old historical-result snapshot**, not the current corrected v3 result set.
  The excerpt must not feed current scientific tables, claims, or run accounting.
- Installation metadata does **not** establish Issues or Contents write scopes.
  The unrestricted listing returns six installations, but none matches the target
  `qcznlp` account in the explicitly scoped view. Unrelated account values are
  omitted. These two installation records are derived scoped views, not complete
  raw responses. Absence here does not establish that no connection exists.

Account-level repository permission, a working read, and an available tool do
not prove a successful application-authorized write. Conversely, the old issue
403 is not evidence of an attempted Contents/Git write failure. **Write recovery
is unverified**, rather than newly tested and failed or assumed repaired.

Following the plugin-management skill, the existing connected service was used;
no replacement plugin, identity, credential, or permission setting was introduced.
The plugin-policy inspection tool is not an OAuth/admin-scope inspector and was
not misused for this question. One **non-blocking owner question** was accepted:
confirm normal existing-integration repository access and Issues write permission.
No reply was present when the readback was recorded. Do not ask it repeatedly or
request a token in chat. It does not pause evaluation or authorize a denied retry.

## Remaining work and live computation

Current local WIP is not claimed to be published. Complete real outcomes,
native/combined paper reconstruction, final visual/prose review, actual GPU
resumes, full durability, and a reviewed assembled source release are still
required. The existing complete-paper pre-commit and source-transition gates
were not waived. The historical HF deletion denial is also untouched.

At **01:05:48 UTC**, the original BEIR observer (**e08382**, exit zero) records
**14/168** tasks, eight exact FEVER workers live/R, and no coordinator failure.
The original sessions **41515 / 37455** were directly polled this turn and
returned live handles (**0215d6 / c94937**), not terminal exits. All six downstream
CPU waiters remain live/S with no actual children/results or failure: observer
tools **94bf01, 1b62dd, 9523d4, ad20ab, 3a24f4**, all exit zero. Their unchanged
JSON outputs are copied here. No scientific task or source changed.

The authoritative main and bibliography remain at
`45a3d297e08dfbb99908508a68ee71f077f29203bc3da8b017f6564769bc177e` and
`fca6bf5762a2cecab795a1a60c1338ab4eb999acb5d78c437b9df0e5f3553218`
(tool **b684df**, exit zero). This turn is **PROGRESS** in correcting the
publication-access diagnosis and initiating the precise account-side follow-up,
plus a verified wait for actual evaluation. It is not new scientific completion.

At **01:10:02 UTC**, the original observer (**f8ee29**, exit zero,
`evaluation-access-final.json`) records **16/168**: the AdamW-source/AdamW-rule
and AdamW-source/Muon-rule seed-314159 FEVER evaluations have now exited zero.
Both original sessions again return live handles (**b99a40 / e96abb**) with
their genuine FEVER returns and automatic seed-161803 replacement starts.
There are still eight exact live/R FEVER workers and no coordinator failure.
These two task completions are additional actual progress; the earlier 14/168
observation is preserved as history, not overwritten.

The seven observer/denial copies are byte-identical to their original files
(**d7472f / 877b13**, exit zero, plus the final observation's separate copy).
The readback JSON hash is
`779f169d546a42f7eac3c68dd426d20e19f952a635f24d8cf68f18d9032e9320`.
The optional consistency check first exits 127 because `jq` is not installed
(**2ad227**); the existing Python interpreter then checks the same record
successfully (**b253e5**, exit zero), without installing anything. This checks
record consistency only, not GitHub write permission or scientific admission.
The operational document diff check also exits zero (**edfb72**).
