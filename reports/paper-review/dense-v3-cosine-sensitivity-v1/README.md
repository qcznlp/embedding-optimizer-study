# What coordinate-deletion mass measures in the actual retrievers

This **post-result interpretation analysis** covers all **61 genuine states /
854 task-state cells**. It does not replace a frozen feature, rerun training,
change an inference decision or establish a Muon mechanism. The question is why
larger degrading attribution mass can accompany better retrieval.

The useful empirical observation is that degrading mass closely tracks the
scale of coordinate sensitivity. Its helpful/degrading balance is also affected
by recomputing the hardest negative. Consequently, reading that mass as an
independent amount of harmful representational content is not justified.

## A local balance identity, and an exact finite decomposition

Let query and document vectors be unit normalized. Write their cosine as
$s=q^T d$. Attenuating coordinate $j$ by $\sqrt{1-\delta}$ in both vectors
has derivative, at $\delta=0$,

$$g_j(q,d)=-q_jd_j+\frac{s}{2}(q_j^2+d_j^2).$$

Since both squared norms equal one, $\sum_j g_j(q,d)=0$. Subtracting the
original hardest-negative derivative from the positive derivative gives a
margin derivative with the same zero sum. This is a consequence of cosine
scale invariance, **not a new optimization theorem or a Muon property**. A
unique original hardest negative makes this the local derivative of the actual
max-negative margin; all 13,664 state/query observations here have no such tie.

Task averaging preserves the zero sum. If helpful and degrading mass are
computed from these first-order task-mean effects, they are therefore equal:

$$H^{(1)}_t=B^{(1)}_t=\tfrac12\sum_j|g_{tj}|.$$

This does **not** assert equality for finite coordinate deletion. For the
actual deletion let $m$ be the original margin, $n$ its original hardest
negative, $c_j=q_j(p_j-n_j)$, and $m^{\mathrm{fixed}}_{-j}$ the deleted margin
against that same negative. Then

$$a_j=-c_j+r_j+u_j,$$

where $r_j=m^{\mathrm{fixed}}_{-j}-(m-c_j)$ is the renormalization term and
$u_j=m_{-j}-m^{\mathrm{fixed}}_{-j}\leq0$ is the penalty for reselecting the
hardest negative. The earlier [constructed-vector note](../coordinate-utility-interpretation-v1/README.md)
already established this finite identity and switching sign. The present
analysis measures their sizes on the actual complete population.

## Actual full-population results

The original 224-query, 14-task, eight-candidate, 768-coordinate definitions
are unchanged. Every original margin and nDCG task-mean attribution is
**bitwise identical** after the new literal deletion pass. The pass newly
retains per-task direct, renormalization, fixed-negative, switching and
first-order components. All rates and stages are included.

| Descriptive readout | Observed value |
| --- | ---: |
| Mean task-relative L1 error, finite effects versus first-order effects | 4.5664% |
| Range of state-mean relative L1 errors | 2.3542%--8.1508% |
| Largest individual task-state relative L1 error | 32.5364% |
| Mean fraction of query-coordinate deletions changing the hardest negative | 0.9545% |
| Mean switching mass / total absolute task attribution mass | 4.2321% |
| Degrading mass versus first-order half-L1 sensitivity, 60-state Pearson | 0.9554 |
| Degrading mass versus full-vector shortlist margin, 60-state Pearson | 0.8372 |
| Shortlist margin versus full BEIR, 60-state Pearson | 0.9579 |

These Pearson coefficients are **unadjusted exploratory descriptions** of the
60 trained states, not independent training-seed evidence, hypothesis tests,
the original residual coefficients or held-out predictions. All fifteen pairs
of the six declared scalar summaries appear in
[descriptive_associations.csv](actual/descriptive_associations.csv), not just
the strongest associations. The approximation is not uniformly close: the
worst task-state error must remain visible.

The signed balance is especially informative. Averaging equally over all
61 states and 14 tasks gives:

| Signed coordinate-sum component | Mean |
| --- | ---: |
| Direct score contribution removed | -0.2418362606 |
| Renormalization contribution | +0.2418374276 |
| Hardest-negative switching contribution | -0.0093441876 |
| Actual total deletion effect | -0.0093430205 |

The direct and renormalization terms almost cancel in this particular pooled
readout. The remaining helpful-minus-degrading mass is 0.0093430205, almost
entirely accounted for algebraically by switching; the fixed-negative remainder
has the opposite sign and magnitude 0.0000011670. This is exact accounting of
the measurement, **not a causal decomposition of retrieval improvement**.
Individual task/state components, including opposite signs, are retained in
[task_summary.csv](actual/task_summary.csv) and all 61
[state summaries](actual/state_summary.csv).

This clarifies why the original positive degrading-mass/retrieval association
need not be paradoxical: positive deletion mass can grow with a representation's
overall ranking sensitivity. It does **not** show that this causes Muon's gain,
that dimension use is irrelevant, or that all finite deletion effects are local.
The original nine contrasts, rotation checks, held-dose predictions and recipe
sensitivity controls remain the scientific comparison. No new first-order
feature is substituted into those frozen tests.

## Independent verification and preserved evidence

The actual literal analysis, session **29917**, completed at **23:34:46 UTC**
with observed exit zero (terminal **457651**). Completion SHA-256:
`b7bb5584fb3e462dc734ca30b0705c75de3f9e751270ebdb7edfbf41343ee75a`.
It checks 247 metadata/vector/attribution/bridge inputs before and after use,
against the existing immutable HF manifest
`0b498557040c2329e6935fdc539bce6d029b8bbf02f8ffcbc194912203b8cc38`.
Synthetic finite-difference and scale-invariance checks are separate controls,
not additional scientific states or results.

A separately written algebraic verifier uses the **previously anonymously
downloaded HF payload** and copied frozen source evidence, without reading the
original producer inputs or importing the new literal analysis. All six
component arrays for all 61 states agree with a maximum absolute error of
**3.96e-16**, below the declared diagnostic bound 1e-12; scalar readback error
is 2.22e-16. Its subtractive algebra is only an independent reference on these
actual well-conditioned vectors: minimum retained coordinate energy is 0.847.
It does not replace the original literal-deletion kernel. Session **8012**,
terminal **681eb9**, exited zero at **23:36:33 UTC**. See
[independent-hf-readback.json](independent-hf-readback.json).

The standalone verifier supports restored inputs:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python reports/paper-review/dense-v3-cosine-sensitivity-v1/verify_cosine_readout.py \
  --input reports/paper-review/dense-v3-cosine-sensitivity-v1/actual \
  --payload-root /path/to/verified-functional-hf-snapshot \
  --frozen-source-root reports/paper-review/dense-v3-cosine-sensitivity-v1/frozen-source \
  --output /tmp/new-independent-cosine-readback.json
```

The [functional restoration guide](../../../docs/functional-analysis-restoration.md)
provides the actual public dataset revision and download command. This readback
checks the independent algebra from raw vectors and scalar summaries; it does
not re-encode models, repeat the literal producer, run statistical inference,
prove physical second-host execution or publish this local source tree.

## Paper review and current next action

The [citation/wording review](citation-and-wording-review.md) checks all 14
cited keys against primary sources and records one contributor-name error.
[Five deferred prose replacements](deferred-prose-candidate.json) additionally
clarify grid/selection estimands, the selected AdamW boundary rate, and the
primary complete-recipe versus continuation hidden-rule comparison. They are
**not installed**: the original document/replay waiters and their inputs remain
unchanged. Integrate reviewed wording only after the current native document
and combined replay finish, then recompile and review the actual final paper.
The new sensitivity interpretation is optional explanatory analysis, not a
new completion gate or pretext for retraining.

At **23:37:07 UTC**, continuation BEIR is **10/168** with eight exact registered
workers live and no terminal failure. The six downstream CPU waiters remain
live, without actual summary collectors, full paper/replay, outcome upload or
GPU resume started. All 24 scientific training runs / 120 checkpoints remain
complete. The full goal is active; no manuscript, numerical protocol, live
dispatcher, original evidence or remote source was changed by this work.
