# Post-result interpretation of cosine coordinate deletion

This bounded CPU analysis was specified after viewing the complete primary
results, including the positive degrading-mass/retrieval association. It is
exploratory measurement interpretation, not a replacement primary analysis,
an optimizer mechanism claim, or an additional completion prerequisite.

Use all 61 authenticated FP32 vector states and the complete FP64 attribution
outputs: 224 paired queries, 14 tasks, eight candidates, 768 coordinates. The
existing immutable functional/calibration HF manifest authenticates every input.
Do not load models, encode again, change frozen numerical sources, train, select
rates/stages/tasks, or edit the manuscript while its document waiter is active.

For unit vectors, let m = s_positive - s_original_hardest_negative and let
c_j = q_j (p_j - n_j). Literal deletion with renormalization gives the exact
finite-margin decomposition a_j = -c_j + r_j + u_j. Here r_j is the difference
between the fixed-negative deleted margin and m - c_j, while u_j <= 0 is the
penalty for reselecting the hardest negative after deletion. This identity and
the switching sign were already established in the earlier constructed-vector
interpretation note. The present work measures their sizes on actual states.

The additional local derivative uses coordinate attenuation by sqrt(1-delta):
g_j(q,d) = -q_j d_j + s(q,d) (q_j^2 + d_j^2) / 2.
Its sum across coordinates is zero for unit vectors. The fixed-negative margin
derivative is the positive-minus-negative difference of those expressions and
also sums to zero. This follows from cosine scale invariance, not an optimizer
property or a novel optimization theorem. Away from a tie, it is also the local
derivative of the hardest-negative margin. At ties the selected branch derivative
need not equal the directional derivative of the maximum; retain and count ties.

Task averaging is performed before the original helpful/degrading sign split.
For the first-order effects the helpful and degrading masses therefore balance,
both equal to half the L1 sensitivity. Finite deletion need not be well
approximated by this derivative: retain actual approximation error, normalization
remainder, negative switching, coordinate energy and all task/state results.
Neither a near balance nor a positive association proves a Muon explanation.

Execution: reuse the unchanged original cosine/rank functions in a separate
reader; literally delete every coordinate to obtain the newly needed per-query
fixed-negative and switched margins. Require the task-mean original attribution
arrays to agree bit for bit. Store all task-mean component arrays and all 854
task/61 state summaries. Verify exact algebra to a predeclared 1e-12 diagnostic
absolute bound, separately from exact original-array equality. Synthetic
derivative/scale-invariance controls are not scientific observations. Report
unadjusted descriptive Pearson associations across the 60 trained states for
all declared scalar summaries and BEIR; no new hypothesis tests or selection.

No novelty or causal claim is implied. The frozen functional metrics, inference,
decisions, original evidence, current paper and all live dispatch inputs remain
unchanged. Any failed attempt remains retained in a separately named directory.
