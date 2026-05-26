# Metric definitions and provenance

passwedge implements reliability-science metrics for repeated-attempt evaluation of
long-horizon LLM agents. Every metric below is implemented **exactly as defined in the
cited source**, or — where a source gives a concept without a single canonical estimator —
as an explicitly documented *operationalization*. We never present an operationalization as
a verbatim reproduction.

## Sources

- **[BP1]** Khanal, Tao, Zhou. *Beyond pass@1: A Reliability Science Framework for
  Long-Horizon LLM Agents.* arXiv:2603.29231 (2026). Defines pass^k, RDC, VAF, GDS, MOP.
- **[DPK]** Hariri, Samandar, Hinczewski, Chaudhary. *Don't Pass@k: A Bayesian Framework
  for Large Language Model Evaluation.* arXiv:2510.04265. Dirichlet-posterior framework;
  reference implementation [`scorio`](https://github.com/mohsenhariri/scorio) (MIT).
- **[CHEN]** Chen et al. *Evaluating Large Language Models Trained on Code.*
  arXiv:2107.03374 (2021). pass@k unbiased estimator.

> **Honest-marketing note.** [DPK] does **not** define a metric named "Bayes@k". passwedge's
> Bayesian posterior helpers are *our* operationalization of that paper's framework. Do not
> describe them as a paper-defined "Bayes@k".

---

## Capability metrics

### pass@k  — [CHEN]
Probability that **at least one** of `k` attempts succeeds. Unbiased estimator over `n`
attempts with `c` correct (sampling without replacement):

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Numerically stable product form used by passwedge (Codex paper):

```
pass@k = 1 - prod_{i=n-c+1}^{n} (1 - k/i)        for k <= n-c, else 1.0
```

### pass^k  — [BP1] Definition 2
Probability that **all** `k` independent repeated episodes succeed:
`pass^k = Pr[∩_{i=1}^{k} E(W^(i)) = 1.0]`.

- **Unbiased (sampling without replacement)** estimator over `n` attempts, `c` correct:
  `pass^k = C(c, k) / C(n, k)`   (0 if `k > c`).
- **Plugin** estimator: `pass^k ≈ (c/n)^k`.

passwedge does not claim a paper-specified estimator for pass^k beyond Definition 2; the
hypergeometric form is the standard unbiased estimator for "all k drawn are correct".

---

## Reliability metrics — [BP1]

### RDC — Reliability Decay Curve — Definition 3
Function mapping a duration bucket `d` to pass^k:  `RDC(M, k): d ↦ pass^k(M, d)`,
where `pass^k(M, d)` is the mean pass^k over tasks in bucket `d`.

**RDS — Reliability Decay Slope** (OLS slope summarizing the curve over bucket index
`b ∈ {0,1,2,3}`):

```
RDS = Σ_b (b - b̄)(y_b - ȳ) / Σ_b (b - b̄)²
```

The paper's equation writes `y_b = GDS(M, b)`. passwedge implements the slope
metric-agnostically (`reliability_decay_slope(values)`); the caller supplies the per-bucket
statistic (pass^k for the RDC slope, or GDS to match the paper's RDS equation verbatim).

### VAF — Variance Amplification Factor — Definition 4
```
VAF(M) = σ²[pass@1(M, T) | d=long] / σ²[pass@1(M, T) | d=short]
```
Variance is taken across **task instances** `T` within a bucket (not across repeated runs).
passwedge uses sample variance (`ddof=1`) by default; `ddof` is configurable. Returns
`+inf` if the short-bucket variance is 0 (documented edge case).

### GDS — Graceful Degradation Score — Definition 5
Per-episode partial credit over subtasks `{s_i}` with criticality weights `{w_i}`,
`Σ_i w_i = 1`:
```
GDS(τ, T) = Σ_i w_i · 𝟙[subtask s_i completed correctly]      ∈ [0, 1]
```
Weights are normalized to sum to 1 (a ValueError is raised if they cannot be).

### MOP — Meltdown Onset Point — Definition 6
Over a tool-call action sequence `a_0, a_1, …`:
```
p_t(tool_i) = |{ j ∈ [t-w, t] : a_j = tool_i }| / w        (window distribution)
H(t)        = - Σ_i p_t(tool_i) · log2 p_t(tool_i)          (window entropy, bits)
MOP         = first t* with  H(t*) > θ_H  AND  H(t*) - H(t*-w) > δ
            = ∞  if no such t* exists
```
Entropy is in **bits** (log base 2), consistent with the bit-valued threshold.

> **Operationalization note.** [BP1] writes the window as the closed interval `[t-w, t]`.
> Taken literally that is `w+1` points divided by `w`, which yields probabilities > 1 and a
> negative "entropy" whenever one tool fills the window. passwedge instead uses a
> length-`w` window (the `w` most recent actions). For a full window the counts form a
> proper distribution with `H ∈ [0, log2(#tools)]`; near the start (`t < w-1`) the window
> is shorter, counts sum to < 1, and `H` stays non-negative. For `t < w`, `H(t-w)` is
> taken as 0.

**Paper calibration constants** ([BP1] §6.4 / Appendix A.2), exposed as
`MOP_PAPER_DEFAULTS`:
```
θ_H = 1.711 bits,   δ = 0.000,   w = 5 steps
```
> ⚠ These constants were calibrated on the paper's 396-task benchmark and are
> **dataset-specific**. Applying them verbatim to other data will produce false positives.
> `meltdown_onset_point(...)` therefore **requires** `w`, `theta_h`, `delta` as explicit
> arguments (no silent defaults — fail-closed). Pass `**MOP_PAPER_DEFAULTS` to opt in.

---

## Bayesian posterior helpers — operationalization of [DPK]

Binary success modelled as `c` successes in `n` trials with a Beta prior `Beta(a, b)`
(the binary case of [DPK]'s Dirichlet framework); posterior `Beta(α, β)`,
`α = a + c`, `β = b + (n - c)`.

- Priors: `"jeffreys"` = Beta(0.5, 0.5) (default), `"uniform"`/`"paper"` = Beta(1, 1),
  or an explicit `(a, b)`.
- Posterior mean: `α / (α + β)`  (equals pass@1 under the uniform prior — [DPK]).
- Equal-tail credible interval from the Beta quantile function.
- `E[p^k]` under the posterior (expected all-k-success):
  `E[p^k] = Π_{i=0}^{k-1} (α+i)/(α+β+i) = B(α+k, β)/B(α, β)`.
- Posterior expected pass@k (≥1 success in k):
  `1 - Π_{i=0}^{k-1} (β+i)/(α+β+i)`.

The categorical/Dirichlet generalization is provided for multi-outcome data.
`prior="paper"` reproduces [DPK]/`scorio` numerics under the uniform prior (verified in
the test suite when `scorio` is installed).
