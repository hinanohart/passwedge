"""Bayesian posterior helpers -- operationalization of *Don't Pass@k* (arXiv:2510.04265).

The paper models evaluation outcomes as categorical data with a Dirichlet prior and
reports posterior means + credible intervals instead of point estimates. For the binary
success/failure case this reduces to a Beta posterior. See ``docs/DEFINITIONS.md``.

Honest-marketing note: arXiv:2510.04265 does **not** define a metric named "Bayes@k".
These helpers are passwedge's operationalization of that paper's framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import betaln

__all__ = [
    "PRIOR_PRESETS",
    "BetaPosterior",
    "DirichletPosterior",
    "beta_posterior",
    "dirichlet_posterior",
]

# (a, b) pseudo-counts for the Beta prior.
PRIOR_PRESETS: dict[str, tuple[float, float]] = {
    "jeffreys": (0.5, 0.5),
    "uniform": (1.0, 1.0),
    "paper": (1.0, 1.0),  # arXiv:2510.04265 uniform Dirichlet -> posterior mean == pass@1
    "haldane": (0.0, 0.0),
}


def _resolve_prior(prior: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(prior, str):
        try:
            return PRIOR_PRESETS[prior]
        except KeyError:
            raise ValueError(
                f"unknown prior {prior!r}; choose from {sorted(PRIOR_PRESETS)} or pass (a, b)"
            ) from None
    a, b = prior
    if a < 0 or b < 0:
        raise ValueError("prior pseudo-counts must be non-negative")
    return float(a), float(b)


@dataclass(slots=True, frozen=True)
class BetaPosterior:
    """Posterior ``Beta(alpha, beta)`` over a single task's success probability p."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                "posterior alpha, beta must be > 0 (a Haldane prior with 0 successes or "
                "0 failures yields an improper posterior)"
            )

    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def var(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Equal-tail credible interval at the given level (e.g. 0.95)."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        lo = float(stats.beta.ppf(tail, self.alpha, self.beta))
        hi = float(stats.beta.ppf(1.0 - tail, self.alpha, self.beta))
        return lo, hi

    def expected_pow_k(self, k: int) -> float:
        """E[p^k] under the posterior -- expected probability all k attempts succeed.

        ``= B(alpha+k, beta) / B(alpha, beta) = prod_{i=0}^{k-1} (alpha+i)/(alpha+beta+i)``,
        evaluated in log space for stability.
        """
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return 1.0
        return float(np.exp(betaln(self.alpha + k, self.beta) - betaln(self.alpha, self.beta)))

    def expected_pass_at_k(self, k: int) -> float:
        """E[1-(1-p)^k] under the posterior -- expected pass@k (>=1 of k succeeds)."""
        if k < 0:
            raise ValueError("k must be >= 0")
        if k == 0:
            return 0.0
        return float(
            1.0 - np.exp(betaln(self.alpha, self.beta + k) - betaln(self.alpha, self.beta))
        )


def beta_posterior(c: int, n: int, prior: str | tuple[float, float] = "jeffreys") -> BetaPosterior:
    """Beta posterior from c successes in n trials. Default prior: Jeffreys Beta(0.5, 0.5)."""
    if n < 0 or c < 0:
        raise ValueError("n, c must be non-negative")
    if c > n:
        raise ValueError(f"successes c={c} cannot exceed trials n={n}")
    a, b = _resolve_prior(prior)
    return BetaPosterior(alpha=a + c, beta=b + (n - c))


@dataclass(slots=True, frozen=True)
class DirichletPosterior:
    """Posterior ``Dirichlet(alpha)`` over K categorical outcomes (arXiv:2510.04265)."""

    alpha: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.alpha.ndim != 1 or self.alpha.size < 2:
            raise ValueError("alpha must be a 1-D vector of length >= 2")
        if np.any(self.alpha <= 0):
            raise ValueError("posterior concentrations must be > 0")

    def mean(self) -> NDArray[np.float64]:
        return np.asarray(self.alpha / self.alpha.sum(), dtype=np.float64)

    def credible_interval(self, level: float = 0.95) -> NDArray[np.float64]:
        """Marginal equal-tail intervals per category (each margin is Beta)."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must be in (0, 1)")
        tail = (1.0 - level) / 2.0
        total = self.alpha.sum()
        out = np.empty((self.alpha.size, 2), dtype=np.float64)
        for i, a_i in enumerate(self.alpha):
            b_i = total - a_i
            out[i, 0] = stats.beta.ppf(tail, a_i, b_i)
            out[i, 1] = stats.beta.ppf(1.0 - tail, a_i, b_i)
        return out


def dirichlet_posterior(
    counts: NDArray[np.int_] | list[int], prior: str | float = "jeffreys"
) -> DirichletPosterior:
    """Dirichlet posterior from per-category counts.

    ``prior`` may be ``"jeffreys"`` (0.5 each), ``"uniform"``/``"paper"`` (1.0 each),
    or a float concentration applied to every category.
    """
    arr = np.asarray(counts, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("counts must be a 1-D vector of length >= 2")
    if np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    if isinstance(prior, str):
        presets = {"jeffreys": 0.5, "uniform": 1.0, "paper": 1.0}
        if prior not in presets:
            raise ValueError(f"unknown prior {prior!r}; use {sorted(presets)} or a float")
        conc = presets[prior]
    else:
        conc = float(prior)
        if conc <= 0:
            raise ValueError("float prior concentration must be > 0")
    return DirichletPosterior(alpha=arr + conc)
