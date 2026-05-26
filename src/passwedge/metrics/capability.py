"""Capability metrics: pass@k (Chen 2021) and pass^k (Beyond pass@1, Def. 2).

See ``docs/DEFINITIONS.md`` for provenance. pass@k is the probability that *at
least one* of k attempts succeeds; pass^k is the probability that *all* k
succeed. Both have unbiased sampling-without-replacement estimators over n
attempts with c successes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["PassPowKEstimator", "pass_at_k", "pass_pow_k"]

PassPowKEstimator = Literal["unbiased", "plugin"]


def _validate(n: int, c: int, k: int) -> None:
    if n < 0 or c < 0 or k < 0:
        raise ValueError("n, c, k must be non-negative")
    if c > n:
        raise ValueError(f"successes c={c} cannot exceed attempts n={n}")
    if k > n:
        raise ValueError(f"k={k} cannot exceed attempts n={n}")
    if k == 0:
        raise ValueError("k must be >= 1")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: probability at least one of k attempts succeeds.

    Uses the numerically stable product form from Chen et al. (2021):
    ``1 - prod_{i=n-c+1}^{n} (1 - k/i)``.
    """
    _validate(n, c, k)
    if n - c < k:
        # every size-k subset must contain at least one of the c successes
        return 1.0
    factors = 1.0 - k / np.arange(n - c + 1, n + 1, dtype=np.float64)
    return float(1.0 - np.prod(factors))


def pass_pow_k(n: int, c: int, k: int, estimator: str = "unbiased") -> float:
    """pass^k: probability that *all* k attempts succeed (Beyond pass@1, Def. 2).

    ``estimator="unbiased"`` uses the hypergeometric form C(c,k)/C(n,k),
    computed stably as ``prod_{i=0}^{k-1} (c-i)/(n-i)`` (0 when k > c).
    ``estimator="plugin"`` uses ``(c/n)**k``. Allowed values are given by
    :data:`PassPowKEstimator`.
    """
    if estimator not in ("unbiased", "plugin"):
        raise ValueError(f"unknown estimator {estimator!r}; use 'unbiased' or 'plugin'")
    _validate(n, c, k)
    if estimator == "plugin":
        return float((c / n) ** k)
    if k > c:
        return 0.0
    idx = np.arange(k, dtype=np.float64)
    return float(np.prod((c - idx) / (n - idx)))
