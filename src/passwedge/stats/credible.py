"""Credible intervals for a Beta posterior: equal-tail and highest-posterior-density."""

from __future__ import annotations

import math
from collections.abc import Callable

from scipy import stats

__all__ = ["equal_tail_interval", "hpd_interval"]


def _check(alpha: float, beta: float, level: float) -> None:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha, beta must be > 0")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")


def equal_tail_interval(alpha: float, beta: float, level: float = 0.95) -> tuple[float, float]:
    """Equal-tail credible interval of a ``Beta(alpha, beta)`` at the given level."""
    _check(alpha, beta, level)
    tail = (1.0 - level) / 2.0
    lo = float(stats.beta.ppf(tail, alpha, beta))
    hi = float(stats.beta.ppf(1.0 - tail, alpha, beta))
    return lo, hi


def _golden_min(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-10) -> float:
    """Golden-section minimizer of a unimodal function on ``[lo, hi]``."""
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(200):
        if b - a < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = f(d)
    return (a + b) / 2.0


def hpd_interval(alpha: float, beta: float, level: float = 0.95) -> tuple[float, float]:
    """Highest-posterior-density interval of a ``Beta(alpha, beta)``.

    Found by minimizing interval width over the lower-tail mass; for a monotone density
    (``alpha < 1`` or ``beta < 1``) this collapses to a one-sided interval touching a
    boundary, which the optimizer recovers. Equal-tail and HPD coincide when alpha==beta.
    """
    _check(alpha, beta, level)
    if alpha == beta:
        return equal_tail_interval(alpha, beta, level)

    def width(lower_mass: float) -> float:
        lo_q = float(stats.beta.ppf(lower_mass, alpha, beta))
        hi_q = float(stats.beta.ppf(lower_mass + level, alpha, beta))
        return hi_q - lo_q

    upper = 1.0 - level
    q = _golden_min(width, 0.0, upper)
    q = min(max(q, 0.0), upper)
    lo = float(stats.beta.ppf(q, alpha, beta))
    hi = float(stats.beta.ppf(q + level, alpha, beta))
    return lo, hi
