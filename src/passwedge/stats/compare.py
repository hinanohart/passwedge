"""Compare two pass rates: a two-proportion z-test and credible-interval disjointness."""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["ZTestResult", "intervals_disjoint", "two_proportion_z_test"]


@dataclass(slots=True, frozen=True)
class ZTestResult:
    """Result of a two-sided two-proportion z-test."""

    z: float
    p_value: float
    diff: float  # p1 - p2


def _normal_sf(x: float) -> float:
    """Upper tail of the standard normal via erfc (no scipy dependency needed)."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def two_proportion_z_test(c1: int, n1: int, c2: int, n2: int) -> ZTestResult:
    """Two-sided two-proportion z-test comparing pass rates c1/n1 vs c2/n2.

    Uses the pooled-variance z statistic. The two-sided p-value is computed from the
    standard normal survival function via ``math.erfc``. Raises if either sample is
    empty or both rates are identical degenerate (0 pooled variance).
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError("n1, n2 must be positive")
    if not (0 <= c1 <= n1) or not (0 <= c2 <= n2):
        raise ValueError("require 0 <= c <= n for both samples")
    p1 = c1 / n1
    p2 = c2 / n2
    pooled = (c1 + c2) / (n1 + n2)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    diff = p1 - p2
    if se == 0.0:
        # both pooled proportions are 0 or 1: no variance to test against
        z = 0.0 if diff == 0.0 else math.copysign(math.inf, diff)
        p_value = 1.0 if diff == 0.0 else 0.0
        return ZTestResult(z=z, p_value=p_value, diff=diff)
    z = diff / se
    p_value = 2.0 * _normal_sf(abs(z))
    return ZTestResult(z=z, p_value=p_value, diff=diff)


def intervals_disjoint(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """True if the two closed intervals do not overlap (a conservative significance cue)."""
    a_lo, a_hi = min(a), max(a)
    b_lo, b_hi = min(b), max(b)
    return a_hi < b_lo or b_hi < a_lo
