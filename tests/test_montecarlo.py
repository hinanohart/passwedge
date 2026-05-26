"""Monte-Carlo check that the pass@k / pass^k estimators match sampling-without-replacement
frequencies (seed fixed for determinism)."""

from __future__ import annotations

import numpy as np
import pytest

from passwedge.metrics.capability import pass_at_k, pass_pow_k


@pytest.mark.parametrize(("n", "c", "k"), [(12, 7, 3), (20, 13, 5), (15, 4, 2), (10, 5, 4)])
def test_monte_carlo_matches_estimators(n: int, c: int, k: int) -> None:
    rng = np.random.default_rng(20260527)
    samples = 80_000
    correct = np.zeros(n, dtype=bool)
    correct[:c] = True
    # draw k distinct indices per trial via argsort of random keys
    order = np.argsort(rng.random((samples, n)), axis=1)[:, :k]
    drawn = correct[order]  # (samples, k)
    emp_all = float(drawn.all(axis=1).mean())
    emp_any = float(drawn.any(axis=1).mean())
    assert emp_all == pytest.approx(pass_pow_k(n, c, k), abs=0.01)
    assert emp_any == pytest.approx(pass_at_k(n, c, k), abs=0.01)
