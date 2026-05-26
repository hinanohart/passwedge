"""Cross-check against scorio (arXiv:2510.04265's reference implementation, MIT).

scorio is an optional ``repro`` dependency, not a runtime or default-dev dependency, so
this module skips unless scorio is installed. When present, it confirms passwedge matches
the paper authors' numerics exactly: pass@k, pass^k (scorio ``pass_hat_k``), and the
posterior mean under a uniform prior (scorio's default).
"""

from __future__ import annotations

import numpy as np
import pytest

from passwedge.metrics.bayes import beta_posterior
from passwedge.metrics.capability import pass_at_k, pass_pow_k

se = pytest.importorskip("scorio.eval")

CASES = [(7, 10), (3, 10), (9, 12), (0, 5), (5, 5), (1, 8)]


def _binary_matrix(c: int, n: int) -> np.ndarray:
    return np.array([[1] * c + [0] * (n - c)])


def _scalar(x: object) -> float:
    return float(np.asarray(x).ravel()[0])


@pytest.mark.parametrize(("c", "n"), CASES)
def test_bayes_uniform_prior_matches_scorio(c: int, n: int) -> None:
    mu, _sigma = se.bayes(_binary_matrix(c, n), np.array([0.0, 1.0]))
    assert _scalar(mu) == pytest.approx(beta_posterior(c, n, "paper").mean(), abs=1e-9)


@pytest.mark.parametrize(("c", "n"), CASES)
@pytest.mark.parametrize("k", [1, 2, 5])
def test_pass_at_k_matches_scorio(c: int, n: int, k: int) -> None:
    if k > n:
        pytest.skip("k > n")
    got = _scalar(se.pass_at_k(_binary_matrix(c, n), k))
    assert got == pytest.approx(pass_at_k(n, c, k), abs=1e-9)


@pytest.mark.parametrize(("c", "n"), CASES)
@pytest.mark.parametrize("k", [1, 2, 5])
def test_pass_hat_k_matches_scorio(c: int, n: int, k: int) -> None:
    if k > n:
        pytest.skip("k > n")
    got = _scalar(se.pass_hat_k(_binary_matrix(c, n), k))
    assert got == pytest.approx(pass_pow_k(n, c, k), abs=1e-9)
