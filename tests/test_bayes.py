"""Bayesian posterior helpers: closed forms vs numerical integration + prior semantics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from passwedge.metrics.bayes import (
    PRIOR_PRESETS,
    BetaPosterior,
    beta_posterior,
    dirichlet_posterior,
)


def test_prior_presets_means() -> None:
    assert beta_posterior(3, 4, "uniform").mean() == pytest.approx((3 + 1) / (4 + 2))
    assert beta_posterior(3, 4, "jeffreys").mean() == pytest.approx((3 + 0.5) / (4 + 1))
    # paper (uniform Dirichlet) posterior mean is order-equivalent to pass@1
    assert beta_posterior(3, 4, "paper").mean() == pytest.approx((3 + 1) / (4 + 2))
    assert PRIOR_PRESETS["jeffreys"] == (0.5, 0.5)


def test_explicit_prior_and_validation() -> None:
    p = beta_posterior(2, 5, (2.0, 2.0))
    assert p.alpha == 4.0 and p.beta == 5.0
    with pytest.raises(ValueError):
        beta_posterior(6, 5)  # c > n
    with pytest.raises(ValueError):
        beta_posterior(0, 5, "weird")
    with pytest.raises(ValueError):
        beta_posterior(0, 5, "haldane")  # improper posterior (alpha = 0)


def test_expected_pow_k_matches_integration() -> None:
    post = beta_posterior(7, 10, "jeffreys")
    a, b = post.alpha, post.beta
    for k in (1, 2, 3, 5, 8):
        numeric, _ = integrate.quad(lambda p, k=k: p**k * stats.beta.pdf(p, a, b), 0.0, 1.0)
        assert post.expected_pow_k(k) == pytest.approx(numeric, abs=1e-7)


def test_expected_pass_at_k_matches_integration() -> None:
    post = beta_posterior(7, 10, "jeffreys")
    a, b = post.alpha, post.beta
    for k in (1, 2, 4):
        numeric, _ = integrate.quad(
            lambda p, k=k: (1.0 - (1.0 - p) ** k) * stats.beta.pdf(p, a, b), 0.0, 1.0
        )
        assert post.expected_pass_at_k(k) == pytest.approx(numeric, abs=1e-7)


def test_expected_pow_k_edges() -> None:
    post = beta_posterior(3, 4)
    assert post.expected_pow_k(0) == 1.0
    assert post.expected_pass_at_k(0) == 0.0
    assert post.expected_pow_k(1) == pytest.approx(post.mean())
    with pytest.raises(ValueError):
        post.expected_pow_k(-1)


def test_credible_interval_brackets_mean() -> None:
    post = beta_posterior(7, 10, "jeffreys")
    lo, hi = post.credible_interval(0.95)
    assert lo < post.mean() < hi
    assert lo == pytest.approx(stats.beta.ppf(0.025, post.alpha, post.beta))
    assert hi == pytest.approx(stats.beta.ppf(0.975, post.alpha, post.beta))
    with pytest.raises(ValueError):
        post.credible_interval(1.5)


def test_variance_matches_formula() -> None:
    post = BetaPosterior(alpha=4.0, beta=6.0)
    assert post.var() == pytest.approx(stats.beta.var(4.0, 6.0))


def test_dirichlet_posterior() -> None:
    post = dirichlet_posterior([3, 1, 0], "uniform")
    assert np.allclose(post.alpha, [4.0, 2.0, 1.0])
    assert post.mean().sum() == pytest.approx(1.0)
    ci = post.credible_interval(0.9)
    assert ci.shape == (3, 2)
    assert np.all(ci[:, 0] <= ci[:, 1])
    with pytest.raises(ValueError):
        dirichlet_posterior([5], "uniform")  # need >= 2 categories
