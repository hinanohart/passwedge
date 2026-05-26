"""pass@k / pass^k: analytic cross-checks against independent combinatorics + properties."""

from __future__ import annotations

from math import comb

import pytest
from hypothesis import given
from hypothesis import strategies as st

from passwedge.metrics.capability import pass_at_k, pass_pow_k


def ref_pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def ref_pass_pow_k(n: int, c: int, k: int) -> float:
    if k > c:
        return 0.0
    return comb(c, k) / comb(n, k)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(1, 1, 1), (5, 0, 1), (5, 5, 5), (10, 3, 2), (12, 9, 5), (20, 7, 3), (8, 4, 4)],
)
def test_pass_at_k_matches_combinatorics(n: int, c: int, k: int) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(ref_pass_at_k(n, c, k), abs=1e-12)


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [(1, 1, 1), (5, 0, 1), (5, 5, 5), (10, 3, 2), (12, 9, 5), (20, 7, 3), (8, 4, 4)],
)
def test_pass_pow_k_matches_combinatorics(n: int, c: int, k: int) -> None:
    assert pass_pow_k(n, c, k) == pytest.approx(ref_pass_pow_k(n, c, k), abs=1e-12)


def test_chen2021_known_value() -> None:
    # n=200, c=100, k=10 -- a value reproducible from the Codex paper estimator.
    assert pass_at_k(200, 100, 10) == pytest.approx(ref_pass_at_k(200, 100, 10), abs=1e-12)


def test_k_equals_one_agrees() -> None:
    for n, c in [(4, 1), (10, 7), (12, 9)]:
        assert pass_at_k(n, c, 1) == pytest.approx(c / n)
        assert pass_pow_k(n, c, 1) == pytest.approx(c / n)
        assert pass_pow_k(n, c, 1, estimator="plugin") == pytest.approx(c / n)


def test_plugin_vs_unbiased_endpoints() -> None:
    assert pass_pow_k(10, 10, 3, estimator="plugin") == pytest.approx(1.0)
    assert pass_pow_k(10, 0, 3, estimator="plugin") == pytest.approx(0.0)
    assert pass_pow_k(10, 2, 3, estimator="unbiased") == 0.0  # k > c


def test_edge_all_or_nothing() -> None:
    assert pass_at_k(7, 7, 3) == 1.0
    assert pass_at_k(7, 0, 3) == 0.0
    assert pass_pow_k(7, 7, 3) == pytest.approx(1.0)
    assert pass_pow_k(7, 0, 3) == 0.0


@pytest.mark.parametrize("estimator", ["unbiased", "plugin"])
def test_invalid_inputs(estimator: str) -> None:
    with pytest.raises(ValueError):
        pass_pow_k(5, 6, 2, estimator=estimator)  # c > n
    with pytest.raises(ValueError):
        pass_pow_k(5, 3, 6, estimator=estimator)  # k > n
    with pytest.raises(ValueError):
        pass_pow_k(5, 3, 0, estimator=estimator)  # k < 1
    with pytest.raises(ValueError):
        pass_at_k(5, 3, 0)


def test_unknown_estimator() -> None:
    with pytest.raises(ValueError, match="unknown estimator"):
        pass_pow_k(5, 3, 2, estimator="bogus")


# ---- property-based ----


@st.composite
def nck(draw: st.DrawFn) -> tuple[int, int, int]:
    n = draw(st.integers(min_value=1, max_value=60))
    c = draw(st.integers(min_value=0, max_value=n))
    k = draw(st.integers(min_value=1, max_value=n))
    return n, c, k


@given(nck())
def test_range_bounds(t: tuple[int, int, int]) -> None:
    n, c, k = t
    assert 0.0 <= pass_at_k(n, c, k) <= 1.0
    assert 0.0 <= pass_pow_k(n, c, k) <= 1.0
    assert 0.0 <= pass_pow_k(n, c, k, estimator="plugin") <= 1.0


@given(nck())
def test_pow_le_at(t: tuple[int, int, int]) -> None:
    # all-k-succeed is a subset of >=1-of-k-succeeds
    n, c, k = t
    assert pass_pow_k(n, c, k) <= pass_at_k(n, c, k) + 1e-12


@given(st.integers(1, 40).flatmap(lambda n: st.tuples(st.just(n), st.integers(0, n))))
def test_monotonicity_in_k(nc: tuple[int, int]) -> None:
    n, c = nc
    at = [pass_at_k(n, c, k) for k in range(1, n + 1)]
    powk = [pass_pow_k(n, c, k) for k in range(1, n + 1)]
    assert all(at[i] <= at[i + 1] + 1e-12 for i in range(len(at) - 1))  # pass@k increases
    assert all(powk[i] >= powk[i + 1] - 1e-12 for i in range(len(powk) - 1))  # pass^k decreases
