"""Statistical helpers: z-test, interval disjointness, equal-tail and HPD intervals."""

from __future__ import annotations

import pytest
from scipy import stats

from passwedge.stats.compare import intervals_disjoint, two_proportion_z_test
from passwedge.stats.credible import equal_tail_interval, hpd_interval


def test_two_proportion_z_test_known() -> None:
    res = two_proportion_z_test(8, 10, 3, 10)
    assert res.diff == pytest.approx(0.5)
    assert res.z == pytest.approx(2.2473, abs=1e-3)
    assert res.p_value == pytest.approx(2.0 * stats.norm.sf(abs(res.z)), abs=1e-9)


def test_two_proportion_z_test_degenerate() -> None:
    res = two_proportion_z_test(0, 10, 0, 10)  # zero pooled variance, equal rates
    assert res.z == 0.0
    assert res.p_value == 1.0


def test_two_proportion_z_test_validation() -> None:
    with pytest.raises(ValueError):
        two_proportion_z_test(1, 0, 1, 5)
    with pytest.raises(ValueError):
        two_proportion_z_test(6, 5, 1, 5)


def test_intervals_disjoint() -> None:
    assert intervals_disjoint((0.1, 0.2), (0.3, 0.4)) is True
    assert intervals_disjoint((0.3, 0.4), (0.1, 0.2)) is True
    assert intervals_disjoint((0.1, 0.35), (0.3, 0.4)) is False
    assert intervals_disjoint((0.4, 0.3), (0.2, 0.1)) is True  # unordered tuples handled


def test_equal_tail_interval_matches_scipy() -> None:
    lo, hi = equal_tail_interval(4.0, 6.0, 0.9)
    assert lo == pytest.approx(stats.beta.ppf(0.05, 4.0, 6.0))
    assert hi == pytest.approx(stats.beta.ppf(0.95, 4.0, 6.0))
    with pytest.raises(ValueError):
        equal_tail_interval(0.0, 1.0, 0.9)


def test_hpd_symmetric_equals_equal_tail() -> None:
    assert hpd_interval(5.0, 5.0, 0.9) == pytest.approx(equal_tail_interval(5.0, 5.0, 0.9))


def test_hpd_is_narrower_for_skewed() -> None:
    a, b, level = 2.0, 8.0, 0.9
    hpd = hpd_interval(a, b, level)
    eqt = equal_tail_interval(a, b, level)
    hpd_w = hpd[1] - hpd[0]
    eqt_w = eqt[1] - eqt[0]
    assert hpd_w <= eqt_w + 1e-9
    # the HPD still carries (approximately) the requested mass
    mass = stats.beta.cdf(hpd[1], a, b) - stats.beta.cdf(hpd[0], a, b)
    assert mass == pytest.approx(level, abs=1e-3)
