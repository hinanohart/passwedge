"""RDC / VAF / GDS / MOP: analytic checks, an independent entropy reference, and the
fail-closed contract on MOP thresholds."""

from __future__ import annotations

import math

import pytest

from passwedge.metrics.reliability import (
    MOP_PAPER_DEFAULTS,
    graceful_degradation_score,
    meltdown_onset_point,
    reliability_decay_curve,
    reliability_decay_slope,
    variance_amplification_factor,
    window_entropy_curve,
)


# ---- independent reference for the sliding-window entropy ----
def ref_curve(actions: list[str], w: int) -> list[float]:
    seq = list(actions)
    out: list[float] = []
    for t in range(len(seq)):
        window = seq[max(0, t - w + 1) : t + 1]
        counts: dict[str, int] = {}
        for a in window:
            counts[a] = counts.get(a, 0) + 1
        h = 0.0
        for cnt in counts.values():
            p = cnt / w
            if p > 0:
                h -= p * math.log2(p)
        out.append(h)
    return out


# ---- RDC / RDS ----
def test_reliability_decay_curve() -> None:
    curve = reliability_decay_curve({"short": [(4, 3), (4, 4)], "long": [(4, 1)]}, k=2)
    assert curve["short"] == pytest.approx((0.5 + 1.0) / 2)  # pass^2(4,3)=3/6, pass^2(4,4)=1
    assert curve["long"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        reliability_decay_curve({"empty": []}, k=1)


def test_reliability_decay_slope() -> None:
    assert reliability_decay_slope([1.0, 0.5, 0.0]) == pytest.approx(-0.5)
    assert reliability_decay_slope([0.2, 0.2, 0.2]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        reliability_decay_slope([0.5])


# ---- VAF ----
def test_variance_amplification_factor() -> None:
    vaf = variance_amplification_factor([0.0, 0.5, 1.0], [0.4, 0.5, 0.6])
    assert vaf == pytest.approx(0.25 / 0.01)
    assert math.isinf(variance_amplification_factor([0.0, 0.5, 1.0], [0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        variance_amplification_factor([0.5], [0.5, 0.6])  # too few for ddof=1


# ---- GDS ----
def test_graceful_degradation_score_mapping() -> None:
    gds = graceful_degradation_score(
        {"a": True, "b": False, "c": True}, {"a": 0.5, "b": 0.3, "c": 0.2}
    )
    assert gds == pytest.approx(0.7)


def test_graceful_degradation_score_sequence_uniform() -> None:
    assert graceful_degradation_score([True, False, True]) == pytest.approx(2 / 3)


def test_graceful_degradation_score_normalizes_weights() -> None:
    # unnormalized weights are renormalized to sum to 1
    assert graceful_degradation_score([True, False], [3.0, 1.0]) == pytest.approx(0.75)


def test_graceful_degradation_score_errors() -> None:
    with pytest.raises(ValueError):
        graceful_degradation_score([True, False], [-1.0, 2.0])
    with pytest.raises(ValueError):
        graceful_degradation_score({"a": True}, {"b": 1.0})  # missing key
    with pytest.raises(ValueError):
        graceful_degradation_score([])
    with pytest.raises(ValueError):
        graceful_degradation_score([True, False], [1.0])  # length mismatch
    with pytest.raises(TypeError):
        graceful_degradation_score({"a": True}, [1.0])  # type mismatch


# ---- window entropy + MOP ----
def test_window_entropy_matches_reference() -> None:
    for seq in (["a", "a", "a"], ["a", "b", "a", "c", "b"], list("aaaaabcdef")):
        got = window_entropy_curve(seq, 5)
        assert got == pytest.approx(ref_curve(seq, 5), abs=1e-12)


def test_window_entropy_uniform_full_window_is_zero() -> None:
    curve = window_entropy_curve(["x"] * 10, 5)
    assert curve[-1] == pytest.approx(0.0)  # full window of one tool -> 0 bits


def test_window_entropy_invalid_w() -> None:
    with pytest.raises(ValueError):
        window_entropy_curve(["a"], 0)


def test_mop_no_meltdown_is_inf() -> None:
    assert meltdown_onset_point(["a"] * 12, w=5, theta_h=1.711, delta=0.0) == math.inf


def test_mop_detects_diversity_spike() -> None:
    seq = list("aaaaabcdef")  # entropy rises as tool diversity explodes
    curve = ref_curve(seq, 5)
    theta_h, delta, w = 1.711, 0.0, 5
    expected = math.inf
    for t, h in enumerate(curve):
        h_prev = curve[t - w] if t - w >= 0 else 0.0
        if h > theta_h and (h - h_prev) > delta:
            expected = float(t)
            break
    assert meltdown_onset_point(seq, w=w, theta_h=theta_h, delta=delta) == expected
    assert math.isfinite(expected)  # this sequence does melt down


def test_mop_requires_explicit_thresholds() -> None:
    # fail-closed: thresholds are keyword-only with no defaults
    with pytest.raises(TypeError):
        meltdown_onset_point(["a", "b", "c"])  # type: ignore[call-arg]


def test_mop_paper_defaults_constant() -> None:
    assert MOP_PAPER_DEFAULTS["theta_h"] == 1.711
    assert MOP_PAPER_DEFAULTS["delta"] == 0.0
    assert MOP_PAPER_DEFAULTS["w"] == 5
