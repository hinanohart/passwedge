"""Reliability metrics from *Beyond pass@1* (arXiv:2603.29231): RDC, VAF, GDS, MOP.

Formulas are implemented exactly as defined in the paper (Definitions 3-6); see
``docs/DEFINITIONS.md`` for the verbatim equations and provenance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .capability import pass_pow_k

__all__ = [
    "MOP_PAPER_DEFAULTS",
    "graceful_degradation_score",
    "meltdown_onset_point",
    "reliability_decay_curve",
    "reliability_decay_slope",
    "variance_amplification_factor",
    "window_entropy_curve",
]

# arXiv:2603.29231 Section 6.4 / Appendix A.2. Calibrated on the paper's 396-task
# benchmark and therefore DATASET-SPECIFIC -- re-calibrate before reusing verbatim.
MOP_PAPER_DEFAULTS: dict[str, float] = {"theta_h": 1.711, "delta": 0.000, "w": 5}


def reliability_decay_curve(
    trials_by_bucket: Mapping[str, Sequence[tuple[int, int]]],
    k: int,
    estimator: str = "unbiased",
) -> dict[str, float]:
    """RDC (Def. 3): map each duration bucket to its mean pass^k.

    ``trials_by_bucket`` maps a bucket label to a sequence of ``(n, c)`` pairs, one
    per task in that bucket. Returns ``{bucket: mean pass^k over its tasks}``.
    """
    curve: dict[str, float] = {}
    for bucket, tasks in trials_by_bucket.items():
        if not tasks:
            raise ValueError(f"bucket {bucket!r} has no tasks")
        vals = [pass_pow_k(n, c, k, estimator=estimator) for (n, c) in tasks]
        curve[bucket] = float(np.mean(vals))
    return curve


def reliability_decay_slope(values: Sequence[float]) -> float:
    """RDS: OLS slope of per-bucket values against bucket index b = 0, 1, 2, ...

    ``sum_b (b - b_bar)(y_b - y_bar) / sum_b (b - b_bar)^2``. Supply per-bucket pass^k
    (to summarize the RDC) or per-bucket GDS (to match the paper's RDS equation).
    """
    y = np.asarray(values, dtype=np.float64)
    if y.size < 2:
        raise ValueError("need at least two buckets to fit a slope")
    b = np.arange(y.size, dtype=np.float64)
    b_dev = b - b.mean()
    denom = float(np.sum(b_dev * b_dev))
    if denom == 0.0:  # unreachable for distinct indices, guarded for safety
        raise ValueError("degenerate bucket indices")
    return float(np.sum(b_dev * (y - y.mean())) / denom)


def variance_amplification_factor(
    pass1_long: Sequence[float], pass1_short: Sequence[float], ddof: int = 1
) -> float:
    """VAF (Def. 4): var(pass@1 | long) / var(pass@1 | short), across task instances.

    Each input is the per-task pass@1 (= c/n) for tasks in that duration bucket.
    Returns ``+inf`` if the short-bucket variance is 0.
    """
    long_arr = np.asarray(pass1_long, dtype=np.float64)
    short_arr = np.asarray(pass1_short, dtype=np.float64)
    if long_arr.size <= ddof or short_arr.size <= ddof:
        raise ValueError(f"each bucket needs more than ddof={ddof} tasks")
    var_short = float(np.var(short_arr, ddof=ddof))
    var_long = float(np.var(long_arr, ddof=ddof))
    if var_short == 0.0:
        return math.inf
    return var_long / var_short


def graceful_degradation_score(
    subtask_results: Mapping[str, bool] | Sequence[bool],
    weights: Mapping[str, float] | Sequence[float] | None = None,
) -> float:
    """GDS (Def. 5): weighted partial credit ``sum_i w_i * 1[subtask i correct]``.

    Weights are normalized to sum to 1. If ``weights`` is None, uniform weights are
    used. ``subtask_results`` and ``weights`` may both be mappings (keyed by subtask
    id) or both be ordered sequences.
    """
    if isinstance(subtask_results, Mapping):
        keys = list(subtask_results.keys())
        results = np.array([bool(subtask_results[key]) for key in keys], dtype=np.float64)
        if weights is None:
            w = np.ones(len(keys), dtype=np.float64)
        elif isinstance(weights, Mapping):
            missing = set(keys) - set(weights)
            if missing:
                raise ValueError(f"weights missing subtasks: {sorted(missing)}")
            w = np.array([float(weights[key]) for key in keys], dtype=np.float64)
        else:
            raise TypeError("weights must be a mapping when subtask_results is a mapping")
    else:
        results = np.array([bool(x) for x in subtask_results], dtype=np.float64)
        if weights is None:
            w = np.ones(results.size, dtype=np.float64)
        elif isinstance(weights, Mapping):
            raise TypeError("weights must be a sequence when subtask_results is a sequence")
        else:
            w = np.asarray(weights, dtype=np.float64)
    if results.size == 0:
        raise ValueError("need at least one subtask")
    if w.size != results.size:
        raise ValueError("weights and subtask_results must have the same length")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return float(np.dot(w / total, results))


def window_entropy_curve(actions: Sequence[object], w: int) -> list[float]:
    """H(t) for every step t: Shannon entropy (bits) of the tool distribution over the
    ``w`` most recent actions, with counts divided by ``w`` (Def. 6).

    Operationalization note: the paper writes the window as the closed interval
    ``[t-w, t]``; taken literally that is ``w+1`` points divided by ``w``, which can give
    probabilities > 1 and a negative "entropy". passwedge therefore uses a length-``w``
    window (the ``w`` most recent actions). Full windows give a proper distribution with
    ``H in [0, log2(#tools)]``; near the start (``t < w-1``) the window is shorter, so the
    counts sum to < 1 and ``H`` stays non-negative.
    """
    if w < 1:
        raise ValueError("window size w must be >= 1")
    seq = list(actions)
    curve: list[float] = []
    for t in range(len(seq)):
        lo = max(0, t - w + 1)
        window = seq[lo : t + 1]
        counts: dict[object, int] = {}
        for a in window:
            counts[a] = counts.get(a, 0) + 1
        h = 0.0
        for cnt in counts.values():
            p = cnt / w
            if p > 0.0:
                h -= p * math.log2(p)
        curve.append(h)
    return curve


def meltdown_onset_point(
    actions: Sequence[object], *, w: int, theta_h: float, delta: float
) -> float:
    """MOP (Def. 6): first step t* with ``H(t*) > theta_h`` and ``H(t*)-H(t*-w) > delta``.

    Returns ``math.inf`` if no meltdown is detected. Thresholds are **required** (no
    defaults): they are dataset-specific and applying the paper's values verbatim to
    other data produces false positives. Pass ``**MOP_PAPER_DEFAULTS`` to opt in to the
    paper's calibration. For ``t < w`` the prior entropy ``H(t-w)`` is taken as 0.
    """
    if w < 1:
        raise ValueError("window size w must be >= 1")
    curve = window_entropy_curve(actions, w)
    for t, h_t in enumerate(curve):
        h_prev = curve[t - w] if t - w >= 0 else 0.0
        if h_t > theta_h and (h_t - h_prev) > delta:
            return float(t)
    return math.inf
