"""Reliability-science metrics."""

from __future__ import annotations

from .bayes import (
    PRIOR_PRESETS,
    BetaPosterior,
    DirichletPosterior,
    beta_posterior,
    dirichlet_posterior,
)
from .capability import pass_at_k, pass_pow_k
from .reliability import (
    MOP_PAPER_DEFAULTS,
    graceful_degradation_score,
    meltdown_onset_point,
    reliability_decay_curve,
    reliability_decay_slope,
    variance_amplification_factor,
    window_entropy_curve,
)

__all__ = [
    "MOP_PAPER_DEFAULTS",
    "PRIOR_PRESETS",
    "BetaPosterior",
    "DirichletPosterior",
    "beta_posterior",
    "dirichlet_posterior",
    "graceful_degradation_score",
    "meltdown_onset_point",
    "pass_at_k",
    "pass_pow_k",
    "reliability_decay_curve",
    "reliability_decay_slope",
    "variance_amplification_factor",
    "window_entropy_curve",
]
