"""passwedge -- reliability-science metrics for repeated-attempt LLM-agent evaluation.

Capability vs. reliability: pass@1 says whether a model *can* do a task once; pass^k and
the reliability metrics say whether it does so *consistently* across repeated attempts on
long-horizon tasks. See ``docs/DEFINITIONS.md`` for formulas and provenance.
"""

from __future__ import annotations

from .data import Episode, Step, Trial, coerce_trial
from .metrics import (
    MOP_PAPER_DEFAULTS,
    PRIOR_PRESETS,
    BetaPosterior,
    DirichletPosterior,
    beta_posterior,
    dirichlet_posterior,
    graceful_degradation_score,
    meltdown_onset_point,
    pass_at_k,
    pass_pow_k,
    reliability_decay_curve,
    reliability_decay_slope,
    variance_amplification_factor,
    window_entropy_curve,
)
from .report import ReliabilitySummary, render_markdown, summarize
from .stats import (
    equal_tail_interval,
    hpd_interval,
    intervals_disjoint,
    two_proportion_z_test,
)

__version__ = "0.0.1a2"

__all__ = [
    "MOP_PAPER_DEFAULTS",
    "PRIOR_PRESETS",
    "BetaPosterior",
    "DirichletPosterior",
    "Episode",
    "ReliabilitySummary",
    "Step",
    "Trial",
    "__version__",
    "beta_posterior",
    "coerce_trial",
    "dirichlet_posterior",
    "equal_tail_interval",
    "graceful_degradation_score",
    "hpd_interval",
    "intervals_disjoint",
    "meltdown_onset_point",
    "pass_at_k",
    "pass_pow_k",
    "reliability_decay_curve",
    "reliability_decay_slope",
    "render_markdown",
    "summarize",
    "two_proportion_z_test",
    "variance_amplification_factor",
    "window_entropy_curve",
]
