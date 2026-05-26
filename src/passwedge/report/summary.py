"""Aggregate a set of Trials into a reliability summary (the report data model)."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from ..data import Trial
from ..metrics.bayes import beta_posterior
from ..metrics.capability import pass_at_k, pass_pow_k
from ..metrics.reliability import reliability_decay_curve, reliability_decay_slope

__all__ = ["KStat", "ReliabilitySummary", "summarize"]


@dataclass(slots=True, frozen=True)
class KStat:
    """Macro-averaged pass@k and pass^k at a given k."""

    k: int
    pass_at_k: float
    pass_pow_k: float


@dataclass(slots=True)
class ReliabilitySummary:
    """Headline reliability statistics over a set of tasks."""

    n_tasks: int
    total_attempts: int
    total_successes: int
    k_stats: list[KStat]
    main_k: int
    posterior_mean: float
    credible_interval: tuple[float, float]
    credible_level: float
    rdc: dict[str, float] = field(default_factory=dict)
    rds: float | None = None

    @property
    def pass_at_1(self) -> float:
        return self.total_successes / self.total_attempts if self.total_attempts else 0.0

    @property
    def main_pass_pow_k(self) -> float:
        for s in self.k_stats:
            if s.k == self.main_k:
                return s.pass_pow_k
        raise KeyError(f"main_k={self.main_k} not in computed k stats")


def summarize(
    trials: list[Trial],
    ks: tuple[int, ...] = (1, 2, 5),
    main_k: int | None = None,
    prior: str = "jeffreys",
    credible_level: float = 0.95,
) -> ReliabilitySummary:
    """Compute macro-averaged pass@k / pass^k, a pooled Bayesian posterior, and (if the
    trials carry ``duration_bucket`` labels) the RDC curve and its slope.

    ``main_k`` defaults to the largest feasible requested k. Tasks with fewer than k
    attempts are skipped for that k (so pass@k stays well defined).
    """
    if not trials:
        raise ValueError("need at least one trial to summarize")
    if any(k < 1 for k in ks):
        raise ValueError("all k must be >= 1")

    total_n = sum(t.n for t in trials)
    total_c = sum(t.c for t in trials)

    k_stats: list[KStat] = []
    for k in sorted(set(ks)):
        at_vals = [pass_at_k(t.n, t.c, k) for t in trials if t.n >= k]
        pow_vals = [pass_pow_k(t.n, t.c, k) for t in trials if t.n >= k]
        if not at_vals:
            continue
        k_stats.append(
            KStat(
                k=k,
                pass_at_k=sum(at_vals) / len(at_vals),
                pass_pow_k=sum(pow_vals) / len(pow_vals),
            )
        )
    if not k_stats:
        raise ValueError("no task has enough attempts for any requested k")

    available_ks = [s.k for s in k_stats]
    resolved_main_k = main_k if main_k is not None else max(available_ks)
    if resolved_main_k not in available_ks:
        raise ValueError(
            f"main_k={resolved_main_k} has no feasible tasks; available {available_ks}"
        )

    posterior = beta_posterior(total_c, total_n, prior=prior)
    ci = posterior.credible_interval(credible_level)

    rdc: dict[str, float] = {}
    rds: float | None = None
    bucketed = [t for t in trials if t.duration_bucket is not None]
    if bucketed:
        by_bucket: OrderedDict[str, list[tuple[int, int]]] = OrderedDict()
        for t in bucketed:
            assert t.duration_bucket is not None
            by_bucket.setdefault(t.duration_bucket, []).append((t.n, t.c))
        feasible = OrderedDict(
            (b, pairs)
            for b, pairs in by_bucket.items()
            if all(n >= resolved_main_k for n, _ in pairs)
        )
        if feasible:
            rdc = reliability_decay_curve(feasible, resolved_main_k)
            if len(rdc) >= 2:
                rds = reliability_decay_slope(list(rdc.values()))

    return ReliabilitySummary(
        n_tasks=len(trials),
        total_attempts=total_n,
        total_successes=total_c,
        k_stats=k_stats,
        main_k=resolved_main_k,
        posterior_mean=posterior.mean(),
        credible_interval=ci,
        credible_level=credible_level,
        rdc=rdc,
        rds=rds,
    )
