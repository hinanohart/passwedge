"""Domain-agnostic data model for repeated-attempt evaluation.

Three layers of increasing granularity (largest first):

* :class:`Trial`   -- the repeated attempts for a *single task* (the unit for
  pass@k / pass^k / Bayesian posteriors).
* :class:`Episode` -- one attempt (rollout) at the task: a success flag plus,
  optionally, a tool-call trace and per-subtask results.
* :class:`Step`    -- one action / tool call inside an episode (used by MOP).

You can start from a bare ``list[bool]`` and add structure only where a metric
needs it. :func:`coerce_trial` accepts the whole gradation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = ["Episode", "Step", "Trial", "TrialLike", "coerce_trial"]


@dataclass(slots=True)
class Step:
    """One action inside an episode. ``tool`` names the tool/action invoked."""

    tool: str
    success: bool | None = None
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Episode:
    """A single attempt at a task.

    ``success`` is the programmatic verdict E(W) for the whole attempt.
    ``steps`` is the optional ordered tool-call trace (for MOP).
    ``subtask_results`` maps subtask id -> completed-correctly (for GDS).
    """

    success: bool
    steps: list[Step] = field(default_factory=list)
    subtask_results: dict[str, bool] = field(default_factory=dict)
    task_id: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def tool_sequence(self) -> list[str]:
        """Ordered tool names across this episode's steps."""
        return [s.tool for s in self.steps]


@dataclass(slots=True)
class Trial:
    """The repeated attempts (episodes) for one task.

    ``n`` = number of attempts, ``c`` = number of successes -- the sufficient
    statistics for the capability and Bayesian metrics.
    """

    episodes: list[Episode] = field(default_factory=list)
    task_id: str | None = None
    duration_bucket: str | None = None

    @property
    def n(self) -> int:
        return len(self.episodes)

    @property
    def c(self) -> int:
        return sum(1 for e in self.episodes if e.success)

    @property
    def outcomes(self) -> list[bool]:
        return [e.success for e in self.episodes]


# A Trial may be supplied directly, as a list of Episodes, or as raw outcomes
# (bools or 0/1 ints). dict form is handled by the JSON adapter, not here.
TrialLike = Trial | Sequence[Episode] | Sequence[bool] | Sequence[int]


def _coerce_outcome(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"integer outcome must be 0 or 1, got {value!r}")
    raise TypeError(f"cannot coerce {type(value).__name__} to a boolean outcome")


def coerce_trial(
    data: TrialLike,
    *,
    task_id: str | None = None,
    duration_bucket: str | None = None,
) -> Trial:
    """Normalize any supported input into a :class:`Trial`.

    Accepts a :class:`Trial`, a sequence of :class:`Episode`, or a sequence of
    raw outcomes (``bool`` or ``0``/``1`` ints). Mappings are rejected here --
    use the adapters in :mod:`passwedge.adapters` for structured JSON.
    """
    if isinstance(data, Trial):
        return data
    if isinstance(data, Mapping):
        raise TypeError("use passwedge.adapters for mapping/JSON input, not coerce_trial")
    if isinstance(data, (str, bytes)):
        raise TypeError("a string is not a valid Trial input")
    items = list(data)
    if items and isinstance(items[0], Episode):
        episodes = []
        for e in items:
            if not isinstance(e, Episode):
                raise TypeError("mixed Episode and non-Episode items are not allowed")
            episodes.append(e)
    else:
        episodes = [Episode(success=_coerce_outcome(x)) for x in items]
    return Trial(episodes=episodes, task_id=task_id, duration_bucket=duration_bucket)
