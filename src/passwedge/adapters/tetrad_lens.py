"""Thin reader for tetrad-lens (and any JSON trace) exports -- no package dependency.

tetrad-lens (hinanohart/tetrad-lens) tags agent traces with McLuhan-tetrad attributes;
it records *trace attributes*, not pass/fail outcomes. This reader therefore consumes a
list of trace records (as plain JSON) and lets you say which field carries the per-episode
success verdict and which groups records into tasks. Coupling is by data shape only, so
tetrad-lens can evolve its schema without breaking passwedge.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from ..data import Episode, Trial

__all__ = ["DEFAULT_SUCCESS_VALUES", "trials_from_trace_records"]

DEFAULT_SUCCESS_VALUES: tuple[object, ...] = (True, 1, "success", "pass", "passed", "ok")


def _dotted_get(record: dict[str, Any], path: str) -> Any:
    cur: Any = record
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"field path {path!r} not found in record")
        cur = cur[part]
    return cur


def trials_from_trace_records(
    records: Sequence[Any],
    *,
    task_key: str,
    success_key: str,
    bucket_key: str | None = None,
    success_values: Sequence[object] = DEFAULT_SUCCESS_VALUES,
) -> list[Trial]:
    """Group trace records into one :class:`Trial` per task.

    Each record becomes one :class:`Episode`. ``task_key``/``success_key``/``bucket_key``
    are dotted field paths. An episode succeeds iff its ``success_key`` value is in
    ``success_values``. Trials preserve first-seen task order.
    """
    grouped: OrderedDict[str, list[Episode]] = OrderedDict()
    buckets: dict[str, str | None] = {}
    success_set = set(success_values)
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"trace record must be an object, got {type(rec).__name__}")
        task_id = str(_dotted_get(rec, task_key))
        verdict = _dotted_get(rec, success_key)
        success = verdict in success_set
        grouped.setdefault(task_id, []).append(Episode(success=success, task_id=task_id))
        if bucket_key is not None and task_id not in buckets:
            bucket_val = _dotted_get(rec, bucket_key)
            buckets[task_id] = str(bucket_val)
    return [
        Trial(episodes=eps, task_id=task_id, duration_bucket=buckets.get(task_id))
        for task_id, eps in grouped.items()
    ]
