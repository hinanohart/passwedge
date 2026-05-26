"""Generic readers: raw outcomes and JSON task records -> :class:`Trial` objects.

This is the first-class input path. A JSON task record may be *rich*::

    {"task_id": "t1", "duration_bucket": "long",
     "episodes": [{"success": true, "steps": [{"tool": "bash"}],
                   "subtask_results": {"s1": true}}, ...]}

or *simple*::

    {"task_id": "t1", "outcomes": [true, false, true]}

A whole file is a JSON list of such records.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..data import Episode, Step, Trial, coerce_trial

__all__ = ["from_outcomes", "load_trials", "read_trials_file", "trial_from_record"]


def from_outcomes(
    outcomes: Sequence[bool] | Sequence[int],
    *,
    task_id: str | None = None,
    duration_bucket: str | None = None,
) -> Trial:
    """Build a :class:`Trial` from a flat list of pass/fail outcomes."""
    return coerce_trial(outcomes, task_id=task_id, duration_bucket=duration_bucket)


def _parse_step(raw: Any) -> Step:
    if not isinstance(raw, dict):
        raise ValueError(f"step must be an object, got {type(raw).__name__}")
    if "tool" not in raw:
        raise ValueError("step object requires a 'tool' field")
    success = raw.get("success")
    if success is not None and not isinstance(success, bool):
        raise ValueError("step 'success' must be a boolean or null")
    meta = raw.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("step 'meta' must be an object")
    return Step(tool=str(raw["tool"]), success=success, meta=dict(meta))


def _parse_episode(raw: Any) -> Episode:
    if not isinstance(raw, dict):
        raise ValueError(f"episode must be an object, got {type(raw).__name__}")
    if "success" not in raw:
        raise ValueError("episode object requires a 'success' field")
    if not isinstance(raw["success"], bool):
        raise ValueError("episode 'success' must be a boolean")
    steps = [_parse_step(s) for s in raw.get("steps", [])]
    subtasks_raw = raw.get("subtask_results", {})
    if not isinstance(subtasks_raw, dict):
        raise ValueError("'subtask_results' must be an object")
    subtask_results = {str(k): bool(v) for k, v in subtasks_raw.items()}
    return Episode(
        success=raw["success"],
        steps=steps,
        subtask_results=subtask_results,
        task_id=None,
    )


def trial_from_record(record: Any) -> Trial:
    """Parse one JSON task record (rich or simple form) into a :class:`Trial`."""
    if not isinstance(record, dict):
        raise ValueError(f"task record must be an object, got {type(record).__name__}")
    task_id = record.get("task_id")
    bucket = record.get("duration_bucket")
    if "episodes" in record:
        episodes = [_parse_episode(e) for e in record["episodes"]]
        for e in episodes:
            e.task_id = task_id if isinstance(task_id, str) else None
        return Trial(
            episodes=episodes,
            task_id=task_id if isinstance(task_id, str) else None,
            duration_bucket=bucket if isinstance(bucket, str) else None,
        )
    if "outcomes" in record:
        return coerce_trial(
            list(record["outcomes"]),
            task_id=task_id if isinstance(task_id, str) else None,
            duration_bucket=bucket if isinstance(bucket, str) else None,
        )
    raise ValueError("task record must contain either 'episodes' or 'outcomes'")


def load_trials(data: Any) -> list[Trial]:
    """Parse a JSON-decoded list of task records into a list of :class:`Trial`."""
    if not isinstance(data, list):
        raise ValueError("trials JSON must be a list of task records")
    return [trial_from_record(rec) for rec in data]


def read_trials_file(path: str | Path) -> list[Trial]:
    """Read and parse a JSON file of task records."""
    text = Path(path).read_text(encoding="utf-8")
    return load_trials(json.loads(text))
