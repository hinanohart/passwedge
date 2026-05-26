"""Adapters: generic JSON (rich + simple) and the tetrad-lens trace reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from passwedge.adapters.generic import (
    from_outcomes,
    load_trials,
    read_trials_file,
    trial_from_record,
)
from passwedge.adapters.tetrad_lens import trials_from_trace_records


def test_from_outcomes() -> None:
    t = from_outcomes([True, False, True], task_id="t1")
    assert t.n == 3 and t.c == 2 and t.task_id == "t1"


def test_trial_from_rich_record() -> None:
    rec = {
        "task_id": "t1",
        "duration_bucket": "long",
        "episodes": [
            {"success": True, "steps": [{"tool": "bash"}], "subtask_results": {"s1": True}},
            {"success": False},
        ],
    }
    t = trial_from_record(rec)
    assert t.n == 2 and t.c == 1
    assert t.duration_bucket == "long"
    assert t.episodes[0].tool_sequence == ["bash"]
    assert t.episodes[0].subtask_results == {"s1": True}
    assert t.episodes[0].task_id == "t1"


def test_trial_from_simple_record() -> None:
    t = trial_from_record({"task_id": "t2", "outcomes": [True, True, False]})
    assert t.n == 3 and t.c == 2


def test_record_errors() -> None:
    with pytest.raises(ValueError):
        trial_from_record({"task_id": "x"})  # neither episodes nor outcomes
    with pytest.raises(ValueError):
        trial_from_record([1, 2, 3])  # not an object
    with pytest.raises(ValueError):
        trial_from_record({"episodes": [{"steps": []}]})  # missing success
    with pytest.raises(ValueError):
        trial_from_record({"episodes": [{"success": "yes"}]})  # success not bool


def test_load_and_roundtrip_file(tmp_path: Path) -> None:
    data = [
        {"task_id": "t1", "outcomes": [True, False]},
        {"task_id": "t2", "episodes": [{"success": True}]},
    ]
    p = tmp_path / "trials.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    trials = read_trials_file(p)
    assert len(trials) == 2
    assert [t.task_id for t in trials] == ["t1", "t2"]
    assert load_trials(data)[0].n == 2


def test_load_trials_requires_list() -> None:
    with pytest.raises(ValueError):
        load_trials({"not": "a list"})


def test_tetrad_lens_grouping() -> None:
    records = [
        {"task": "t1", "verdict": "success", "bucket": "short"},
        {"task": "t1", "verdict": "fail", "bucket": "short"},
        {"task": "t2", "verdict": True, "bucket": "long"},
    ]
    trials = trials_from_trace_records(
        records, task_key="task", success_key="verdict", bucket_key="bucket"
    )
    assert len(trials) == 2
    by_id = {t.task_id: t for t in trials}
    assert by_id["t1"].n == 2 and by_id["t1"].c == 1
    assert by_id["t1"].duration_bucket == "short"
    assert by_id["t2"].c == 1 and by_id["t2"].duration_bucket == "long"


def test_tetrad_lens_dotted_path_and_custom_success() -> None:
    records = [
        {"meta": {"task_id": "a"}, "result": {"status": "ok"}},
        {"meta": {"task_id": "a"}, "result": {"status": "error"}},
    ]
    trials = trials_from_trace_records(
        records,
        task_key="meta.task_id",
        success_key="result.status",
        success_values=["ok"],
    )
    assert trials[0].n == 2 and trials[0].c == 1


def test_tetrad_lens_errors() -> None:
    with pytest.raises(ValueError):
        trials_from_trace_records([42], task_key="t", success_key="s")  # not an object
    with pytest.raises(KeyError):
        trials_from_trace_records([{"t": "x"}], task_key="t", success_key="missing")
