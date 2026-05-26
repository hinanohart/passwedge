"""Data model: coercion gradation and Trial sufficient statistics."""

from __future__ import annotations

import pytest

from passwedge.data import Episode, Step, Trial, coerce_trial


def test_coerce_from_bools() -> None:
    t = coerce_trial([True, False, True], task_id="t1", duration_bucket="short")
    assert t.n == 3
    assert t.c == 2
    assert t.outcomes == [True, False, True]
    assert t.task_id == "t1"
    assert t.duration_bucket == "short"


def test_coerce_from_ints() -> None:
    t = coerce_trial([1, 0, 1, 1])
    assert t.n == 4 and t.c == 3
    with pytest.raises(ValueError):
        coerce_trial([0, 2, 1])  # 2 is not a valid outcome


def test_coerce_from_episodes() -> None:
    eps = [Episode(success=True), Episode(success=False)]
    t = coerce_trial(eps)
    assert t.n == 2 and t.c == 1


def test_coerce_passthrough_trial() -> None:
    original = Trial(episodes=[Episode(success=True)], task_id="x")
    assert coerce_trial(original) is original


def test_coerce_rejects_mapping_and_str() -> None:
    with pytest.raises(TypeError):
        coerce_trial({"outcomes": [True]})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        coerce_trial("TTF")  # type: ignore[arg-type]


def test_episode_tool_sequence() -> None:
    ep = Episode(success=True, steps=[Step(tool="bash"), Step(tool="edit")])
    assert ep.tool_sequence == ["bash", "edit"]


def test_mixed_episode_items_rejected() -> None:
    with pytest.raises(TypeError):
        coerce_trial([Episode(success=True), object()])  # type: ignore[list-item]
