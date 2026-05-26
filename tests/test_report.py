"""Report: aggregation summary and Markdown rendering."""

from __future__ import annotations

import pytest

from passwedge.adapters.generic import from_outcomes
from passwedge.data import Trial
from passwedge.report.markdown import COMMENT_MARKER, render_markdown
from passwedge.report.summary import summarize


def _trials() -> list[Trial]:
    return [
        from_outcomes([True, True, True, False], task_id="t1", duration_bucket="short"),
        from_outcomes([True, False, False, False], task_id="t2", duration_bucket="long"),
    ]


def test_summarize_basic() -> None:
    s = summarize(_trials(), ks=(1, 2), prior="jeffreys")
    assert s.n_tasks == 2
    assert s.total_attempts == 8
    assert s.total_successes == 4
    assert s.pass_at_1 == pytest.approx(0.5)
    assert s.main_k == 2
    # macro pass^2: t1 = C(3,2)/C(4,2)=0.5, t2 = 0 -> mean 0.25
    assert s.main_pass_pow_k == pytest.approx(0.25)
    lo, hi = s.credible_interval
    assert lo < s.posterior_mean < hi
    # bucketed -> RDC present with a (negative) decay slope
    assert set(s.rdc) == {"short", "long"}
    assert s.rds is not None and s.rds < 0


def test_summarize_no_buckets() -> None:
    s = summarize([from_outcomes([True, False, True])], ks=(1,))
    assert s.rdc == {}
    assert s.rds is None


def test_summarize_errors() -> None:
    with pytest.raises(ValueError):
        summarize([], ks=(1,))
    with pytest.raises(ValueError):
        summarize([from_outcomes([True, False])], ks=(5,))  # no task has >=5 attempts


def test_render_markdown_contains_marker_and_table() -> None:
    s = summarize(_trials(), ks=(1, 2))
    md = render_markdown(s)
    assert md.startswith(COMMENT_MARKER)
    assert "pass@k" in md and "pass^k" in md
    assert "Reliability Decay Curve" in md
    assert "arXiv:2603.29231" in md
    assert "⬅ main" in md
