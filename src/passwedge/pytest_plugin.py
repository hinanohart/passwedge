"""pytest plugin: ``@pytest.mark.passk`` runs a flaky test repeatedly and asserts a
pass@k / pass^k reliability threshold instead of a single all-or-nothing outcome.

Usage::

    @pytest.mark.passk(attempts=20, k=5, min_pass_pow_k=0.9)
    def test_agent_completes_task():
        assert run_agent().solved

The body is executed ``attempts`` times; an attempt fails iff it raises. The plugin then
checks the requested thresholds (``min_pass_at_k`` for >=1-of-k, ``min_pass_pow_k`` for
all-of-k) and fails the single test item with a summary if any threshold is unmet.
"""

from __future__ import annotations

from typing import Any

import pytest

from .metrics.capability import pass_at_k, pass_pow_k


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "passk(attempts, k, min_pass_at_k, min_pass_pow_k): repeat a test and assert "
        "a pass@k / pass^k reliability threshold.",
    )


def _resolve_kwargs(marker: pytest.Mark) -> tuple[int, int, float | None, float | None]:
    attempts = int(marker.kwargs.get("attempts", marker.kwargs.get("n", 10)))
    k = int(marker.kwargs.get("k", 1))
    min_at = marker.kwargs.get("min_pass_at_k")
    min_pow = marker.kwargs.get("min_pass_pow_k")
    if attempts < 1:
        raise ValueError("passk: attempts must be >= 1")
    if k < 1 or k > attempts:
        raise ValueError("passk: require 1 <= k <= attempts")
    if min_at is None and min_pow is None:
        raise ValueError("passk: set min_pass_at_k and/or min_pass_pow_k")
    return (
        attempts,
        k,
        (None if min_at is None else float(min_at)),
        (None if min_pow is None else float(min_pow)),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    marker = pyfuncitem.get_closest_marker("passk")
    if marker is None:
        return None  # not ours; let pytest handle normally
    attempts, k, min_at, min_pow = _resolve_kwargs(marker)

    funcargs: dict[str, Any] = pyfuncitem.funcargs
    argnames = pyfuncitem._fixtureinfo.argnames
    testargs = {name: funcargs[name] for name in argnames}

    successes = 0
    for _ in range(attempts):
        try:
            pyfuncitem.obj(**testargs)
        except Exception:
            continue
        successes += 1

    pa = pass_at_k(attempts, successes, k)
    pp = pass_pow_k(attempts, successes, k)
    problems: list[str] = []
    if min_at is not None and pa < min_at:
        problems.append(f"pass@{k}={pa:.4f} < required {min_at:.4f}")
    if min_pow is not None and pp < min_pow:
        problems.append(f"pass^{k}={pp:.4f} < required {min_pow:.4f}")
    if problems:
        raise AssertionError(
            f"passk: {successes}/{attempts} attempts passed; " + "; ".join(problems)
        )
    return True  # we executed the test ourselves
