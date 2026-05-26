"""The ``@pytest.mark.passk`` plugin, exercised through pytest's own ``pytester``."""

from __future__ import annotations

import pytest

from passwedge.pytest_plugin import _resolve_kwargs

pytestmark = pytest.mark.filterwarnings("ignore")


def _mark(**kwargs: object) -> pytest.Mark:
    return pytest.Mark("passk", (), kwargs)


def test_resolve_kwargs_defaults_and_aliases() -> None:
    assert _resolve_kwargs(_mark(min_pass_at_k=0.5)) == (10, 1, 0.5, None)
    assert _resolve_kwargs(_mark(n=20, k=5, min_pass_pow_k=0.9)) == (20, 5, None, 0.9)


def test_resolve_kwargs_validation() -> None:
    with pytest.raises(ValueError):
        _resolve_kwargs(_mark(attempts=0, min_pass_at_k=0.5))
    with pytest.raises(ValueError):
        _resolve_kwargs(_mark(attempts=5, k=6, min_pass_at_k=0.5))  # k > attempts
    with pytest.raises(ValueError):
        _resolve_kwargs(_mark(attempts=5, k=1))  # no threshold set


def test_passk_marker_passes_when_reliable(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.passk(attempts=10, k=1, min_pass_at_k=0.5)
        def test_always_ok():
            assert True
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1)


def test_passk_marker_fails_when_unreliable(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.passk(attempts=10, k=3, min_pass_pow_k=0.9)
        def test_always_fails():
            assert False
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*0/10 attempts passed*"])


def test_passk_marker_requires_threshold(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        import pytest

        @pytest.mark.passk(attempts=5, k=1)
        def test_no_threshold():
            assert True
        """
    )
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(failed=1)  # ValueError: set a threshold
