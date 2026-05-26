"""CLI: the ``ci`` subcommand's report, gating exit codes, and $GITHUB_OUTPUT emission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from passwedge.cli import main


def _write_trials(tmp_path: Path) -> Path:
    data = [
        {"task_id": "t1", "duration_bucket": "short", "outcomes": [True, True, True, False]},
        {"task_id": "t2", "duration_bucket": "long", "outcomes": [True, False, False, False]},
    ]
    p = tmp_path / "trials.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_ci_passes_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _write_trials(tmp_path)
    code = main(
        ["ci", "--input", str(p), "--k", "1,2", "--metric", "pass_at_1", "--fail-under", "0.4"]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "passwedge reliability report" in out


def test_ci_fails_threshold(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _write_trials(tmp_path)
    code = main(
        ["ci", "--input", str(p), "--k", "1,2", "--metric", "pass_pow_k", "--fail-under", "0.5"]
    )
    assert code == 1
    assert "< --fail-under" in capsys.readouterr().err


def test_ci_writes_report_and_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = _write_trials(tmp_path)
    report = tmp_path / "report.md"
    gh = tmp_path / "gh_out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    code = main(["ci", "--input", str(p), "--k", "1,2", "--output", str(report), "--github-output"])
    assert code == 0
    assert report.read_text(encoding="utf-8").startswith("<!-- passwedge-report -->")
    body = gh.read_text(encoding="utf-8")
    assert "pass_at_1=0.500000" in body
    assert "pass_pow_k=0.250000" in body
    assert "passed=true" in body


def test_version_exits(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "passwedge" in capsys.readouterr().out
