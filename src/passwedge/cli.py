"""``passwedge`` command-line interface. The ``ci`` subcommand turns a JSON file of task
outcomes into a Markdown report, emits ``$GITHUB_OUTPUT`` values, and exits non-zero when a
chosen metric falls below ``--fail-under`` (the GitHub Action's engine).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from . import __version__
from .adapters.generic import read_trials_file
from .metrics.bayes import PRIOR_PRESETS
from .report.markdown import render_markdown
from .report.summary import ReliabilitySummary, summarize

_METRIC_GETTERS: dict[str, Callable[[ReliabilitySummary], float]] = {
    "pass_at_1": lambda s: s.pass_at_1,
    "pass_pow_k": lambda s: s.main_pass_pow_k,
    "posterior_mean": lambda s: s.posterior_mean,
    "ci_lower": lambda s: s.credible_interval[0],
}


def _parse_ks(text: str) -> tuple[int, ...]:
    try:
        ks = tuple(int(p) for p in text.split(",") if p.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid --k list {text!r}") from None
    if not ks:
        raise argparse.ArgumentTypeError("--k must list at least one integer")
    return ks


def _credible_level(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--credible-level must be a number, got {text!r}"
        ) from None
    if not 0.0 < value < 1.0:
        raise argparse.ArgumentTypeError(f"--credible-level must be in (0, 1), got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="passwedge", description=__doc__)
    parser.add_argument("--version", action="version", version=f"passwedge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ci = sub.add_parser("ci", help="summarize a trials JSON file and gate on a threshold")
    ci.add_argument("--input", "-i", required=True, type=Path, help="JSON file of task records")
    ci.add_argument("--k", type=_parse_ks, default=(1, 2, 5), help="comma-separated k values")
    ci.add_argument("--main-k", type=int, default=None, help="headline k (default: largest k)")
    ci.add_argument(
        "--prior",
        choices=sorted(PRIOR_PRESETS),
        default="jeffreys",
        help="Bayesian prior preset (default: jeffreys)",
    )
    ci.add_argument("--credible-level", type=_credible_level, default=0.95)
    ci.add_argument("--metric", choices=sorted(_METRIC_GETTERS), default="pass_pow_k")
    ci.add_argument("--fail-under", type=float, default=None, help="exit 1 if metric < this")
    ci.add_argument("--output", "-o", type=Path, default=None, help="write Markdown report here")
    ci.add_argument("--title", default="passwedge reliability report")
    ci.add_argument(
        "--github-output",
        action="store_true",
        help="append key=value lines to the file named by $GITHUB_OUTPUT",
    )
    return parser


def _emit_github_output(summary: ReliabilitySummary, metric_value: float, passed: bool) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    lines = [
        f"pass_at_1={summary.pass_at_1:.6f}",
        f"pass_pow_k={summary.main_pass_pow_k:.6f}",
        f"posterior_mean={summary.posterior_mean:.6f}",
        f"ci_lower={summary.credible_interval[0]:.6f}",
        f"ci_upper={summary.credible_interval[1]:.6f}",
        f"metric_value={metric_value:.6f}",
        f"passed={'true' if passed else 'false'}",
    ]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _run_ci(args: argparse.Namespace) -> int:
    trials = read_trials_file(args.input)
    summary = summarize(
        trials,
        ks=args.k,
        main_k=args.main_k,
        prior=args.prior,
        credible_level=args.credible_level,
    )
    markdown = render_markdown(summary, title=args.title)
    if args.output is not None:
        args.output.write_text(markdown, encoding="utf-8")
    sys.stdout.write(markdown)

    metric_value = _METRIC_GETTERS[args.metric](summary)
    passed = args.fail_under is None or metric_value >= args.fail_under
    if args.github_output:
        _emit_github_output(summary, metric_value, passed)
    if not passed:
        sys.stderr.write(
            f"\npasswedge: {args.metric}={metric_value:.4f} < --fail-under={args.fail_under}\n"
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # "ci" is currently the only (required) subcommand.
    return _run_ci(args)


if __name__ == "__main__":
    raise SystemExit(main())
