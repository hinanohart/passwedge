"""Render a :class:`ReliabilitySummary` as Markdown (the GitHub Action's PR comment)."""

from __future__ import annotations

from .summary import ReliabilitySummary

__all__ = ["COMMENT_MARKER", "render_markdown"]

# Stable marker so the GitHub Action can find and upsert its own comment.
COMMENT_MARKER = "<!-- passwedge-report -->"


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def render_markdown(
    summary: ReliabilitySummary, *, title: str = "passwedge reliability report"
) -> str:
    """Render a compact Markdown report. The first line is :data:`COMMENT_MARKER`."""
    lo, hi = summary.credible_interval
    lines: list[str] = [
        COMMENT_MARKER,
        f"## {title}",
        "",
        f"- **Tasks:** {summary.n_tasks} &nbsp;|&nbsp; "
        f"**Attempts:** {summary.total_attempts} &nbsp;|&nbsp; "
        f"**pass@1:** {_pct(summary.pass_at_1)}",
        f"- **Posterior mean** ({int(summary.credible_level * 100)}% CI): "
        f"{_pct(summary.posterior_mean)} [{_pct(lo)}, {_pct(hi)}]",
        "",
        "| k | pass@k (≥1 of k) | pass^k (all of k) |",
        "| ---: | ---: | ---: |",
    ]
    for s in summary.k_stats:
        marker = " ⬅ main" if s.k == summary.main_k else ""
        lines.append(f"| {s.k}{marker} | {_pct(s.pass_at_k)} | {_pct(s.pass_pow_k)} |")

    if summary.rdc:
        lines += ["", f"**Reliability Decay Curve** (mean pass^{summary.main_k} by bucket):", ""]
        lines.append("| bucket | pass^k |")
        lines.append("| --- | ---: |")
        for bucket, val in summary.rdc.items():
            lines.append(f"| {bucket} | {_pct(val)} |")
        if summary.rds is not None:
            lines += ["", f"Reliability Decay Slope (RDS): `{summary.rds:+.4f}` per bucket."]

    lines += [
        "",
        "<sub>Metrics: pass@k (Chen 2021), pass^k / RDC (arXiv:2603.29231), "
        "Bayesian posterior (arXiv:2510.04265). See passwedge docs/DEFINITIONS.md.</sub>",
    ]
    return "\n".join(lines) + "\n"
