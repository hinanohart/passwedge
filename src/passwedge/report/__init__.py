"""Reporting: aggregate Trials into a summary and render it as Markdown."""

from __future__ import annotations

from .markdown import COMMENT_MARKER, render_markdown
from .summary import KStat, ReliabilitySummary, summarize

__all__ = [
    "COMMENT_MARKER",
    "KStat",
    "ReliabilitySummary",
    "render_markdown",
    "summarize",
]
