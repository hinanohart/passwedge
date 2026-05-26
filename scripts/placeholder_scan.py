#!/usr/bin/env python3
"""Fail closed if the shipped metric code contains placeholders.

Guards against the failure mode where a metric silently returns random or stubbed values
(see the project's ship-and-yank lesson). Scans ``src/passwedge`` for:

* randomness in production code (metrics must be deterministic given their inputs);
* placeholder markers (TODO / FIXME / XXX / PLACEHOLDER / "stub"/"dummy"/"fake");
* ``NotImplementedError`` raised in an already-shipped public surface.

Exits non-zero (with a report) if any are found. Run from the repo root.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "passwedge"

RANDOM_RE = re.compile(r"\b(np\.random|numpy\.random|default_rng|random\.\w+|randint|rand\()")
MARKER_RE = re.compile(r"(?i)\b(TODO|FIXME|XXX|PLACEHOLDER|DUMMY|FAKE_VALUE)\b|\bstub\b")


def scan() -> list[str]:
    problems: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(SRC.parent.parent)
        for lineno, line in enumerate(text.splitlines(), start=1):
            # ignore the import line of the stdlib random module only if unused elsewhere
            if RANDOM_RE.search(line):
                problems.append(f"{rel}:{lineno}: randomness in production code -> {line.strip()}")
            if MARKER_RE.search(line):
                problems.append(f"{rel}:{lineno}: placeholder marker -> {line.strip()}")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:  # pragma: no cover
            problems.append(f"{rel}: syntax error {exc}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc_node = node.exc
                name = None
                if isinstance(exc_node, ast.Call) and isinstance(exc_node.func, ast.Name):
                    name = exc_node.func.id
                elif isinstance(exc_node, ast.Name):
                    name = exc_node.id
                if name == "NotImplementedError":
                    problems.append(f"{rel}:{node.lineno}: NotImplementedError in shipped code")
    return problems


def main() -> int:
    problems = scan()
    if problems:
        print("PLACEHOLDER SCAN FAILED:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("placeholder scan: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
