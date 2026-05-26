# Releasing passwedge

passwedge publishes to PyPI from a local clean build with `twine` (the same flow used
across the org). OIDC Trusted Publishing is intentionally not used for `0.0.1a1` because
its first-time setup requires a manual PyPI web step; the local-twine flow keeps releases
fully scriptable.

## Steps

1. Bump `version` in `pyproject.toml` and `__version__` in `src/passwedge/__init__.py`,
   and add a `CHANGELOG.md` entry.
2. Run the full gate:
   ```bash
   ruff check . && ruff format --check . && mypy
   python scripts/placeholder_scan.py
   coverage run --source=passwedge -m pytest -q && coverage report --fail-under=85
   ```
3. Build and verify:
   ```bash
   rm -rf dist build
   python -m build
   twine check --strict dist/*
   ```
4. Verify a clean-venv install of the built wheel before uploading.
5. Upload:
   ```bash
   twine upload dist/*
   ```
6. Tag and create the GitHub release:
   ```bash
   git tag v<version> && git push origin v<version>
   gh release create v<version> --title v<version> --notes-file <notes>
   ```
