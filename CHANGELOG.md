# Changelog

All notable changes to passwedge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches `0.1.0`.

## [0.0.1a2] - 2026-05-27

Polish from a second independent verification pass. The metrics, formulas, and numeric
outputs are unchanged; these are documentation, validation, and tooling refinements.

### Changed
- `two_proportion_z_test`: the docstring now matches behavior -- a zero pooled variance
  (both samples entirely pass or entirely fail) returns `z=0, p_value=1` instead of
  raising. Removed a branch that was provably unreachable.
- `passwedge ci` validates `--prior` (against the known presets) and `--credible-level`
  (must lie in `(0, 1)`) at the argument layer, so invalid input now exits with a clean
  usage error (exit code 2) rather than a raw traceback.

### Fixed
- The GitHub Action no longer surfaces an opaque `ENOENT` when `passwedge` crashes before
  writing its report: it emits a warning and lets the dedicated fail-closed gate step
  report the failure.

### Tests
- Added an in-process `pytester` test covering the plugin's fixture-argument resolution
  path.

## [0.0.1a1] - 2026-05-27

First public pre-alpha.

### Added
- Capability metrics: `pass_at_k` (Chen et al. 2021) and `pass_pow_k`
  (Beyond pass@1, arXiv:2603.29231, Def. 2) with unbiased and plugin estimators.
- Bayesian posterior helpers (`beta_posterior`, `dirichlet_posterior`) operationalizing
  arXiv:2510.04265: posterior mean, equal-tail credible interval, `E[p^k]`, and posterior
  expected pass@k. Numerics cross-checked against `scorio` under a uniform prior.
- Reliability metrics (arXiv:2603.29231): `reliability_decay_curve` / `reliability_decay_slope`
  (RDC/RDS), `variance_amplification_factor` (VAF), `graceful_degradation_score` (GDS),
  `window_entropy_curve` and `meltdown_onset_point` (MOP, fail-closed thresholds).
- Statistics: equal-tail and HPD credible intervals; two-proportion z-test and
  credible-interval disjointness.
- A domain-agnostic `Trial`/`Episode`/`Step` data model with generic JSON / bool-list
  adapters and a thin tetrad-lens trace reader (no package dependency).
- Three distribution forms: library, `@pytest.mark.passk` pytest plugin, and a
  composite GitHub Action that upserts a reliability report on pull requests.

### Notes
- All metric definitions and provenance are in [`docs/DEFINITIONS.md`](docs/DEFINITIONS.md).
- arXiv:2510.04265 does not define a metric named "Bayes@k"; the Bayesian helpers are
  passwedge's operationalization of that paper's framework.

[0.0.1a2]: https://github.com/hinanohart/passwedge/releases/tag/v0.0.1a2
[0.0.1a1]: https://github.com/hinanohart/passwedge/releases/tag/v0.0.1a1
