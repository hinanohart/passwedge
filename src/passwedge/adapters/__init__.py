"""Input adapters: raw outcomes / JSON records / trace exports -> Trial objects."""

from __future__ import annotations

from .generic import from_outcomes, load_trials, read_trials_file, trial_from_record
from .tetrad_lens import DEFAULT_SUCCESS_VALUES, trials_from_trace_records

__all__ = [
    "DEFAULT_SUCCESS_VALUES",
    "from_outcomes",
    "load_trials",
    "read_trials_file",
    "trial_from_record",
    "trials_from_trace_records",
]
