"""Testing utilities for wavefront learning models.

Submodules:
- test_running: Sanity checks, profiling, and inference (verifying the model runs)
- test_results: Evaluation metrics, sample collection, and performance measurement
"""

from testing.test_results import collect_samples, test_high_res, test_model
from testing.test_running import run_inference, run_profiler, run_sanity_check

__all__ = [
    "collect_samples",
    "run_inference",
    "run_profiler",
    "run_sanity_check",
    "test_high_res",
    "test_model",
]
