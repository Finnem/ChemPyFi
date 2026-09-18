"""CovaLED / fp-CovaLED analysis (depends on chempyfi for modeling; optional OPI for workflows)."""

from .analysis import (
    compute_covaled,
    compute_covaled_interaction_matrix,
    compute_fp_covaled,
    extract_interaction_energy,
    interaction_energy_from_matrix,
)
from .led_data import LEDData, LEDSystemData
from .pipeline import analyze_led_data, analyze_output
from .workflow import analyze, analyze_from_led_matrices, run_and_analyze

compute_fp_covaled_interactions = compute_fp_covaled
from .results import CovaLEDResult, FPCovaLEDResult, LEDMatrix

__all__ = [
    "LEDData",
    "LEDSystemData",
    "analyze",
    "analyze_from_led_matrices",
    "analyze_led_data",
    "analyze_output",
    "run_and_analyze",
    "compute_covaled",
    "compute_covaled_interaction_matrix",
    "compute_fp_covaled",
    "extract_interaction_energy",
    "interaction_energy_from_matrix",
    "CovaLEDResult",
    "FPCovaLEDResult",
    "LEDMatrix",
]
