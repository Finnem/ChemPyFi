"""Developer oracle: legacy LEDAW vs OPI gas-phase CovaLED equivalence."""

from .fragment_align import FragmentAlignmentError, FragmentMapping, align_square_matrix, permutation_to_align
from .harness import compare_gas_phase_led_paths, human_summary, resolve_fixture_paths, write_report

__all__ = [
    "FragmentAlignmentError",
    "FragmentMapping",
    "align_square_matrix",
    "compare_gas_phase_led_paths",
    "human_summary",
    "permutation_to_align",
    "resolve_fixture_paths",
    "write_report",
]
