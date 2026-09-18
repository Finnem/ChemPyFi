"""CovaLED workflows: pure data and OPI-backed analysis."""

from __future__ import annotations

from typing import Any, Optional

from .analysis import analyze_led_system, compute_covaled_interaction_matrix, compute_fp_covaled
from .pipeline import analyze_led_data, analyze_output
from .results import CovaLEDResult, FPCovaLEDResult, LEDMatrix


def analyze_from_led_matrices(
    df_super,
    df_ligand,
    df_receptor,
    ligand_fragment_ids,
    receptor_fragment_ids,
    inter_pair_values=None,
) -> Any:
    """Legacy pandas entry (no ORCA)."""
    led_int = compute_covaled_interaction_matrix(
        df_super, df_ligand, df_receptor, list(ligand_fragment_ids), list(receptor_fragment_ids)
    )
    chempyfi_covaled = CovaLEDResult(
        interaction_matrix=LEDMatrix(led_int),
        ligand_fragment_ids=tuple(ligand_fragment_ids),
        receptor_fragment_ids=tuple(receptor_fragment_ids),
    )
    if inter_pair_values is None:
        return chempyfi_covaled
    fp = compute_fp_covaled(
        led_int, list(ligand_fragment_ids), list(receptor_fragment_ids), inter_pair_values
    )
    from .analysis import interaction_energy_from_matrix

    intra = interaction_energy_from_matrix(led_int) - sum(inter_pair_values.values())
    return FPCovaLEDResult(pairwise=fp, redistributed_intra_total=intra, source_matrix=LEDMatrix(led_int))


def analyze(model, *args, **kwargs) -> Any:
    """
    Model-driven workflow (OPI calculation execution not implemented here).

    Use :func:`analyze_output` when ORCA ``.out`` files exist, or
    :func:`analyze_led_data` with pre-built ``LEDData``.
    """
    raise NotImplementedError(
        "chempyfi_covaled.workflow.analyze(model) requires OPI Calculator execution wiring. "
        "Use chempyfi_covaled.pipeline.analyze_output(...) or analyze_led_data(...)."
    )


def run_and_analyze(*args, **kwargs) -> Any:
    """Reserved: run ORCA via OPI then analyze (not implemented)."""
    raise NotImplementedError("run_and_analyze requires OPI execution wiring.")
