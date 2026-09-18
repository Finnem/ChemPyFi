"""Pure CovaLED Step 7 / fp-CovaLED Step 8 (pandas only; no ORCA/OPI/LEDAW)."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd

from .led_data import LEDData, LEDSystemData


def compute_covaled_interaction_matrix(
    df_super: pd.DataFrame,
    df_ligand: pd.DataFrame,
    df_receptor: pd.DataFrame,
    ligand_fpled_frags: List[int],
    receptor_fpled_frags: List[int],
) -> pd.DataFrame:
    """CovaLED Step 7: subtract subsystem LED from supersystem."""
    led_int = df_super.copy()
    all_frags = sorted(set(ligand_fpled_frags + receptor_fpled_frags))
    for i, frag_i in enumerate(all_frags):
        for j, frag_j in enumerate(all_frags):
            # i == j is intra-fragment EL-PREP and must be subsystem-subtracted.
            # Skipping the diagonal left supersystem self-energies (~10^6 kJ/mol)
            # in the interaction map and made E_int unphysical.
            if frag_i > frag_j:
                continue
            try:
                super_val = df_super.loc[frag_i, frag_j]
                if pd.isna(super_val):
                    continue
                in_ligand_i = frag_i in ligand_fpled_frags
                in_ligand_j = frag_j in ligand_fpled_frags
                if in_ligand_i and in_ligand_j:
                    subsys_i = ligand_fpled_frags.index(frag_i) + 1
                    subsys_j = ligand_fpled_frags.index(frag_j) + 1
                    try:
                        subsys_val = df_ligand.loc[subsys_i, subsys_j]
                        if pd.notna(subsys_val):
                            led_int.loc[frag_i, frag_j] = super_val - subsys_val
                    except KeyError:
                        pass
                elif not in_ligand_i and not in_ligand_j:
                    subsys_i = receptor_fpled_frags.index(frag_i) + 1
                    subsys_j = receptor_fpled_frags.index(frag_j) + 1
                    try:
                        subsys_val = df_receptor.loc[subsys_i, subsys_j]
                        if pd.notna(subsys_val):
                            led_int.loc[frag_i, frag_j] = super_val - subsys_val
                    except KeyError:
                        pass
            except KeyError:
                continue
    return led_int


def compute_fp_covaled(
    led_int: pd.DataFrame,
    ligand_fpled_frags: List[int],
    receptor_fpled_frags: List[int],
    fragment_pair_values: Dict[str, float],
) -> Dict[str, float]:
    """fp-CovaLED Step 8: redistribute intra-molecular LED to inter-molecular pairs."""
    intra_ligand = 0.0
    for i, fpled_i in enumerate(ligand_fpled_frags):
        for fpled_j in ligand_fpled_frags[i:]:
            try:
                val = led_int.loc[fpled_i, fpled_j]
                if pd.notna(val):
                    intra_ligand += float(val)
            except (KeyError, ValueError):
                pass
    intra_receptor = 0.0
    for i, fpled_i in enumerate(receptor_fpled_frags):
        for fpled_j in receptor_fpled_frags[i:]:
            try:
                val = led_int.loc[fpled_i, fpled_j]
                if pd.notna(val):
                    intra_receptor += float(val)
            except (KeyError, ValueError):
                pass
    intra_total = intra_ligand + intra_receptor
    total_abs_inter = sum(abs(val) for val in fragment_pair_values.values())
    if total_abs_inter > 0 and abs(intra_total) > 1e-6:
        fpled_interactions = {}
        for frag_pair, inter_val in fragment_pair_values.items():
            scale_factor = abs(inter_val) / total_abs_inter
            redistribution = intra_total * scale_factor
            fpled_interactions[frag_pair] = inter_val + redistribution
        return fpled_interactions
    return fragment_pair_values.copy()


def interaction_energy_from_matrix(led_int: pd.DataFrame) -> float:
    """Scalar CovaLED interaction energy from a Step 7 matrix.

    Matrix
        CovaLED Step 7 interaction map (supersystem minus subsystems),
        including intra-fragment diagonal terms after that subtraction.
    Pairs
        Every stored pair with fragment index ``i <= j``: ligand–receptor
        couplings **and** intra-ligand / intra-receptor residuals, including
        the diagonal. This is the extract_COVALED / LED_interaction.xlsx sum
        that yields −182.57 kJ/mol for 5L4Q, not ligand–receptor-only.
    Triangle
        LEDAW stores the lower triangle as NaN (absent cells). Lower-triangle
        values are ignored even if a caller filled them symmetrically, so a
        full symmetric matrix is not double-counted.
    Units
        kJ/mol.

    Do not use generic ``np.sum(matrix)`` or pandas ``DataFrame.sum().sum()``
    on a dense symmetric copy: that either double-counts off-diagonals or
    includes unstored NaNs as a poisoned total.
    """
    values = np.asarray(led_int, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("interaction matrix must be square")
    upper = np.triu(np.ones(values.shape, dtype=bool), k=0)
    return float(np.nansum(np.where(upper, values, np.nan)))


def extract_interaction_energy(led_int: pd.DataFrame) -> float:
    """Alias for :func:`interaction_energy_from_matrix` (legacy name)."""
    return interaction_energy_from_matrix(led_int)


def compute_covaled_from_system(system: LEDSystemData) -> pd.DataFrame:
    """Step 7 from normalized ``LEDSystemData`` (no Excel)."""
    system.validate_partition()
    return compute_covaled_interaction_matrix(
        system.supersystem.to_pandas(),
        system.ligand.to_pandas(),
        system.receptor.to_pandas(),
        list(system.ligand_fragment_ids),
        list(system.receptor_fragment_ids),
    )


def analyze_led_system(
    system: LEDSystemData,
    inter_pair_values: Union[Dict[str, float], None] = None,
):
    """Step 7 and optional Step 8 from ``LEDSystemData``."""
    from .results import CovaLEDResult, FPCovaLEDResult, LEDMatrix

    led_int = compute_covaled_from_system(system)
    chempyfi_covaled = CovaLEDResult(
        interaction_matrix=LEDMatrix(led_int),
        ligand_fragment_ids=system.ligand_fragment_ids,
        receptor_fragment_ids=system.receptor_fragment_ids,
    )
    if inter_pair_values is None:
        return chempyfi_covaled
    fp = compute_fp_covaled(
        led_int,
        list(system.ligand_fragment_ids),
        list(system.receptor_fragment_ids),
        inter_pair_values,
    )
    intra = interaction_energy_from_matrix(led_int) - sum(inter_pair_values.values())
    return FPCovaLEDResult(pairwise=fp, redistributed_intra_total=intra, source_matrix=LEDMatrix(led_int))


compute_covaled = compute_covaled_interaction_matrix
