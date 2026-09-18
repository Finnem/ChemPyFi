"""OPI output → LEDData → CovaLED (no LEDAW / Excel)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from .analysis import analyze_led_system
from .led_data import LEDData, LEDSystemData
from .opi_led import extract_led_data
from .transforms import inter_molecular_pairs


def analyze_led_data(
    supersystem: LEDData,
    ligand: LEDData,
    receptor: LEDData,
    ligand_fragment_ids,
    receptor_fragment_ids,
    inter_pair_values: Optional[Dict[str, float]] = None,
):
    system = LEDSystemData(
        supersystem=supersystem,
        ligand=ligand,
        receptor=receptor,
        ligand_fragment_ids=tuple(ligand_fragment_ids),
        receptor_fragment_ids=tuple(receptor_fragment_ids),
    )
    return analyze_led_system(system, inter_pair_values)


def analyze_output(
    orca_super: Union[str, Path],
    orca_ligand: Union[str, Path],
    orca_receptor: Union[str, Path],
    ligand_fragment_ids,
    receptor_fragment_ids,
    *,
    inter_pair_values: Optional[Dict[str, float]] = None,
    derive_inter_pairs_from_super: bool = False,
):
    """
    Parse three ORCA outputs via OPI and run CovaLED (+ optional Step 8).

    When ``derive_inter_pairs_from_super`` is True, inter-molecular pair
    energies are taken from the supersystem ``LEDData.total`` (standard LED,
    not fp-redistributed).
    """
    super_d = extract_led_data(orca_super)
    lig_d = extract_led_data(orca_ligand)
    rec_d = extract_led_data(orca_receptor)
    pairs = inter_pair_values
    if derive_inter_pairs_from_super and pairs is None:
        pairs = inter_molecular_pairs(
            super_d.total,
            super_d.fragment_ids,
            tuple(ligand_fragment_ids),
            tuple(receptor_fragment_ids),
        )
    return analyze_led_data(
        super_d,
        lig_d,
        rec_d,
        ligand_fragment_ids,
        receptor_fragment_ids,
        inter_pair_values=pairs,
    )
