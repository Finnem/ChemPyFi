"""
LED extraction and CovaLED analysis from ORCA outputs via OPI (``orca-pi``).

Production path:

    ORCA ``.out`` + ``.out.property.json`` → ``chempyfi_covaled.opi_led.extract_led_data`` → CovaLED / fp-CovaLED

Requires ``pip install 'chempyfi[opi]'`` and ORCA runs with JSON property output enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

__all__ = [
    "compute_led_interaction_matrix",
    "extract_interaction_energy",
    "parse_orca_fragments",
    "determine_fragment_groups",
    "clear_subsystem_cache",
    "compute_fp_led_interactions",
    "compute_fp_led_interactions_with_mapping",
]

_COMPONENT_ATTR = {
    "TOTAL": "total",
    "REF": "reference",
    "CORR": "correlation",
    "CORRELATION": "correlation",
    "ELECTROSTAT": "electrostatics",
    "EXCHANGE": "exchange",
}

_LED_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def parse_orca_fragments(orca_out_path: Union[str, Path]) -> Tuple[List[int], List[List[float]]]:
    """Parse fragment assignments from ORCA ``INPUT FILE`` *xyz section."""
    orca_out_path = Path(orca_out_path)
    if not orca_out_path.exists():
        raise FileNotFoundError(f"ORCA output file not found: {orca_out_path}")

    atom_to_fragment: List[int] = []
    positions: List[List[float]] = []
    in_input_section = False
    in_xyz_section = False

    with open(orca_out_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "INPUT FILE" in line:
                in_input_section = True
                continue
            if in_input_section and "*xyz" in line.replace(" ", ""):
                in_xyz_section = True
                continue
            if in_xyz_section and "|" in line:
                parts = line.split("|", 1)
                if len(parts) < 2:
                    continue
                content = parts[1].strip()
                if not content:
                    continue
                if ">" in content:
                    content = content.split(">", 1)[1].strip()
                if content == "*":
                    break
                tokens = content.split()
                if len(tokens) >= 4 and "(" in tokens[0] and ")" in tokens[0]:
                    try:
                        start = tokens[0].index("(")
                        end = tokens[0].index(")")
                        frag_num = int(tokens[0][start + 1 : end])
                        atom_to_fragment.append(frag_num - 1)
                        positions.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
                    except (ValueError, IndexError):
                        continue

    if not atom_to_fragment:
        raise ValueError(f"Could not parse fragment assignments from ORCA output: {orca_out_path}")
    return atom_to_fragment, positions


def determine_fragment_groups(
    atom_to_fragment: List[int],
    ligand_atom_indices: Optional[List[int]] = None,
    assign_mixed: str = "error",
) -> Tuple[List[int], List[int]]:
    """Map atoms to ligand/receptor fragment IDs (1-based, supersystem labels)."""
    if ligand_atom_indices is None:
        ligand_atom_indices = list(range(len(atom_to_fragment) // 2))

    ligand_atom_set = set(ligand_atom_indices)
    receptor_atom_set = set(range(len(atom_to_fragment))) - ligand_atom_set
    fragment_atoms: Dict[int, List[int]] = {}
    for atom_idx, frag_idx in enumerate(atom_to_fragment):
        fragment_atoms.setdefault(frag_idx, []).append(atom_idx)

    ligand_fragments: List[int] = []
    receptor_fragments: List[int] = []
    for frag_idx, atoms_in_frag in sorted(fragment_atoms.items()):
        atoms_in_frag_set = set(atoms_in_frag)
        lig_in = atoms_in_frag_set & ligand_atom_set
        rec_in = atoms_in_frag_set & receptor_atom_set
        frag_1based = frag_idx + 1
        if lig_in and rec_in:
            if assign_mixed == "error":
                raise ValueError(f"Fragment {frag_1based} spans ligand and receptor")
            if assign_mixed == "majority":
                (ligand_fragments if len(lig_in) >= len(rec_in) else receptor_fragments).append(frag_1based)
            elif assign_mixed == "ligand":
                ligand_fragments.append(frag_1based)
            elif assign_mixed == "receptor":
                receptor_fragments.append(frag_1based)
            else:
                raise ValueError(f"Invalid assign_mixed: {assign_mixed}")
        elif lig_in:
            ligand_fragments.append(frag_1based)
        elif rec_in:
            receptor_fragments.append(frag_1based)
    return sorted(ligand_fragments), sorted(receptor_fragments)


def clear_subsystem_cache() -> None:
    _LED_CACHE.clear()


def _require_opi():
    try:
        import opi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "CovaLED from ORCA requires orca-pi. Install: pip install 'chempyfi[opi]'"
        ) from e


def _component_dataframe(led, component: str) -> pd.DataFrame:
    from chempyfi_covaled.exceptions import MissingLEDDataError

    key = component.upper()
    if key == "TOTAL":
        arr = led.total
    else:
        attr = _COMPONENT_ATTR.get(key)
        if attr is None:
            raise ValueError(f"Unknown LED component: {component}")
        arr = getattr(led.components, attr, None)
    if arr is None:
        raise MissingLEDDataError(f"LED component {component}", source=led.source_path)
    labels = list(led.fragment_ids)
    return pd.DataFrame(arr, index=labels, columns=labels)


def _load_led_matrix(out_path: Union[str, Path], component: str, *, use_cache: bool) -> pd.DataFrame:
    from chempyfi_covaled.opi_led import extract_led_data

    _require_opi()
    out_path = Path(out_path)
    cache_key = (str(out_path.resolve()), component.upper())
    if use_cache and cache_key in _LED_CACHE:
        return _LED_CACHE[cache_key].copy()
    led = extract_led_data(out_path)
    df = _component_dataframe(led, component)
    if use_cache:
        _LED_CACHE[cache_key] = df.copy()
    return df


def compute_led_interaction_matrix(
    orca_super: Union[str, Path],
    orca_ligand: Union[str, Path],
    orca_receptor: Union[str, Path],
    ligand_fpled_frags: Optional[List[int]] = None,
    receptor_fpled_frags: Optional[List[int]] = None,
    ligand_atom_indices: Optional[List[int]] = None,
    component: str = "TOTAL",
    method: str = "DLPNO-CCSD(T)",
    conversion_factor: float = 2625.5,
    use_cache: bool = True,
    force_regenerate: bool = False,
    assign_mixed_fragments: str = "error",
    verbose: bool = False,
) -> pd.DataFrame:
    """CovaLED Step 7 from OPI ``.property.json`` sidecars (``method`` retained for API compat)."""
    del method, conversion_factor, force_regenerate, verbose  # OPI JSON only
    if ligand_fpled_frags is None or receptor_fpled_frags is None:
        atom_to_fragment, _ = parse_orca_fragments(orca_super)
        ligand_fpled_frags, receptor_fpled_frags = determine_fragment_groups(
            atom_to_fragment, ligand_atom_indices, assign_mixed=assign_mixed_fragments
        )
    df_super = _load_led_matrix(orca_super, component, use_cache=use_cache)
    df_lig = _load_led_matrix(orca_ligand, component, use_cache=use_cache)
    df_rec = _load_led_matrix(orca_receptor, component, use_cache=use_cache)
    from chempyfi_covaled.analysis import compute_covaled_interaction_matrix

    return compute_covaled_interaction_matrix(
        df_super, df_lig, df_rec, ligand_fpled_frags, receptor_fpled_frags
    )


def extract_interaction_energy(led_int: pd.DataFrame) -> float:
    from chempyfi_covaled.analysis import extract_interaction_energy as _fn

    return _fn(led_int)


def _fp_dict_from_total_matrix(df: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for frag_i in df.index:
        if pd.isna(frag_i):
            continue
        for frag_j in df.columns:
            if pd.isna(frag_j) and frag_i > frag_j:
                continue
            if frag_i <= frag_j:
                energy = df.loc[frag_i, frag_j]
                if pd.notna(energy):
                    out[f"{int(frag_i)}_{int(frag_j)}"] = float(energy)
    return out


def compute_fp_led_interactions_with_mapping(
    main_filenames: List[Union[str, Path]],
    alternative_filenames: List[Union[str, Path]],
    fragment_mappings: Dict[str, Dict[int, int]],
    conversion_factor: float = 2625.5,
    method: str = "DLPNO-CCSD(T)",
    LEDAW_output_path: Optional[Union[str, Path]] = None,
    use_temp_dir: bool = True,
    use_dataframes: bool = True,
    cleanup_excel: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    """Gas-phase fp-CovaLED Step 8 via OPI + ``chempyfi_covaled`` (no LEDAW)."""
    del alternative_filenames, conversion_factor, LEDAW_output_path, use_temp_dir, cleanup_excel, verbose
    _require_opi()
    if len(main_filenames) < 3:
        raise ValueError("main_filenames must be [supersystem, ligand subsystem, receptor subsystem]")
    super_p, lig_p, rec_p = main_filenames[0], main_filenames[1], main_filenames[2]
    lig_ids = sorted(fragment_mappings.get("SUBSYS1", {}).values())
    rec_ids = sorted(fragment_mappings.get("SUBSYS2", {}).values())
    if not lig_ids or not rec_ids:
        raise ValueError("fragment_mappings must define SUBSYS1 and SUBSYS2 supersystem fragment IDs")

    from chempyfi_covaled.analysis import compute_covaled_from_system, compute_fp_covaled
    from chempyfi_covaled.fp_led import build_gas_phase_fp_led_matrices
    from chempyfi_covaled.opi_led import extract_led_data
    from chempyfi_covaled.led_data import LEDSystemData
    from chempyfi_covaled.opi_led import led_data_to_standard_matrices
    from chempyfi_covaled.transforms import inter_molecular_pairs_from_dataframe

    led_super = extract_led_data(super_p)
    led_lig = extract_led_data(lig_p)
    led_rec = extract_led_data(rec_p)
    system = LEDSystemData(
        supersystem=led_super,
        ligand=led_lig,
        receptor=led_rec,
        ligand_fragment_ids=tuple(lig_ids),
        receptor_fragment_ids=tuple(rec_ids),
    )
    system.validate_partition()
    step7 = compute_covaled_from_system(system)
    inter_pairs = inter_molecular_pairs_from_dataframe(step7, lig_ids, rec_ids)
    if use_dataframes:
        fp_mats = build_gas_phase_fp_led_matrices(led_data_to_standard_matrices(led_super), method)
        return _fp_dict_from_total_matrix(fp_mats["TOTAL"])
    pairs = compute_fp_covaled(step7, lig_ids, rec_ids, inter_pairs)
    return pairs


def compute_fp_led_interactions(
    orca_out_path: Union[str, Path],
    method: str = "DLPNO-CCSD(T)",
    conversion_factor: float = 2625.5,
    force_regenerate: bool = False,
    verbose: bool = False,
) -> Dict[str, float]:
    """Upper-triangle fp-LED ``TOTAL`` pairs for one supersystem OPI output."""
    del conversion_factor, force_regenerate, verbose
    _require_opi()
    from chempyfi_covaled.fp_led import build_gas_phase_fp_led_matrices
    from chempyfi_covaled.opi_led import extract_led_data
    from chempyfi_covaled.opi_led import led_data_to_standard_matrices

    led = extract_led_data(orca_out_path)
    fp_mats = build_gas_phase_fp_led_matrices(led_data_to_standard_matrices(led), method)
    return _fp_dict_from_total_matrix(fp_mats["TOTAL"])
