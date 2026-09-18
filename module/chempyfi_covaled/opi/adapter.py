"""OPI (orca-pi) adapter — sole ``import opi`` site for CovaLED ORCA I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from chempyfi.modeling.molecular_model import MolecularModel

from .intent import OrcaCalculationIntent, build_intent_from_model
from .errors import OpiDependencyError


def is_opi_available() -> bool:
    try:
        import opi  # noqa: F401
    except ImportError:
        return False
    return True


def require_opi():
    if not is_opi_available():
        raise OpiDependencyError()


def intent_from_model(
    model: "MolecularModel",
    header: str,
    mdci_header: str = "",
    center_fragment_last: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
) -> OrcaCalculationIntent:
    return build_intent_from_model(
        model,
        header,
        mdci_header,
        model.fragment_connection_bonds,
        center_fragment_last=center_fragment_last,
        charge=charge,
        multiplicity=multiplicity,
    )


def to_opi_structure(model: "MolecularModel", charge: int = 0, multiplicity: int = 1):
    """Build ``opi.input.structures.structure.Structure`` from a local model."""
    require_opi()
    from opi.input.structures.structure import Structure

    mol = model.molecule
    from rdkit import Chem

    rw = Chem.RWMol(mol)
    frag_by_atom = {}
    for frag_idx, fragment in enumerate(model.fragments, start=1):
        for atom_idx in fragment.model_atom_indices:
            frag_by_atom[int(atom_idx)] = frag_idx
    for idx, atom in enumerate(rw.GetAtoms()):
        frag_id = frag_by_atom.get(idx, 1)
        atom.SetProp("_opi_frag", str(frag_id))
    structure = Structure.from_rdkitmol(rw.GetMol(), charge=charge, multiplicity=multiplicity)
    # OPI 2.x: assign fragment_id on atoms when supported
    if hasattr(structure, "atoms"):
        for i, atom in enumerate(structure.atoms):
            if i in frag_by_atom and hasattr(atom, "fragment_id"):
                atom.fragment_id = frag_by_atom[i]
    return structure


def configure_led_calculation(calc, header: str, mdci_header: str = "", covalent_bonds=None):
    """Apply method/header blocks to an OPI ``Calculator`` (caller sets structure)."""
    require_opi()
    from opi.input.core import ArbitraryStringPos

    for line in str(header or "").splitlines():
        if line.strip():
            calc.input.add_arbitrary_string(line.strip(), pos=ArbitraryStringPos.TOP)
    if mdci_header or covalent_bonds:
        lines = ["%mdci"]
        if covalent_bonds:
            bond_str = " ".join(f"{{{a} {b}}}" for a, b in covalent_bonds)
            lines.append(f"  Covalent {bond_str}")
        for line in str(mdci_header or "").splitlines():
            if line.strip():
                lines.append(f"  {line}")
        lines.append("end")
        calc.input.add_arbitrary_string("\n".join(lines), pos=ArbitraryStringPos.BEFORE_COORDS)


def model_from_fragment_write_args(
    mol,
    fragment_indices: Sequence[Sequence[int]],
    covalent_bonds: Sequence[Tuple[int, int]],
) -> "MolecularModel":
    """Minimal ``MolecularModel`` for ``write_fragment_input`` → OPI (no cap metadata)."""
    from chempyfi.modeling.molecular_model import Fragment, MolecularModel

    n = mol.GetNumAtoms()
    fragments = tuple(Fragment(tuple(int(i) for i in frag)) for frag in fragment_indices)
    model_to_source = tuple(int(i) for i in range(n))
    bonds = tuple(tuple(sorted((int(a), int(b)))) for a, b in covalent_bonds)
    return MolecularModel(
        molecule=mol,
        fragments=fragments,
        model_to_source=model_to_source,
        boundaries=(),
        fragment_connection_bonds=bonds,
    )


def write_orca_input_opi(
    model: "MolecularModel",
    out_path: Path,
    header: str,
    mdci_header: str = "",
    basename: str = "orca",
    center_fragment_last: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
):
    """Write ORCA input via OPI from a pre-built ``MolecularModel``."""
    require_opi()
    from opi.core import Calculator

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if basename == "orca" and out_path.suffix:
        basename = out_path.stem
    intent = intent_from_model(
        model, header, mdci_header, center_fragment_last=center_fragment_last, charge=charge, multiplicity=multiplicity
    )
    calc = Calculator(basename=basename, working_dir=out_path.parent, version_check=False)
    calc.json_via_input = False
    calc.structure = to_opi_structure(model, charge=charge, multiplicity=multiplicity)
    configure_led_calculation(calc, header, mdci_header, intent.mdci_covalent_bonds)
    calc.write_input()
    return intent


def intent_from_opi_structure(model: "MolecularModel", header: str, mdci_header: str, **kwargs) -> OrcaCalculationIntent:
    """Intent reconstructed after round-trip through OPI structure builder."""
    charge = kwargs.get("charge", 0)
    multiplicity = kwargs.get("multiplicity", 1)
    if not is_opi_available():
        return intent_from_model(model, header, mdci_header, **kwargs)
    _ = to_opi_structure(model, charge=charge, multiplicity=multiplicity)
    return intent_from_model(model, header, mdci_header, **kwargs)


def _legacy_tuple_from_opi_structure(structure) -> Tuple[List[int], List[List[float]]]:
    atom_to_fragment: List[int] = []
    positions: List[List[float]] = []
    for atom in structure.atoms:
        frag_id = getattr(atom, "fragment_id", None) or 1
        atom_to_fragment.append(int(frag_id) - 1)
        coords = atom.coordinates
        positions.append([float(coords.x), float(coords.y), float(coords.z)])
    return atom_to_fragment, positions


def parse_fragment_output(path: Path) -> Tuple[List[int], List[List[float]]]:
    """
    Parse fragment assignments and coordinates from ORCA output via OPI.

    Requires ORCA ``.property.json`` sidecar (OPI ``Output.parse(read_prop_json=True)``).
    For INPUT FILE echo parsing only, use ``extract_fragment_mapping_from_output(..., use_opi=False)``.
    """
    require_opi()
    path = Path(path)
    from opi.output.core import Output

    output = Output(str(path))
    output.parse(read_prop_json=True, read_gbw_json=False, do_create_property_json=False)
    structure = output.get_structure(with_fragments=True)
    if structure is None or not structure.atoms:
        raise ValueError(
            f"OPI could not extract geometry/fragments from {path}. "
            "Ensure a .property.json exists or use parse_orca_fragments for INPUT FILE echo."
        )
    return _legacy_tuple_from_opi_structure(structure)


def extract_fragment_mapping_from_output(
    path: Path, *, use_opi: bool = False
) -> Tuple[List[int], List[List[float]]]:
    """Legacy INPUT FILE parser by default; set ``use_opi=True`` when ``.property.json`` is present."""
    if use_opi and is_opi_available():
        return parse_fragment_output(path)
    from chempyfi.orcautil.led_extract import parse_orca_fragments

    return parse_orca_fragments(path)
