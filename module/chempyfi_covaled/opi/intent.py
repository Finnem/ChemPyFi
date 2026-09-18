"""Semantic ORCA calculation description (OPI-agnostic)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple, Union

from rdkit import Chem

from chempyfi.modeling.molecular_model import MolecularModel
from chempyfi.rdutil.geometry import position


@dataclass(frozen=True)
class OrcaAtomSpec:
    model_atom_index: int
    element: str
    fragment_id: int  # 1-based ORCA fragment index
    coordinates_angstrom: Tuple[float, float, float]


@dataclass(frozen=True)
class OrcaCalculationIntent:
    """Portable description of what ``write_fragment_input`` expresses."""

    charge: int
    multiplicity: int
    header_text: str
    mdci_header_text: str
    mdci_covalent_bonds: Tuple[Tuple[int, int], ...]  # model atom indices
    atoms: Tuple[OrcaAtomSpec, ...]
    center_fragment_last: bool = False

    @property
    def fragment_ids(self) -> Tuple[int, ...]:
        return tuple(sorted({a.fragment_id for a in self.atoms}))


def build_intent_from_model(
    model: MolecularModel,
    header: str,
    mdci_header: str,
    covalent_bonds: Sequence[Tuple[int, int]],
    center_fragment_last: bool = False,
    charge: int = 0,
    multiplicity: int = 1,
) -> OrcaCalculationIntent:
    mol = model.molecule
    positions = position(mol)
    fragment_map = {}
    for frag_idx, fragment in enumerate(model.fragments, start=1):
        for atom_idx in fragment.model_atom_indices:
            fragment_map[int(atom_idx)] = frag_idx
    atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = positions[idx]
        atoms.append(
            OrcaAtomSpec(
                model_atom_index=idx,
                element=atom.GetSymbol(),
                fragment_id=fragment_map.get(idx, 1),
                coordinates_angstrom=(float(pos[0]), float(pos[1]), float(pos[2])),
            )
        )
    ordered_bonds = tuple(
        _order_covalent_bonds(
            covalent_bonds,
            [list(f.model_atom_indices) for f in model.fragments],
            center_fragment_last=center_fragment_last,
        )
    )
    return OrcaCalculationIntent(
        charge=charge,
        multiplicity=multiplicity,
        header_text=str(header or "").strip(),
        mdci_header_text=str(mdci_header or "").strip(),
        mdci_covalent_bonds=ordered_bonds,
        atoms=tuple(atoms),
        center_fragment_last=center_fragment_last,
    )


def build_intent_from_legacy_args(
    mol: Chem.Mol,
    fragment_indices: Sequence[Sequence[int]],
    covalent_bonds: Sequence[Tuple[int, int]],
    header: str,
    mdci_header: str,
    center_fragment_last: bool = False,
) -> OrcaCalculationIntent:
    fragment_map = {}
    for frag_idx, frag in enumerate(fragment_indices, start=1):
        for atom_idx in frag:
            fragment_map[int(atom_idx)] = frag_idx
    positions = position(mol)
    atoms = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = positions[idx]
        atoms.append(
            OrcaAtomSpec(
                model_atom_index=idx,
                element=atom.GetSymbol(),
                fragment_id=fragment_map.get(idx, 1),
                coordinates_angstrom=(float(pos[0]), float(pos[1]), float(pos[2])),
            )
        )
    ordered_bonds = tuple(
        _order_covalent_bonds(covalent_bonds, fragment_indices, center_fragment_last)
    )
    return OrcaCalculationIntent(
        charge=0,
        multiplicity=1,
        header_text=str(header or "").strip(),
        mdci_header_text=str(mdci_header or "").strip(),
        mdci_covalent_bonds=ordered_bonds,
        atoms=tuple(atoms),
        center_fragment_last=center_fragment_last,
    )


def build_intent_from_inp_file(path: Union[str, Path]) -> OrcaCalculationIntent:
    """
    Parse a handwritten-style ``orca.inp`` (``* xyz``, ``Element(n)`` labels, ``%mdci`` block).
    Used by OPI equivalence harnesses; does not import OPI.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    header_lines: List[str] = []
    mdci_lines: List[str] = []
    covalent_bonds: List[Tuple[int, int]] = []
    in_mdci = False
    in_xyz = False
    charge, mult = 0, 1
    atoms: List[OrcaAtomSpec] = []
    atom_idx = 0
    frag_paren = re.compile(r"^([A-Za-z]+)\((\d+)\)$")

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if in_mdci and line.lower() == "end":
            in_mdci = False
            continue
        if line.startswith("%"):
            if line.lower().startswith("%mdci"):
                in_mdci = True
                continue
            if in_mdci:
                if line.lower().startswith("covalent"):
                    for pair in re.findall(r"\{(\d+)\s+(\d+)\}", line):
                        covalent_bonds.append(tuple(sorted((int(pair[0]), int(pair[1])))))
                else:
                    mdci_lines.append(line.strip())
            continue
        if line.startswith("*"):
            if "xyz" in line.lower():
                in_xyz = True
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        charge = int(parts[1])
                        mult = int(parts[2])
                    except ValueError:
                        pass
                continue
            if in_xyz and line == "*":
                break
            continue
        if in_mdci:
            if line.lower().startswith("covalent"):
                for pair in re.findall(r"\{(\d+)\s+(\d+)\}", line):
                    covalent_bonds.append(tuple(sorted((int(pair[0]), int(pair[1])))))
            else:
                mdci_lines.append(line)
            continue
        if in_xyz:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            m = frag_paren.match(tokens[0])
            if not m:
                continue
            element, frag_id = m.group(1), int(m.group(2))
            x, y, z = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
            atoms.append(
                OrcaAtomSpec(
                    model_atom_index=atom_idx,
                    element=element,
                    fragment_id=frag_id,
                    coordinates_angstrom=(x, y, z),
                )
            )
            atom_idx += 1
            continue
        if not atoms and not in_mdci:
            header_lines.append(line)

    return OrcaCalculationIntent(
        charge=charge,
        multiplicity=mult,
        header_text="\n".join(header_lines).strip(),
        mdci_header_text="\n".join(mdci_lines).strip(),
        mdci_covalent_bonds=tuple(sorted(covalent_bonds)),
        atoms=tuple(atoms),
        center_fragment_last=False,
    )


def intents_semantically_equal(a: OrcaCalculationIntent, b: OrcaCalculationIntent, tol: float = 1e-5) -> bool:
    if a.charge != b.charge or a.multiplicity != b.multiplicity:
        return False
    if a.fragment_ids != b.fragment_ids:
        return False
    if set(a.mdci_covalent_bonds) != set(b.mdci_covalent_bonds):
        return False
    if a.header_text.strip() != b.header_text.strip():
        return False
    mdci_a = {ln.strip() for ln in a.mdci_header_text.splitlines() if ln.strip()}
    mdci_b = {ln.strip() for ln in b.mdci_header_text.splitlines() if ln.strip()}
    if mdci_a != mdci_b:
        return False
    if len(a.atoms) != len(b.atoms):
        return False
    by_idx_a = {x.model_atom_index: x for x in a.atoms}
    by_idx_b = {x.model_atom_index: x for x in b.atoms}
    if set(by_idx_a) != set(by_idx_b):
        return False
    for idx in by_idx_a:
        aa, bb = by_idx_a[idx], by_idx_b[idx]
        if aa.element != bb.element or aa.fragment_id != bb.fragment_id:
            return False
        for u, v in zip(aa.coordinates_angstrom, bb.coordinates_angstrom):
            if abs(u - v) > tol:
                return False
    return True


def _order_covalent_bonds(covalent_bonds, fragments, center_fragment_last):
    if not covalent_bonds:
        return []
    if not center_fragment_last or not fragments:
        return sorted(tuple(sorted(b)) for b in covalent_bonds)
    center_set = set(fragments[-1])
    first = []
    rest = []
    for a, b in covalent_bonds:
        if (a in center_set) ^ (b in center_set):
            first.append(tuple(sorted((a, b))))
        else:
            rest.append(tuple(sorted((a, b))))
    return sorted(first) + sorted(rest)
