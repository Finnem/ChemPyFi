"""ORCA CovaLED input generation (OPI only)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple

from chempyfi.modeling.local_environment import extract_local_environment
from chempyfi.rdutil import keep_atoms, read_molecules

from .adapter import model_from_fragment_write_args, write_orca_input_opi
from .errors import OpiDependencyError


def write_fragment_input(
    mol,
    fragment_indices,
    covalent_bonds,
    header,
    mdci_header,
    out_name,
    center_fragment_last=False,
):
    """Write ``orca.inp`` via orca-pi (no handwritten fallback)."""
    try:
        model = model_from_fragment_write_args(mol, fragment_indices, covalent_bonds)
        write_orca_input_opi(
            model,
            Path(out_name),
            header,
            mdci_header or "",
            center_fragment_last=center_fragment_last,
        )
    except ImportError as e:
        raise OpiDependencyError() from e


def write_orca_input(
    mol,
    center_atom_index,
    header,
    out_folder,
    mdci_header=None,
    additional_indices=None,
):
    """Build super/sub1/sub2 CovaLED job inputs under ``out_folder`` (OPI only)."""
    if mdci_header is None:
        mdci_header = ""
    if additional_indices is None:
        additional_indices = []
    if header is None:
        header = "ORCA_HEADER_PLACEHOLDER"
    if isinstance(mol, str):
        mol = next(read_molecules(mol))
    model = extract_local_environment(
        mol,
        center_atom_index,
        verbose=False,
        filter_occluded_fragments=False,
        additional_indices=additional_indices,
    )
    cutout_mol = model.molecule
    split_fragments = [list(f.model_atom_indices) for f in model.fragments]
    split_covalent_bonds = list(model.fragment_connection_bonds)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(os.path.join(out_folder, "super"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "sub1"), exist_ok=True)
    os.makedirs(os.path.join(out_folder, "sub2"), exist_ok=True)

    super_path = os.path.join(out_folder, "super", "orca.inp")
    sub1_path = os.path.join(out_folder, "sub1", "orca.inp")
    sub2_path = os.path.join(out_folder, "sub2", "orca.inp")

    write_fragment_input(
        cutout_mol,
        split_fragments,
        split_covalent_bonds,
        header,
        mdci_header,
        super_path,
        center_fragment_last=True,
    )

    if not split_fragments:
        return

    center_frag_index = len(split_fragments) - 1
    sub1_mol, sub1_frags, sub1_bonds = _subset_fragments(
        cutout_mol,
        split_fragments,
        split_covalent_bonds,
        [i for i in range(len(split_fragments)) if i != center_frag_index],
    )
    write_fragment_input(sub1_mol, sub1_frags, sub1_bonds, header, mdci_header, sub1_path, center_fragment_last=False)

    sub2_mol, sub2_frags, sub2_bonds = _subset_fragments(
        cutout_mol,
        split_fragments,
        split_covalent_bonds,
        [center_frag_index],
    )
    write_fragment_input(sub2_mol, sub2_frags, sub2_bonds, header, mdci_header, sub2_path, center_fragment_last=False)


def _subset_fragments(mol, fragments, covalent_bonds, fragment_indices):
    from rdkit.Chem import AllChem as Chem

    if not fragment_indices:
        return Chem.Mol(), [], []
    selected_fragments = [fragments[i] for i in fragment_indices]
    keep_atoms_set = set()
    for frag in selected_fragments:
        keep_atoms_set.update(frag)
    keep_atoms_sorted = sorted(keep_atoms_set)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_atoms_sorted)}
    new_mol = keep_atoms(mol, keep_atoms_sorted)
    new_fragments = []
    for frag in selected_fragments:
        new_fragments.append([index_map[idx] for idx in frag if idx in index_map])
    new_bonds = []
    for a, b in covalent_bonds:
        if a in index_map and b in index_map:
            new_bonds.append(tuple(sorted((index_map[a], index_map[b]))))
    return new_mol, new_fragments, sorted(new_bonds)
