"""Hydrogen capping at fragment boundaries."""

import logging

import numpy as np
from rdkit.Chem import AllChem as Chem

from ..rdutil import keep_atoms, position

logger = logging.getLogger(__name__)


def create_molecule_cutout(mol, frags, to_replace, verbose=False):
    all_indices = set()
    for frag in frags:
        all_indices.update(frag)
    remaining_mol = Chem.RWMol(keep_atoms(mol, all_indices))
    if verbose:
        logger.info(
            f"keep_atoms: kept {len(all_indices)} / {mol.GetNumAtoms()} atoms"
        )
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(list(all_indices)))}
    added_hydrogens = []
    for atom_pair in to_replace:
        atom1 = mol.GetAtomWithIdx(atom_pair[0])
        atom2 = mol.GetAtomWithIdx(atom_pair[1])
        remaining_mol.AddAtom(Chem.Atom(1))
        new_idx = remaining_mol.GetNumAtoms() - 1
        remaining_mol.AddBond(index_map[atom1.GetIdx()], new_idx, Chem.rdchem.BondType.SINGLE)
        pos1 = position(atom1)
        pos2 = position(atom2)
        direction = pos2 - pos1
        direction /= np.linalg.norm(direction)
        new_position = pos1 + direction * 1.1
        remaining_mol.GetConformer().SetAtomPosition(new_idx, new_position)
        added_hydrogens.append((atom1.GetIdx(), new_idx))
    if verbose:
        logger.info(f"Hydrogen caps: {len(to_replace)}")
    return remaining_mol, index_map, added_hydrogens
