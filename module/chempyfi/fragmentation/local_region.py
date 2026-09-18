"""Spatial neighborhood growth for local-model fragmentation."""

import numpy as np
from rdkit.Chem import AllChem as Chem
from scipy.spatial import cKDTree

from ..rdutil import position


def get_surrounding_indices(mol, center_indices, radius=5):
    positions = position(mol)
    tree = cKDTree(positions)
    surrounding_indices = tree.query_ball_point(positions[center_indices], radius)
    surrounding_indices = [index for sublist in surrounding_indices for index in sublist]
    return surrounding_indices


def extend_connected_indices(mol, indices, component_indices, ring_info=None, max_ring_size=12):
    """Extend indices within a connected component to the next C-C bond cut."""
    to_check = list(indices)
    seen = set(indices)
    keep = set(indices)
    component_set = set(component_indices)
    small_ring_atoms = set()
    if ring_info is not None:
        try:
            for idx in component_set:
                for size in range(3, max_ring_size + 1):
                    if ring_info.IsAtomInRingOfSize(int(idx), int(size)):
                        small_ring_atoms.add(idx)
                        break
        except Exception:
            small_ring_atoms = set()
    while to_check:
        index = to_check.pop()
        index_atom = mol.GetAtomWithIdx(index)
        for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index not in component_set:
                continue
            neighbor_symbol = neighbor.GetSymbol()
            index_symbol = index_atom.GetSymbol()
            single_bond = mol.GetBondBetweenAtoms(index, neighbor_index).GetBondType() == Chem.rdchem.BondType.SINGLE
            both_carbon = (index_symbol == "C") and (neighbor_symbol == "C")
            if not (neighbor_index in seen) and (
                not (
                    single_bond
                    and both_carbon
                    and not (neighbor_index in small_ring_atoms)
                    and not (index in small_ring_atoms)
                )
            ):
                to_check.append(neighbor_index)
            seen.add(neighbor_index)
            if both_carbon:
                if len(neighbor.GetNeighbors()) == 4:
                    keep.add(neighbor_index)
            else:
                keep.add(neighbor_index)
    new_seen = set()
    for index in keep:
        for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor.GetAtomicNum() == 1:
                new_seen.add(neighbor_index)

    to_replace = set()
    for index in keep:
        for neighbor in mol.GetAtomWithIdx(index).GetNeighbors():
            neighbor_index = neighbor.GetIdx()
            if neighbor_index not in keep:
                if neighbor.GetAtomicNum() != 1:
                    to_replace.add((index, neighbor_index))
    keep.update(new_seen)
    return list(keep), list(to_replace)
