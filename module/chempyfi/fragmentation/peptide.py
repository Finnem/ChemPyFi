"""Peptide-aware splitting for local-model fragmentation (not kinematic fragmentation)."""

import logging

from rdkit.Chem import AllChem as Chem

from ..rdutil.select import match_smarts

logger = logging.getLogger(__name__)


def _find_bond_pairs_from_smarts(mol, smarts, map_num_a=1, map_num_b=2):
    matches = match_smarts(mol, smarts)
    if not matches:
        return []
    bond_pairs = set()
    for match in matches:
        atom_a = match[map_num_a - 1]
        atom_b = match[map_num_b - 1]
        bond_pairs.add(tuple(sorted((atom_a, atom_b))))
    return sorted(bond_pairs)


def _split_fragment_on_bond(fragment_set, bond_pair, fragment_mask, neighbor_lists):
    atom_a, atom_b = bond_pair
    if atom_a not in fragment_set or atom_b not in fragment_set:
        return None
    visited = set()
    stack = [atom_a]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for neighbor in neighbor_lists.get(current, []):
            if neighbor not in fragment_mask:
                continue
            if (current == atom_a and neighbor == atom_b) or (current == atom_b and neighbor == atom_a):
                continue
            if neighbor not in visited:
                stack.append(neighbor)
    if atom_b in visited:
        return None
    other = fragment_set - visited
    if not other:
        return None
    return visited, other


def _iterative_split_by_bonds(fragment_set, bond_pairs_by_atom, min_size, neighbor_lists):
    pending = [fragment_set]
    final_fragments = []
    while pending:
        current = pending.pop()
        fragment_mask = set(current)
        split_done = False
        candidate_pairs = set()
        for atom in current:
            for pair in bond_pairs_by_atom.get(atom, []):
                candidate_pairs.add(pair)
        for bond_pair in candidate_pairs:
            if bond_pair[0] not in current or bond_pair[1] not in current:
                continue
            split = _split_fragment_on_bond(current, bond_pair, fragment_mask, neighbor_lists)
            if split is None:
                continue
            left, right = split
            if len(left) > min_size and len(right) > min_size:
                pending.append(left)
                pending.append(right)
                split_done = True
                break
        if not split_done:
            final_fragments.append(current)
    return final_fragments


def _build_bond_pair_map(bond_pairs):
    bond_pairs_by_atom = {}
    for pair in bond_pairs:
        atom_a, atom_b = sorted(pair)
        bond_pairs_by_atom.setdefault(atom_a, []).append((atom_a, atom_b))
        bond_pairs_by_atom.setdefault(atom_b, []).append((atom_a, atom_b))
    return bond_pairs_by_atom


def _split_into_connected_components(mol, fragment_set):
    components = []
    remaining = set(fragment_set)
    while remaining:
        start = next(iter(remaining))
        stack = [start]
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in mol.GetAtomWithIdx(int(current)).GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in remaining and neighbor_idx not in visited:
                    stack.append(neighbor_idx)
        components.append(visited)
        remaining -= visited
    return components


def split_fragments_by_peptide_bonds(all_used_components, mol, verbose=False):
    """
    Further splits fragments by sidechain and backbone C-C bonds.

    Sidechain C-C bond: C-N-C(=O)-C (split at C(=O)-C).
    Backbone C-C bond:  C(=O)(N)-C(N) (split at C(=O)-C).
    """
    sidechain_smarts = "[C:1]([N]-[C](=O)-[C])-[C:2]"
    backbone_smarts = "[C:1](=O)([N])-[C:2]([N])"
    sidechain_bonds = _find_bond_pairs_from_smarts(mol, sidechain_smarts, 1, 2)
    backbone_bonds = _find_bond_pairs_from_smarts(mol, backbone_smarts, 1, 2)
    if verbose:
        logger.info(
            f"SMARTS matches: sidechain={len(sidechain_bonds)}, backbone={len(backbone_bonds)}"
        )
    sidechain_bonds_by_atom = _build_bond_pair_map(sidechain_bonds)
    backbone_bonds_by_atom = _build_bond_pair_map(backbone_bonds)
    all_used_atoms = set()
    for frag in all_used_components:
        all_used_atoms.update(frag)

    sidechain_bonds = [pair for pair in sidechain_bonds if (pair[0] in all_used_atoms and pair[1] in all_used_atoms)]
    backbone_bonds = [pair for pair in backbone_bonds if (pair[0] in all_used_atoms and pair[1] in all_used_atoms)]
    if verbose:
        logger.info(
            f"Filtered bonds: sidechain={len(sidechain_bonds)}, backbone={len(backbone_bonds)}"
        )
    neighbor_lists = {}
    for idx in all_used_atoms:
        neighbor_lists[idx] = [n.GetIdx() for n in mol.GetAtomWithIdx(int(idx)).GetNeighbors()]

    final_fragments = []
    for frag in all_used_components:
        fragment_set = set(frag)
        sidechain_split = _iterative_split_by_bonds(
            fragment_set, sidechain_bonds_by_atom, min_size=3, neighbor_lists=neighbor_lists
        )
        for subfrag in sidechain_split:
            backbone_split = _iterative_split_by_bonds(
                subfrag, backbone_bonds_by_atom, min_size=4, neighbor_lists=neighbor_lists
            )
            for backbone_frag in backbone_split:
                final_fragments.extend(
                    _split_into_connected_components(mol, backbone_frag)
                )
    return [sorted(list(fragment)) for fragment in final_fragments]
