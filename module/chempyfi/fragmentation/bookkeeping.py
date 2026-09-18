"""Fragment membership, cut bonds, and renumbering for local models."""

from rdkit.Chem import AllChem as Chem


def compute_replacement_pairs(mol, frags, kept_atoms):
    to_replace = set()
    for frag in frags:
        frag_set = set(frag)
        for idx in frag_set:
            atom = mol.GetAtomWithIdx(int(idx))
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx not in kept_atoms and neighbor.GetAtomicNum() != 1:
                    to_replace.add((idx, neighbor_idx))
    return list(to_replace)


def get_fragment_connection_bonds(mol, frags, index_map):
    atom_to_fragment = {}
    for frag_idx, frag in enumerate(frags):
        for atom_idx in frag:
            atom_to_fragment[int(atom_idx)] = frag_idx
    connection_bonds = set()
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 not in atom_to_fragment or a2 not in atom_to_fragment:
            continue
        if atom_to_fragment[a1] == atom_to_fragment[a2]:
            continue
        if a1 in index_map and a2 in index_map:
            connection_bonds.add(tuple(sorted((index_map[a1], index_map[a2]))))
    return sorted(connection_bonds)


def reorder_fragments_consecutive(
    mol, fragments, covalent_bonds, center_atom_index, return_renumber_map=False
):
    ordered_fragments = [sorted(frag) for frag in fragments]
    center_idx = None
    if center_atom_index is not None:
        for i, frag in enumerate(ordered_fragments):
            if center_atom_index in frag:
                center_idx = i
                break
    center_frag = None
    if center_idx is not None:
        center_frag = ordered_fragments.pop(center_idx)
    new_order = []
    for frag in ordered_fragments:
        new_order.extend(frag)
    total_atoms = mol.GetNumAtoms()
    if len(new_order) != total_atoms:
        center_set = set(center_frag or [])
        new_order_set = set(new_order)
        remaining = [
            idx for idx in range(total_atoms)
            if idx not in new_order_set and idx not in center_set
        ]
        if remaining:
            ordered_fragments.append(remaining)
            new_order.extend(remaining)
    if center_frag is not None:
        ordered_fragments.append(center_frag)
        new_order.extend(center_frag)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(new_order)}
    reordered_mol = Chem.RenumberAtoms(mol, new_order)
    new_fragments = []
    offset = 0
    for frag in ordered_fragments:
        size = len(frag)
        new_fragments.append(list(range(offset, offset + size)))
        offset += size
    new_covalent_bonds = sorted(
        tuple(sorted((index_map[a], index_map[b]))) for a, b in covalent_bonds
    )
    if return_renumber_map:
        return reordered_mol, new_fragments, new_covalent_bonds, index_map
    return reordered_mol, new_fragments, new_covalent_bonds
