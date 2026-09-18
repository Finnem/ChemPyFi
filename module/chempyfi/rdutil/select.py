from rdkit.Chem import GetPeriodicTable
from rdkit.Chem.rdchem import Atom
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdmolops
from .rw import proximity_bond as rw_proximity_bond
import numpy as np
import re

# Acyclic (non-ring) single bonds between two atoms that are neither terminal nor involved in a triple bond, excluding amide bonds
RotatableBondSmarts = Chem.MolFromSmarts('[!$(*#*)&!D1;!$(C=O)]-!@[!$(*#*)&!D1;!$(N-C=O)]')

def match_smarts(mol, smarts):
    pattern = Chem.MolFromSmarts(smarts)
    ind_map = {}
    for atom in pattern.GetAtoms():
        ind_map[atom.GetAtomMapNum()] = atom.GetIdx()
    matches = mol.GetSubstructMatches(pattern)

    rordered_matches = []
    for match in matches:
        ordered_map_nums = sorted([num for num in ind_map.keys() if num > 0])
        rordered_matches.append([match[ind_map[map_num]] for map_num in ordered_map_nums])
    return rordered_matches

def get_amide_bonds(mol):
    submatches = mol.GetSubstructMatches(Chem.MolFromSmarts('[CX3](=O)[NX3H1]'))
    results = []
    for submatch in submatches:
        results.append([
            [idx for idx in submatch if mol.GetAtomWithIdx(idx).GetSymbol() == "N"][0],
            [idx for idx in submatch if mol.GetAtomWithIdx(idx).GetSymbol() == "C"][0],
            ])
    return results

# Above Expression does not filter certain non rotatable bonds.
NonRotatableBondDeterminations = [
    # amide bonds
    get_amide_bonds,
]

def bond_independent_MCS_matches(mol1, mol2):
    mol1_copy = Chem.Mol(mol1)
    mol2_copy = Chem.Mol(mol2)
    for bond in mol1_copy.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    for bond in mol2_copy.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    MCS = Chem.GetMCS([mol1_copy, mol2_copy])
    mol1_match = mol1.GetSubstructMatch(MCS)
    mol2_match = mol2.GetSubstructMatch(MCS)
    mol1_to_mol2_idx = {mol1_match[i]: mol2_match[i] for i in range(len(mol1_match))}
    return mol1_to_mol2_idx

def bond_independent_substructure_matches(mol, query):
    mol_copy = Chem.Mol(mol)
    query_copy = Chem.Mol(query)
    for bond in mol_copy.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    for bond in query_copy.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    return mol_copy.GetSubstructMatches(query_copy)

def select_atom(possible_atoms, neighborhood = None, element = None, return_first = True):
    """
    Returns a atom (if return first) or all atoms that have the given amount of neighbors.

    Args:
        possible_atoms (RDKit.Mol or iterable of atoms): List of Atoms or Molecule to get the atoms from.
        neighborhood (dict): Dictionary mapping atom symbols to their number of occurences that is beeing sought.
        return_first (bool): Defaults to True. If True, return the first atom found. If False, return all atoms found.

    Returns:
        list(RDKit.Atom) or RDKit.Atom: List of atoms that have the given amount of neighbors.

    """
    atoms = []
    if neighborhood is None:
        neighborhood = {}
    if hasattr(possible_atoms, "GetAtoms"):
        possible_atoms = possible_atoms.GetAtoms()
    for atom in possible_atoms:
        invalid = False
        if not element is None:
            if atom.GetSymbol() != element:
                continue
        for symbol, count in neighborhood.items():
            for n in atom.GetNeighbors():
                if n.GetSymbol() == symbol:
                    count -= 1
            if count != 0:
                invalid = True
                break
        if not invalid:
            if return_first:
                return atom
            atoms.append(atom) 
    return atoms

def get_connected_atoms(atom, as_indices=False, split_on = None):
    """Return a list of atoms part of the connected component of the given atom.

    Args:
        atom (RDKit.Atom): Atom to get the connected atoms from.
        as_indices (bool): Defaults to False. If True, return the indices of the atoms instead of the atoms themselves.
        split_on list((start_idx, end_idx)): Defaults to None. If provided, the connected component will be split at the given atom indices.

    Returns:
        list(RDKit.Atom): List of atoms part of the connected component of the given atom.

    """
    if split_on is None:
        split_on = []
    molecule = atom.GetOwningMol()
    # we keep a list and a set to keep track of the order of when we saw the atoms but have fast lookup
    connected_atom_ids = [atom.GetIdx()]
    seen = set(connected_atom_ids) 
    to_check = list(connected_atom_ids)
    while len(to_check) > 0:
        atom_idx = to_check.pop(0)
        for neighbor in molecule.GetAtomWithIdx(atom_idx).GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if (neighbor_idx not in seen) and (((atom_idx, neighbor_idx) not in split_on) and ((neighbor_idx, atom_idx) not in split_on)):
                seen.add(neighbor_idx)
                to_check.append(neighbor_idx)
                connected_atom_ids.append(neighbor_idx)
    if as_indices:
        return connected_atom_ids
    else:
        return [molecule.GetAtomWithIdx(atom_idx) for atom_idx in connected_atom_ids]

def get_connected_component_indices(mol, split_on = None):
    
    considered_indices = set(range(mol.GetNumAtoms()))
    connected_components = []
    while considered_indices:
        considered_index = considered_indices.pop()
        component = get_connected_atoms(mol.GetAtomWithIdx(considered_index), as_indices=True, split_on=split_on)
        connected_components.append(component)
        considered_indices.difference_update(component)
    return connected_components

def match_rest_groups(mol, query, proximity_bond = True):
    """
    Matches the rest groups in the given Smiles string. Each rest group should have a group RX where X is a number.

    """
    pattern = r'\[(R\d+)\]'
    def replacer(match):
        r_label = match.group(1)  # e.g., R1
        number = ''.join(filter(str.isdigit, r_label))  # Extract '1' from 'R1'
        return f'[{r_label}:{number}]'
    
    query = re.sub(pattern, replacer, query)
    query_mol = Chem.MolFromSmarts(query)
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetIntProp("__orig_idx", atom.GetIdx())
    if proximity_bond:
        mol = rw_proximity_bond(mol, as_property_mol=True)
    # in order to trace back atom indices, we attach atom properties
    
    groups = rdRGroupDecomposition.RGroupDecompose([query_mol], [mol], asRows = False)

    result ={key : [a.GetIntProp("__orig_idx") for a in group[0].GetAtoms() if a.HasProp("__orig_idx")] for key, group in groups[0].items()}
    return result

def get_small_molecules(structure, min_atoms = 6, max_atoms = 200):
    """
        By default returns all small molecules with 6 to 200 atoms.
    """

    relevant_atom_ids = set()
    for atom in structure.GetAtoms():
        if atom.GetAtomicNum() != 1:
            relevant_atom_ids.add(atom.GetIdx())

    small_molecule_indices = []
    while len(relevant_atom_ids) > 0:
        atom_idx = relevant_atom_ids.pop()
        connected_atom_indices = get_connected_atoms(structure.GetAtomWithIdx(atom_idx), as_indices=True)
        if (len(connected_atom_indices) >= min_atoms) and (len(connected_atom_indices) <= max_atoms):
            small_molecule_indices.append(connected_atom_indices)
        relevant_atom_ids.difference_update(connected_atom_indices)
    
    return [[structure.GetAtomWithIdx(a_idx) for a_idx in (atom_indices)] for atom_indices in small_molecule_indices]



def vdw_radius(atom):
    """Return the van der Waals radius of the given atom.

    Args:
        atom (RDKit.Atom): Atom to get the van der Waals radius from. Can be a list of atoms.

    Returns:
        float: van der Waals radius of the given atom.

    """
    try:
        atom = list(atom)
    except TypeError:
        atom = [atom]

    atomic_number = [a.GetAtomicNum() if type(a) == Atom else GetPeriodicTable().GetAtomicNumber(a) for a in atom]

    return [GetPeriodicTable().GetRvdw(a) for a in atom]


def get_kinematic_chain(mol):
    """
    Returns a list of lists of atom indices that are connected to each other. Each entry in the list (i.e. each list of atom indices) is a rigid body. 
    This assumes that rotatable bonds are annotated correctly.
    This is done by traversing the molecule starting from the given atom indices.
    This function does not work for non-rigid cycles. I.e. no internal degrees of freedom of a ring are considered.

    Args:
        mol (RDKit.Mol): Molecule to get the kinematic chain from.

    Returns:
        list(list(int)): List of lists of atoms that are connected to each other.
    """

    Chem.GetSSSR(mol)
    rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
    # symmetrize the rotatable bonds
    rotatable_bonds = set((a, b) for a, b in rotatable_bonds) | set((b, a) for a, b in rotatable_bonds)
    non_rotatable_bonds = []
    for f in NonRotatableBondDeterminations:
        non_rotatable_bonds += f(mol)
    non_rotatable_bonds = set((a, b) for a, b in non_rotatable_bonds) | set((b, a) for a, b in non_rotatable_bonds)
    rotatable_bonds.difference_update(non_rotatable_bonds)
    remaining_indices = set(range(mol.GetNumAtoms()))
    rigid_components = []
    # determine the next bonds to check as those bonds which only contain a starting index once
    while len(remaining_indices) > 0:
        atom_idx = remaining_indices.pop()
        # run a breadth first search to find all connected atoms
        to_check = [atom_idx]
        connected_indices = set(to_check)
        while len(to_check) > 0:
            atom_idx = to_check.pop(0)
            # either we are part of a ring
            within_ring = False
            for ring in mol.GetRingInfo().AtomRings():
                ring = set(ring)
                if atom_idx in ring:
                    to_check = list((ring.difference(connected_indices))) + to_check
                    connected_indices.update(ring)
                    within_ring = True

            # or we are part of a non-rotatable bond
            for bond in mol.GetAtomWithIdx(atom_idx).GetBonds():
                neighbor_idx = bond.GetOtherAtomIdx(atom_idx)
                if ((atom_idx, neighbor_idx) in rotatable_bonds) and (len(mol.GetAtomWithIdx(neighbor_idx).GetNeighbors()) > 1) and (len(mol.GetAtomWithIdx(atom_idx).GetNeighbors()) > 1):
                    continue
                if neighbor_idx not in connected_indices:
                    to_check.append(neighbor_idx)
                    connected_indices.add(neighbor_idx)
        rigid_components.append(tuple(sorted(list(connected_indices))))        
        remaining_indices.difference_update(connected_indices)
    return rigid_components

def get_full_fragmentation(mol, include_endpoints = False):
    """ 
    Fragments a molecule both by scaffold and rigid substructures.

    Args:
        mol (RDKit.Mol): Molecule to fragment.
        
    Returns:
        list: List of lists, where each list contains atom indices of a scaffold or rigid substructure.
        numpy.ndarray: Adjacency matrix of the fragmentation graph.
        dict: Dictionary mapping bond indices to the atom indices of the connected atoms.
    """
    # first we get the kinematic graph
    from . import keep_atoms
    rigid_components, adjacency_matrix, bond_idx_to_atom_bond_idx = get_kinematic_graph(mol)

    # attach atom indices as properties to the atoms
    for a in mol.GetAtoms():
        a.SetIntProp("__orig_idx", a.GetIdx())

    # next we determine scaffolds as part of rigid substructures
    new_fragments = []
    new_atom_bonds = []
    for i, rigid_component in enumerate(rigid_components):
        frag_mol = keep_atoms(Chem.Mol(mol), rigid_component)
        Chem.GetSSSR(frag_mol)
        frag_mol.UpdatePropertyCache(strict = False)
        scaffold = MurckoScaffold.GetScaffoldForMol(frag_mol)
        if scaffold.GetNumAtoms() == 0:
            new_fragments.append(rigid_component)
            continue
        scaffold_atom_indices = [a.GetIntProp("__orig_idx") for a in scaffold.GetAtoms()]
        substituents_mol = Chem.ReplaceCore(frag_mol, scaffold)
        substituent_indices = rdmolops.GetMolFrags(substituents_mol)
        substituent_indices = [[substituents_mol.GetAtomWithIdx(a).GetIntProp("__orig_idx") for a in fragment if substituents_mol.GetAtomWithIdx(a).GetSymbol() != "*"] for fragment in substituent_indices]
        substituent_endpoints = [b.GetBeginAtom().GetIntProp("__orig_idx") if (b.GetEndAtom().GetSymbol() == "*") else b.GetEndAtom().GetIntProp("__orig_idx") for b in substituents_mol.GetBonds() if (b.GetBeginAtom().GetSymbol() == "*") or (b.GetEndAtom().GetSymbol() == "*")]
        if include_endpoints:
            for j, end_point in enumerate(substituent_endpoints):
                for neighbor in mol.GetAtomWithIdx(end_point).GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in scaffold_atom_indices:
                        substituent_indices[j].append(neighbor_idx)
        for endpoint in substituent_endpoints:
            for b in mol.GetAtomWithIdx(endpoint).GetBonds():
                neighbor_idx = b.GetOtherAtomIdx(endpoint)
                if neighbor_idx in scaffold_atom_indices:
                    new_atom_bonds.append((endpoint, neighbor_idx))
        new_fragments.append(scaffold_atom_indices)
        new_fragments.extend(substituent_indices)


    # order fragments by lowest atom index
    new_fragments = sorted(new_fragments, key = lambda x: min(x))
    connected_atom_indices = [bond_idx_to_atom_bond_idx[tuple(bond)] for bond in np.argwhere(adjacency_matrix == 1)]
    new_atom_bonds.extend(connected_atom_indices)
    atom_component_association = {atom : idx for idx, component in enumerate(new_fragments) for atom in component}
    connected_component_indices = [(atom_component_association[b[0]], atom_component_association[b[1]]) for b in new_atom_bonds]
    new_adjacency_matrix = np.zeros((len(new_fragments), len(new_fragments)), dtype = np.int64)
    for i, j in zip(connected_component_indices[::2], connected_component_indices[1::2]):
        new_adjacency_matrix[i, j] = 1
        new_adjacency_matrix[j, i] = 1

    return new_fragments, new_adjacency_matrix, {idx : tuple(bond) for idx, bond in enumerate(new_atom_bonds)}
    


def get_kinematic_graph(mol):
    """
    Returns the kinematic graph of a molecule. The kinematic graph is a graph where each node is a rigid substructure of the molecule.
    Two nodes are connected if they are connected by a rotatable bond.

    Args:
        mol (RDKit.Mol): Molecule to determine the kinematic graph for.

    Returns:
        list: List of lists, where each list contains the atom indices of a rigid substructure.
        numpy.ndarray: Adjacency matrix of the kinematic graph.
        dict: Dictionary mapping bond indices to the atom indices of the connected atoms.
    """

    chain = get_kinematic_chain(mol)
    # now we determine the bonds between rigid substructures
    rotatable_bonds = np.zeros((len(chain), len(chain)), dtype = np.int64)
    rotatable_bond_to_bond_idx = {}
    # a bond is rotatable if it connects two rigid substructures
    for i, rigid_substructure in enumerate(chain):
        for atom_idx in rigid_substructure:
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in rigid_substructure:
                    # identify chain of neighbor
                    for j, neighbor_chain in enumerate(chain):
                        if neighbor.GetIdx() in neighbor_chain:
                            rotatable_bonds[i, j] = 1
                            rotatable_bonds[j, i] = 1
                            rotatable_bond_to_bond_idx[(i, j)] = (atom_idx, neighbor.GetIdx())
                            rotatable_bond_to_bond_idx[(j, i)] = (neighbor.GetIdx(), atom_idx)
                            break
    return chain, rotatable_bonds, rotatable_bond_to_bond_idx


def get_bond_connected_atoms(mol, bond_start_atom_idx, bond_end_atom_idx):
    """
    Returns the atom indices of the atoms connected to bond_start_atom that are not connected to bond_end_atom except by this bond.

    Args:
        mol (RDKit.Mol): Molecule to get the connected atoms from.
        bond_start_atom_idx (int): Index of the atom where the bond starts.
        bond_end_atom_idx (int): Index of the atom where the bond ends.

    Returns:
        list(int): List of atom indices that are connected to bond_start_atom_idx but not to bond_end_atom_idx.
    """

    # get adjacaency matrix
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    # make it symmetric
    adj_matrix += adj_matrix.T; adj_matrix = np.clip(adj_matrix, 0, 1)
    adj_matrix += np.eye(adj_matrix.shape[0], dtype=np.int32)
    
    # eliminate the bond between the two atoms
    adj_matrix[bond_start_atom_idx, bond_end_atom_idx] = 0
    adj_matrix[bond_end_atom_idx, bond_start_atom_idx] = 0

    # propagate until no change is made. Since we want to get as fast as possible we can go exponentially
    last_adj_matrix = None
    while (last_adj_matrix != adj_matrix).any():
        last_adj_matrix = adj_matrix
        adj_matrix = np.dot(adj_matrix, adj_matrix)
        adj_matrix = np.clip(adj_matrix, 0, 1)
    # get the connected atoms
    return np.where(adj_matrix[bond_start_atom_idx] == 1)[0]
