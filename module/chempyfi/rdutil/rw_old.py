import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from rdkit.Chem import AllChem
import logging
from rdkit.Geometry import Point3D
from rdkit import Chem
from ..mathutils.geometry import calculate_angle, align_directions, rigid_transform
from .geometry import position 
from itertools import combinations, chain
from rdkit.Chem import rdmolops
from scipy.spatial.transform import Rotation
from .bond_order_inference import correct_bond_orders as correct_bond_orders_func
from rdkit.Chem.PropertyMol import PropertyMol

def remove_atoms(mol, atoms):
    """Remove the given atoms from the molecule.
    
    Args:
        mol (RDKit.Mol): Molecule to remove the atoms from.
        atoms (list(RDKit.Atom or int)): List of atoms to remove. If list of ints, the atoms with the given indices will be removed.
            Passing indices will generally be faster.        
    Returns:
        RDKit.Mol: Molecule with the given atoms removed.
        
    """
    import rdkit.Chem as Chem
    from .geometry import position

    indices = []
    for atom in atoms:
        if np.issubdtype(type(atom), np.integer):
            indices.append((atom, None))
        else:
            indices.append((atom.GetIdx(), atom))
    # we need to remove the atoms in reverse order to not mess up the indices
    indices.sort(reverse=True)
    mol = Chem.RWMol(mol)
    removed = set()
    for atom_idx, atom in indices:
        if atom_idx in removed:
            continue
        if atom is not None:
            other_atom = mol.GetAtomWithIdx(atom_idx)
            if (other_atom.GetSymbol() != atom.GetSymbol()) or (not np.allclose(position(other_atom), position(atom))):
                raise ValueError("The atom at index {} is not the same as the given atom.".format(atom_idx))
            mol.RemoveAtom(atom_idx)
        else:
            mol.RemoveAtom(atom_idx)
    return mol.GetMol()

def keep_atoms(mol, atoms):
    """Remove all atoms except the given atoms from the molecule.
    
    Args:
        mol (RDKit.Mol): Molecule to remove the atoms from.
        atoms (list(RDKit.Atom or int)): List of atoms to keep. If list of ints, the atoms with the given indices will be kept.
            Passing indices will generally be faster.
            
    Returns:
        RDKit.Mol: Molecule with all atoms except the given atoms removed.
        
    """
    all_atom_indices = set([atom.GetIdx() for atom in mol.GetAtoms()])
    indices = set()
    for atom in atoms:
        if np.issubdtype(type(atom), np.integer):
            indices.add(atom)
        else:
            indices.add(atom.GetIdx())
    to_remove = all_atom_indices - indices
    return remove_atoms(mol, to_remove)

ideal_tetraheder = np.array([[0,0,0], [0.0, 0.0, 1.0], [0.0, 0.8944271909999159, -0.4472135954999579], [-0.816496580927726, -0.408248290463863, -0.408248290463863], [0.816496580927726, -0.408248290463863, -0.408248290463863]])

def add_explicit_hydrogens(mol):
    """Crude function to add hydrogens to a molecule in a simple fashion.
    Currently only works for the organic elements.
    
    Args:
        mol (RDKit.Mol): Molecule to add hydrogens to.
        
    Returns:
        RDKit.Mol: Molecule with explicit hydrogens added.
        
    """
    from rdkit import Chem
    from .geometry import position
    from ..mathutils.geometry import rigid_transform, apply_transform
    from rdkit.Chem.rdmolops import RemoveHs

    mol = RemoveHs(mol, implicitOnly=True)
    mol = Chem.RWMol(mol)
    
    periodic_table = Chem.GetPeriodicTable()
    for atom in list(mol.GetAtoms()):
        possible_valence = periodic_table.GetDefaultValence(atom.GetAtomicNum())

        neighbors = list(atom.GetNeighbors())
        cur_valence = atom.GetExplicitValence()
        missing_hydrogens =  possible_valence - cur_valence

        if missing_hydrogens > 0:
            neighbor_directions = np.array([position(neighbor) - position(atom) for neighbor in neighbors]); neighbor_directions /= np.linalg.norm(neighbor_directions, axis=1)[:,None]
            neighbor_directions = np.concatenate([np.zeros((1, 3)), neighbor_directions])
            
            hydrogen_neighbors = [neighbor for neighbor in neighbors if neighbor.GetAtomicNum() == 1]
            hydrogen_distance = 1.0 if not atom.GetSymbol() == "C" else 1.1
            if len(hydrogen_neighbors) > 0:
                hydrogen_distance = np.mean([np.linalg.norm(position(neighbor) - position(atom)) for neighbor in hydrogen_neighbors])
            if possible_valence == 4:
                # sp3 hybridization => match tetrahedral geometry to neighbors

                    
                    R, t  = rigid_transform(ideal_tetraheder[:len(neighbor_directions)], neighbor_directions)
                    new_positions = apply_transform(ideal_tetraheder[len(neighbor_directions):], R, t)
                    for new_position in new_positions:
                        new_position = new_position * hydrogen_distance + position(atom)
                        mol.AddAtom(Chem.Atom(1))
                        new_idx = mol.GetNumAtoms() - 1
                        mol.AddBond(atom.GetIdx(), new_idx, Chem.BondType.SINGLE)
                        mol.GetConformer().SetAtomPosition(new_idx, new_position)

    return mol
                  


def copy_props(mol_from, mol_to, replace_dict = None):
    if replace_dict is None:
        replace_dict = {}

    for key, prop in mol_from.GetPropsAsDict().items():
        if key in replace_dict:
            key = replace_dict[key]
        if np.issubdtype(type(prop), np.integer):
            mol_to.SetIntProp(key, prop)
        elif np.issubdtype(type(prop), np.floating):
            mol_to.SetDoubleProp(key, prop)
        else:
            mol_to.SetProp(str(key), prop)



def build_molecule(positions, elements, bond_indices, bond_orders, formal_charges=None, hybridizations=None, sanitize=True):
    """
    Constructs an RDKit molecule from atom positions, element symbols, bond indices, bond orders,
    and optionally formal charges and hybridization states.

    Args:
        positions (numpy.ndarray): A NumPy array of shape (N, 3) representing the (x, y, z) coordinates of N atoms.
        elements (list): List of element symbols (e.g., ['C', 'O', 'N']) corresponding to the atoms.
        bond_indices (list): List of tuples representing the bonded atom indices. Bonds are assumed to be between atoms at the same index in `from_atoms` and `to_atoms`.
        bond_orders (list): List of bond orders as strings (e.g., "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "ONEANDAHALF").
        formal_charges (list, optional): List of formal charges corresponding to each atom. If not provided, formal charges default to 0 for all atoms.
        hybridizations (list, optional): List of hybridization states for each atom as strings (e.g., "SP", "SP2", "SP3"). If not provided, hybridization defaults to unspecified for all atoms.
        sanitize (bool, optional): Whether to sanitize the molecule after construction. Defaults to True.

    Returns:
        rdkit.Chem.rdchem.Mol: An RDKit molecule object with the specified atoms, bonds, and coordinates.
    
    Raises:
        ValueError: If the positions array does not have the shape (N, 3).
    
    Example:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        elements = ['C', 'O', 'N']
        bond_indices = [(0, 1), (1, 2)]
        bond_orders = ["DOUBLE", "SINGLE"]
        formal_charges = [0, -1, 0]
        hybridizations = ["SP2", "SP3", "SP2"]

        molecule = build_molecule(positions, elements, bond_indices, bond_orders, formal_charges, hybridizations)
    """
    # Ensure positions is a NumPy array
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions)
    
    # Ensure the array has the right shape
    if positions.shape[1] != 3:
        raise ValueError("Positions array must have shape (N, 3) for 3D coordinates")
    
    # Create an editable molecule
    mol = Chem.RWMol()
    
    # Hybridization state mapping
    hybridization_mapping = {
        "SP": Chem.HybridizationType.SP,
        "SP2": Chem.HybridizationType.SP2,
        "SP3": Chem.HybridizationType.SP3,
        "SP3D": Chem.HybridizationType.SP3D,
        "SP3D2": Chem.HybridizationType.SP3D2
    }
    
    # Add atoms to the molecule and assign optional formal charges and hybridization states
    for i, elem in enumerate(elements):
        atom = Chem.Atom(elem)
        
        # Set formal charge if provided
        if formal_charges is not None:
            atom.SetFormalCharge(int(formal_charges[i]))
        
        # Set hybridization if provided
        if hybridizations is not None:
            hybrid_state = hybridization_mapping.get(hybridizations[i], Chem.HybridizationType.UNSPECIFIED)
            atom.SetHybridization(hybrid_state)
        
        mol.AddAtom(atom)
    
    # Map bond order strings to RDKit bond types
    bond_order_mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
        "ONEANDAHALF": Chem.BondType.AROMATIC  # Map ONEANDAHALF to AROMATIC bond type
    }
    
    # Add bonds to the molecule
    for (start_idx, end_idx), bond_order in zip(bond_indices, bond_orders):
        bond_type = bond_order_mapping.get(bond_order, Chem.BondType.SINGLE)
        mol.AddBond(int(start_idx), int(end_idx), bond_type)
    
    # Create a conformer and set atom positions
    conf = Chem.Conformer(len(positions))
    for i, pos in enumerate(positions):
        x, y, z = pos
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    
    # Finalize the molecule
    mol.UpdatePropertyCache(strict=False)
    if sanitize:
        Chem.SanitizeMol(mol)
    
    return mol


def kekulize_mol(mol):
    from rdkit.Chem import rdmolops
    from rdkit.Chem import rdchem
    rdmolops.FastFindRings(mol)

    mol = mol.GetMol()
    # remove implicit hydrogens
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    atom_to_ring = {atom_idx : set() for atom_idx in range(mol.GetNumAtoms())}
    for ring in atom_rings:
        for atom_idx in ring:
            atom_to_ring[atom_idx].add(ring)
    possibly_aromatic = set()
    for atom in mol.GetAtoms():
        if atom.GetNumImplicitHs() > 0:
            possibly_aromatic.add(atom.GetIdx())
        atom.SetNoImplicit(True)  # Disable automatic calculation of implicit hydrogens
    for bond in mol.GetBonds():
        if (bond.GetBondType() == rdchem.BondType.SINGLE) and \
        ((bond.GetBeginAtomIdx() in possibly_aromatic) or (bond.GetEndAtomIdx() in possibly_aromatic)) and \
        (atom_to_ring[bond.GetBeginAtomIdx()] & atom_to_ring[bond.GetEndAtomIdx()]):
            bond.SetBondType(rdchem.BondType.AROMATIC)

    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    # if kekulize didnt catch the ring, its not aromatic
    for bond in mol.GetBonds():
        if bond.GetBondType() == rdchem.BondType.AROMATIC:
            if not bond.GetIsAromatic():
                bond.SetBondType(rdchem.BondType.SINGLE)

    mol.UpdatePropertyCache(strict=True)
    return mol

def proximity_bond(mol, correct_bond_orders = True, allow_hydrogen_gas = False, try_full = True, as_property_mol = False):
    """
    Infer bonds from the proximity of atoms. This is done by checking the distance of each atom to each other atom.
    If the distance is smaller than the sum of the van-der-waals radii, a bond is inferred.
    This method is not transitive, i.e. if atom A is close to atom B and atom B is close to atom C, A and C are not necessarily close.
    A new molecule is created and returned.

    Args:
        mol (RDKit.Mol): Molecule to infer bonds for.
        kekulize (bool): Defaults to True. If True, molecule will be kekulized after adding bonds.
        handle_valency (str): Defaults to "auto". If "auto", valency violations will be resolved by removing the most unlikely bond. If "ignore", valency violations will be ignored. If "raise", valency violations will raise an error.
        allow_hydrogen_gas (bool): Defaults to False. If True, hydrogen gas will be allowed. This means that hydrogen atoms will not be considered to have valency violations.

    Returns:
        RDKit.Mol: Molecule with inferred bonds.

    """
    from rdkit.Chem import rdchem
    from rdkit.Chem import rdMolTransforms
    from rdkit.Chem import rdmolops
    from .util import bond_lengths, ideal_bond_lengths
    from .geometry import position
    from scipy.spatial import cKDTree

    # get the positions of the atoms
    positions = position(mol)
    periodic_table = Chem.GetPeriodicTable()
    elements = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    # create a kdtree for the atoms
    tree = cKDTree(positions)

    # query the tree for atoms that are closer than maximum sensible bond length
    d_matrix = tree.sparse_distance_matrix(tree, max_distance = 2.5, output_type = 'coo_matrix')

    # get bond_distances for each pair of atoms
    bond_distances = np.array([bond_lengths[elements[i]][elements[j]] for i, j in zip(d_matrix.row, d_matrix.col)])
    
    cond = (d_matrix.data < bond_distances[:,0] * 1.05) & (d_matrix.row < d_matrix.col)
    close_atoms = np.vstack([d_matrix.row[cond], d_matrix.col[cond]]).T
    mol = Chem.RWMol(mol)
    # create a bond for each pair of close atoms
    for idx, (i, j) in enumerate(close_atoms):
        if not allow_hydrogen_gas:
            if (elements[i] == "H") and (elements[j] == "H"):
                continue
        dist = d_matrix.data[cond][idx]
        if dist < bond_distances[cond][idx][3]:
            if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                mol.AddBond(int(i), int(j), rdchem.BondType.TRIPLE)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.TRIPLE)
        elif dist < bond_distances[cond][idx][2]:
            if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                mol.AddBond(int(i), int(j), rdchem.BondType.DOUBLE)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.DOUBLE)
        elif dist < bond_distances[cond][idx][1]:
            bond_order = rdchem.BondType.SINGLE#ONEANDAHALF
            if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                mol.AddBond(int(i), int(j), bond_order)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(bond_order)
        else:
            if mol.GetBondBetweenAtoms(int(i), int(j)) is None:
                mol.AddBond(int(i), int(j), rdchem.BondType.SINGLE)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.SINGLE)

    # check for valency violations
    mol.UpdatePropertyCache(strict=False) 
    if correct_bond_orders:
        mol = correct_bond_orders_func(mol, try_full=try_full)

    return PropertyMol(mol) if as_property_mol else mol.GetMol()



def ff_align(aligned_mol, target_mol, environment_mol = None, ff_atempts = 5, return_query = False):
    """
    Aligns a molecule to a reference molecule using the force field method.
    The molecule to be aligned will be modified in place.

    Args:
        mol (RDKit.Mol): Molecule to align.
        ref_mol (RDKit.Mol): Reference molecule.

    Returns:
        RDKit.Mol: Aligned molecule.

    """
    from rdkit.Chem.rdFMCS import FindMCS
    from scipy.optimize import linear_sum_assignment
    from sklearn.decomposition import PCA
    from .io import write_molecules
    # make sure input molecules have ring info
    Chem.GetSSSR(aligned_mol)
    Chem.GetSSSR(target_mol)

    # identify MCS
    # remove bond orders
    aligned_mol = Chem.RWMol(aligned_mol)
    target_mol = Chem.RWMol(target_mol)
    for bond in aligned_mol.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    for bond in target_mol.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)

    write_molecules(aligned_mol, "aligned.sdf")
    write_molecules(target_mol, "target.sdf")    
    mcs = FindMCS([aligned_mol, target_mol], completeRingsOnly=True, ringMatchesRingOnly=True, timeout = 1).queryMol


    # match MCS in mol and ref_mol

    attachment_points = {}
    substructure_matches = {}
    for name, mol in zip(["aligned", "target"], [aligned_mol, target_mol]):
        mol_match = mol.GetSubstructMatch(mcs)
        substructure_matches[name] = mol_match
        # identify rest groups with attachment points
        for atom in mol.GetAtoms():
            atom.SetIntProp("__orig_index", atom.GetIdx())
        mol_rest = remove_atoms(mol, mol_match)
        mol_frags = Chem.GetMolFrags(mol_rest, asMols=True)

    # determine attachment point for each fragment
        mol_match = np.array(mol_match)
        attachments = {}
        for mol_frag in mol_frags:
            # map back indices
            frag_indices = tuple([atom.GetIntProp("__orig_index") for atom in mol_frag.GetAtoms()])
            for atom_idx in frag_indices:
                mask = np.isin(mol_match, [n.GetIdx() for n in mol.GetAtomWithIdx(atom_idx).GetNeighbors()])
                MCS_neighbor_indices = np.arange(len(mask))[mask]
                if len(MCS_neighbor_indices) == 0:
                    continue
                if (frag_indices in attachments) and (attachments[frag_indices] != MCS_neighbor_indices):
                    raise ValueError("Non-matching fragment has multiple attachment points. This is not supported.")
                attachments[frag_indices] = MCS_neighbor_indices
        # invert attachments dictionary
        possible_attachments = []
        for frag_indices, MCS_neighbor_indices in attachments.items():
            for MCS_neighbor_idx in MCS_neighbor_indices:
                possible_attachments.append((tuple(MCS_neighbor_indices), tuple(frag_indices)))
        attachment_points[name] = possible_attachments

    # next we match attachments by attachment points and number of atoms on ambiguity
    def calculate_score(assignment1, assignment2):
        key_intersection_size = len(set(assignment1[0]).intersection(assignment2[0]))
        if key_intersection_size == 0:
            return 1e6
        value_size_difference = np.abs(len(assignment1[1]) - len(assignment2[1]))
        return value_size_difference/key_intersection_size

    cost_matrix = np.zeros((len(attachment_points["aligned"]), len(attachment_points["target"])))
    for i, value1 in enumerate(attachment_points["aligned"]):
        for j, value2 in enumerate(attachment_points["target"]):
            cost_matrix[i, j] = calculate_score(value1, value2)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    fragment_assignments = {}
    for i, j in zip(row_ind, col_ind):
        fragment_assignments[attachment_points["aligned"][i]] = attachment_points["target"][j]

    aligned_mol = proximity_bond(aligned_mol)
    aligned_mol = Chem.RWMol(aligned_mol)
    target_COMs = {}
    target_attachments = {}
    # we can now step through the assigned fragments and transform them
    for i, (aligned_attachment, target_attachment) in enumerate(fragment_assignments.items()):
        target_attachment_atoms = [target_mol.GetAtomWithIdx(substructure_matches["target"][i]) for i in target_attachment[0]]
        mean_target_attachment_positions = np.mean(position(target_attachment_atoms), axis=0)

        target_fragment_atoms = [target_mol.GetAtomWithIdx(i) for i in target_attachment[1]]
        target_COM = np.mean(position(target_fragment_atoms), axis=0) - mean_target_attachment_positions; target_COM /= np.linalg.norm(target_COM)
        if np.linalg.norm(target_COM) < 1e-6:
            MCS_Neighbors = [n.GetIdx() for n in target_attachment_atoms[0].GetNeighbors() if n.GetIdx() in substructure_matches["target"]]
            target_COM = -(np.mean(position(target_attachment_atoms[0].GetNeighbors(), axis=0) - mean_target_attachment_positions))

        target_COMs[aligned_attachment] = target_COM
        target_attachments[aligned_attachment] = mean_target_attachment_positions

    
    #next we handle unmatched attachments:
    for aligned_attachment in attachment_points["aligned"]:
        if aligned_attachment in fragment_assignments:
            continue
        target_attachment_atom = target_mol.GetAtomWithIdx(substructure_matches["target"][aligned_attachment[0][0]])
        MCS_Neighbors = [n.GetIdx() for n in target_attachment_atom.GetNeighbors() if n.GetIdx() in substructure_matches["target"]]
        target_COM = -(np.mean(position(target_attachment_atom.GetNeighbors()), axis=0) - position(target_attachment_atom))
        target_COMs[aligned_attachment] = target_COM
        target_attachments[aligned_attachment] = position(target_attachment_atom)

        

    for aligned_attachment, target_COM in target_COMs.items():
        aligned_attachment_atoms = [aligned_mol.GetAtomWithIdx(substructure_matches["aligned"][i]) for i in aligned_attachment[0]]
        mean_aligned_attachment_positions = np.mean(position(aligned_attachment_atoms), axis=0)
        aligned_fragment_atoms = [aligned_mol.GetAtomWithIdx(i) for i in aligned_attachment[1]]
        aligned_positions = position(aligned_fragment_atoms)
        aligned_COM = np.mean(aligned_positions, axis=0) - mean_aligned_attachment_positions; aligned_COM /= np.linalg.norm(aligned_COM)
        # get the rotation to align aligned_COM to target_COM
        rot_matrix = align_directions(aligned_COM, target_COM)

        # apply rotation to aligned positions
        aligned_positions = aligned_positions - mean_aligned_attachment_positions
        aligned_positions = aligned_positions @ rot_matrix.T
        aligned_positions = aligned_positions + target_attachments[aligned_attachment]

        # apply translation to aligned positions
        translation = target_attachments[aligned_attachment] - mean_aligned_attachment_positions

        # assign new positions to atoms
        for atom, new_position in zip(aligned_fragment_atoms, aligned_positions):
            aligned_mol.GetConformer().SetAtomPosition(atom.GetIdx(), new_position)


    # copy over all MCS positions
    for aligned_idx, target_idx in zip(substructure_matches["aligned"], substructure_matches["target"]):
        aligned_mol.GetConformer().SetAtomPosition(aligned_idx, target_mol.GetConformer().GetAtomPosition(target_idx))



    if ff_atempts > 0:
        # now we apply a constrained FF minimization
        Chem.SanitizeMol(aligned_mol)
        ff = AllChem.UFFGetMoleculeForceField(aligned_mol)
        for idx in substructure_matches["aligned"]:
            ff.UFFAddPositionConstraint(idx, 0, 1e6)
        aligned_positions = position(aligned_mol)
        if environment_mol is not None:
            # construct kd-Tree of aligned_mol
            aligned_positions = position(aligned_mol)
            aligned_tree = cKDTree(aligned_positions)
            
            # find all atoms in environment_mol that are close to aligned_mol
            environment_positions = position(environment_mol)
            close_atoms = aligned_tree.query_ball_point(environment_positions, 5)
            # invert close atoms
            for atom_idx, aligned_idx in enumerate(close_atoms):
                if len(aligned_idx) == 0:
                    continue
                idx = ff.AddExtraPoint(*[int(x) for x in position(environment_mol.GetAtomWithIdx(atom_idx))], fixed = True)
                for aligned_id in aligned_idx:
                    ff.UFFAddDistanceConstraint(aligned_id, idx -1 , False, 1.5, 1000, 1e4)

        #print(ff.Positions(), ff.NumPoints())
        ff.Initialize()
        for i in range(ff_atempts):
            converged = ff.Minimize(maxIts = 2000)
            if converged == 0:
                break
            #AllChem.MMFFRandomShake(aligned_mol, ff)
        if converged != 0:
            logging.warning("FF minimization did not converge.")
    if return_query:
        return aligned_mol.GetMol(), mcs
    else:
        return aligned_mol

    # first we match by attachment points, for this we need to map the aligned attachment indices to the target indices

def kinematic_chain_weights(mol, chain, return_node_weights = True):
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
                            rotatable_bond_to_bond_idx[(j, j)] = (neighbor.GetIdx(), atom_idx)
                            break
    
    # next we assign edge weights based on the number of atoms connected via them
    # for these we run a bottom up BFS through the rigid component tree
    next_edges = set()
    # leafs are only connected once
    for i, rigid_substructure in enumerate(chain):
        edges = np.where(rotatable_bonds[i] == 1)[0]
        if len(edges) == 1:
            next_edges.add((i, edges[0]))
    
    # now we run a BFS from the leafs to the root, a node is only visited if all its children have been visited (i.e. only one edge has not been visited)
    visited_edges = set()
    edge_weights = np.zeros_like(rotatable_bonds)
    untreated_edges = np.copy(rotatable_bonds)

    while len(next_edges) > 0:
        nodes_to_consider = set()
        for edge in next_edges:
            if edge in visited_edges:
                continue
            # weight is sum of all incoming edges + number of atoms in rigid substructure
            edge_weights[edge[0], edge[1]] = np.sum(edge_weights[:, edge[0]][rotatable_bonds[:, edge[0]] == 1]) + len(chain[edge[0]]) - edge_weights[edge[1], edge[0]]
            visited_edges.add(edge)
            nodes_to_consider.add(edge[1])
            untreated_edges[edge] = 0
        # discover which edges to consider next
        next_edges = set()
        for node in nodes_to_consider:
            # check for each outgoing edge if all other incoming edges have been visited
            for outgoing_edge_end in np.where(untreated_edges[node] == 1)[0]:
                if np.sum(untreated_edges[:, node]) - untreated_edges[outgoing_edge_end, node] == 0:
                    next_edges.add((node, outgoing_edge_end))
    edge_weights = edge_weights
    
    if return_node_weights:
        # figure out weights for each atom
        node_weights = {}
        for edge in np.argwhere(edge_weights > 0):
            edge = tuple(edge)
            atom_edge = rotatable_bond_to_bond_idx[edge]
            if atom_edge[0] not in node_weights:
                node_weights[atom_edge[0]] = []
            node_weights[atom_edge[0]] += [edge_weights[edge]]
        node_weights = {node_idx : np.pad(sorted(weights, reverse=True), pad_width=(0,8-len(weights)), mode="constant") for node_idx, weights in node_weights.items()}
        return node_weights
    else:
        return edge_weights
        
def find_optimal_rotation(axis, points1, points2, weights):
    """
    Find the optimal rotation angle around a fixed axis using an analytical solution.

    Parameters:
    - axis: Rotation axis (3D vector), will be normalized.
    - points1: Source points (n x 3 array).
    - points2: Target points (n x 3 array).
    - weights: Weights for each point pair (n array).

    Returns:
    - theta: Optimal rotation angle in radians.
    """
    # Ensure axis is a unit vector
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)

    # Project points onto the plane perpendicular to the rotation axis
    # For a point p, projection = p - (p ⋅ axis) * axis
    proj1 = points1 - np.dot(points1, axis[:, None]) * axis[None, :]
    proj2 = points2 - np.dot(points2, axis[:, None]) * axis[None, :]

    # Compute cross product terms (p_i × q_i) ⋅ axis
    cross = np.cross(proj1, proj2)
    cross_terms = np.dot(cross, axis)  # Shape (n,)

    # Compute dot product terms (p_i ⋅ q_i)
    dot_terms = np.sum(proj1 * proj2, axis=1)  # Shape (n,)

    # Weighted sums
    S_cross = np.sum(weights * cross_terms)
    S_dot = np.sum(weights * dot_terms)

    # Optimal angle (avoid division by zero with arctan2)
    theta = np.arctan2(S_cross, S_dot)

    return theta


def seed_align(aligned_mol, target_mol, aligned_seed_indices, target_seed_indices):
    """
    Aligns two molecules using a seed of atoms that are already aligned. For this, we:
    1. Map each seed structure to each other.
    2. Calculate the transformation matrix that aligns the seed structures.
    3. Apply the transformation matrix to the whole molecule.
    4. Next we identify the rigid substructures in the aligned_mol.
    5. We run a BFS going over each rotatable bond, try out multiple rotations, score each rotation based on the atoms in the rigid substructure and choose the best ones to continue.
    6. Scoring is based on distance of each closest atom, matching elements and size of remaining atoms.

    Args:
        aligned_mol (RDKit.Mol): Molecule to align.
        target_mol (RDKit.Mol): Reference molecule.
        aligned_seed_indices (list): List of atom indices in aligned_mol to use as seed.
        target_seed_indices (list): List of atom indices in target_mol to use as seed.

    Returns:
        RDKit.Mol: Aligned molecule.

    """

    from rdkit.Chem import rdFMCS
    from rdkit.Chem import rdMolTransforms
    from .geometry import position
    from .util import bond_lengths
    from .select import get_kinematic_chain
    from scipy.spatial import cKDTree

    # first we create a kd-Tree of the target molecule
    target_positions = position(target_mol)
    target_tree = cKDTree(target_positions)


    # first we map the seed atoms onto each other
    aligned_seed_positions = position([aligned_mol.GetAtomWithIdx(i) for i in aligned_seed_indices])
    target_seed_positions = position([target_mol.GetAtomWithIdx(i) for i in target_seed_indices])
    R, t = rigid_transform(aligned_seed_positions, target_seed_positions)
    print(f"{R=}")
    # apply the transformation to the whole molecule
    aligned_positions = position(aligned_mol)
    aligned_positions = aligned_positions @ R.T + t

    import pymolviz as pmv # DEBUG
    pmv.Points(aligned_positions, name = "Aligned_Positions").write("seed_alignment.py") # DEBUG

    # next we identify the kinematic chain of the aligned molecule
    aligned_chain = get_kinematic_chain(aligned_mol)

    # now we need to expand the seed indices to all rigid substructures
    aligned_seed_indices = set(aligned_seed_indices)
    aligned_seed_chain_indices = set([i for i, rigid_component in enumerate(aligned_chain) if len(set(rigid_component).intersection(aligned_seed_indices)) > 0])
    aligned_seed_indices = set(sum([list(aligned_chain[i]) for i in aligned_seed_chain_indices], []))

    # now we determine the bonds between rigid substructures
    rotatable_bonds = np.zeros((len(aligned_chain), len(aligned_chain)), dtype = np.int64)
    rotatable_bond_to_bond_idx = {}
    # a bond is rotatable if it connects two rigid substructures
    for i, rigid_substructure in enumerate(aligned_chain):
        for atom_idx in rigid_substructure:
            atom = aligned_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() not in rigid_substructure:
                    # identify chain of neighbor
                    for j, neighbor_chain in enumerate(aligned_chain):
                        if neighbor.GetIdx() in neighbor_chain:
                            rotatable_bonds[i, j] = 1
                            rotatable_bonds[j, i] = 1
                            rotatable_bond_to_bond_idx[(i, j)] = (atom_idx, neighbor.GetIdx())
                            break
    # next we assign edge weights based on the number of atoms connected via them
    # for these we run a bottom up BFS through the rigid component tree
    aligned_atom_size_weights = kinematic_chain_weights(aligned_mol, aligned_chain)
    target_chain = get_kinematic_chain(target_mol)
    target_atom_size_weights = kinematic_chain_weights(target_mol, target_chain)

    # starting from our seed structure, we now determine the outgoing rotatable bonds
    seen_bonds = set()
    seen_components = set(aligned_seed_chain_indices)
    next_bonds = set()
    # since the existing bonds are symmetric, we only need to consider one direction
    transformation_tree = {
        "transformation": np.eye(4),
        "affected_indices" : [],
        "bond": "root",
        "children": [],
        "used_thetas": []
    }
    tree_node_mapping = {
        0 : transformation_tree
    }
    for bond in np.argwhere(rotatable_bonds == 1):
        if bond[0] in aligned_seed_chain_indices:
            if bond[1] in aligned_seed_chain_indices:
                seen_bonds.add(bond)
                seen_bonds.add((bond[1], bond[0]))
            else:
                next_bonds.add((tuple(bond), 0, ()))
    stored_positions = {}
    bond_counter = 0
    # we should take a recursive approach here
    while len(next_bonds) > 0:
        next_bond, parent_node_id, forbidden_angles = next_bonds.pop()
        parent_node = tree_node_mapping[parent_node_id]
        parent_transform = parent_node["transformation"]
        print("Considering bond", rotatable_bond_to_bond_idx[next_bond])
        seen_bonds.add(next_bond)
        seen_bonds.add((next_bond[1], next_bond[0]))
        # identify connected rigid component

        # here we only check the 1 since we only add bonds where 0 is already seen
        if next_bond[1] in seen_components:
            print(f"Undetected Cycle in molecule for bond {rotatable_bond_to_bond_idx[next_bond]}, This will probably lead to distorted poses.")
            continue
        seen_components.add(next_bond[1])

        # now BFS logic is complete, we now need to determine the best rotation for each bond
        # we need to determine the atoms that are affected by the rotation
        affected_atom_indices = aligned_chain[next_bond[1]]
        # we should also consider outgoing edge entpoints
        considered_atom_indices = list(affected_atom_indices)
        for bond in np.argwhere(rotatable_bonds[next_bond[1]] == 1):
            considered_atom_indices.extend([rotatable_bond_to_bond_idx[(next_bond[1], i)][1] for i in bond])
        bond_start_atom_idx = rotatable_bond_to_bond_idx[next_bond][0]
        bond_end_atom_idx = rotatable_bond_to_bond_idx[next_bond][1]
        base_position = aligned_positions[rotatable_bond_to_bond_idx[next_bond][0]]
        end_bond_position = aligned_positions[rotatable_bond_to_bond_idx[next_bond][1]]
        considered_positions = aligned_positions[considered_atom_indices]
        considered_positions = considered_positions @ parent_transform[:3, :3].T + parent_transform[:3, 3]
        end_bond_position = end_bond_position @ parent_transform[:3, :3].T + parent_transform[:3, 3]
        center = np.mean(considered_positions, axis=0)
        radius = np.max(np.linalg.norm(considered_positions - center, axis=1)) + 1
        # get all atoms that are close to the affected atoms
        target_close_atom_indices = target_tree.query_ball_point(center, radius)
        if len(target_close_atom_indices) == 0:
            transformation = np.eye(4)
        else:
            # get affected and target weights
            considered_weights = np.array([aligned_atom_size_weights.get(i, np.zeros(8)) for i in considered_atom_indices])
            target_weights = np.array([target_atom_size_weights.get(i, np.zeros(8)) for i in target_close_atom_indices])
            # get the best matching

            # now we try different rotations around the bond
            # first we move both molecules to the origin
            considered_positions = considered_positions - base_position
            target_positions = position([target_mol.GetAtomWithIdx(i) for i in target_close_atom_indices]) - base_position
            # next we try rotations around the bond
            # we try 10 different rotations
            best_match = None
            element_differences = np.array([[1 if aligned_mol.GetAtomWithIdx(i).GetSymbol() != target_mol.GetAtomWithIdx(j).GetSymbol() else 0 for j in target_close_atom_indices] for i in considered_atom_indices])
            size_differences = np.sum(np.abs((considered_weights[:, None, :] - target_weights[None, :, :])), axis = -1)
            best_score = 1e6
            bond_direction = (end_bond_position - base_position); bond_direction /= np.linalg.norm(bond_direction)
            best_weights = None
            best_angle = None
            # super ugly solution but I am running out of time:
            seen_atom_indices = set(sum([list(aligned_chain[i]) for i in seen_components], []))
            self_other_positions = aligned_positions[[i for i in seen_atom_indices if i not in [bond_start_atom_idx, bond_end_atom_idx] + list(considered_atom_indices)]]
            self_other_positions = self_other_positions @ parent_transform[:3, :3].T + parent_transform[:3, 3]
            self_other_kd_tree = cKDTree(self_other_positions)
            curr_collision_threshold = 2
            while (best_score > 1e5) and curr_collision_threshold > 1:
                for i in range(36):
                    angle = i * np.pi / 18
                    rotation = Rotation.from_rotvec(bond_direction * angle)
                    rotated_positions = rotation.apply(considered_positions)
                    # check if new positions clash with other atoms
                    if np.any(self_other_kd_tree.query_ball_point(rotated_positions, curr_collision_threshold)):
                        continue
                    # to determine the score, we assign atoms by distance
                    distance_metric = np.linalg.norm(rotated_positions[:, None] - target_positions[None, :], axis = -1)
                    # scale size_differences down to be equal to 1 A distance
                    total = distance_metric
                    best_matching = linear_sum_assignment(total)
                    weights = (element_differences + size_differences)[best_matching]
                    score = np.sum(weights)
                    #nodes with larger size and low size difference should be rewarded
                    weights = 1/(weights + 1) * np.sum(considered_weights[best_matching[0]], axis = 1)
                    if score < best_score:
                        best_score = score
                        best_match = best_matching
                        best_weights = 1/(weights + 1)
                        best_angle = angle
                curr_collision_threshold -= .1
            if curr_collision_threshold <= 1:
                # we failed to find a non-colliding rotation
                # need to backtrack!
                print("Failed to find non-colliding rotation")
                # eliminate parent from grandfather
                tree_node_mapping[parent_node["parent_node_id"]]["children"].remove(parent_node)
                # eliminate 
            # now we determine rotation for the best match analytically
            base_positions = considered_positions[best_match[0]]
            target_positions = target_positions[best_match[1]]

            theta = find_optimal_rotation(bond_direction, base_positions, target_positions, np.ones(len(target_positions))) #best_weights)
            #theta = best_angle
            rotation = Rotation.from_rotvec(bond_direction * theta)

            # apply rotation to all affected atoms
            transformation_to_origin = np.eye(4)
            transformation_to_origin[:3, 3] = -base_position
            transformation_rotation = np.eye(4)
            transformation_rotation[:3, :3] = rotation.as_matrix()
            transformation_from_origin = np.eye(4)
            transformation_from_origin[:3, 3] = base_position
            transformation = transformation_from_origin @ transformation_rotation @ transformation_to_origin

        # add parent_transform to transformation
        transformation = transformation @ parent_transform

        # apply transformation to all affected atoms
        aligned_positions[list(affected_atom_indices)] = (aligned_positions[list(affected_atom_indices)] @ transformation[:3, :3].T + transformation[:3, 3])
        parent_node["children"].append({
            "transformation": transformation,
            "affected_indices" : list(affected_atom_indices),
            "bond": next_bond,
            "children": [],
            "used_thetas": [theta],
            "parent_node" : parent_node_id
        })
        tree_node_mapping[bond_counter] = parent_node["children"][-1]
        for bond in np.argwhere(rotatable_bonds == 1):
            if bond[0] == next_bond[1]:
                if tuple(bond) in seen_bonds:
                    continue
                print(f"Added {bond}")
                next_bonds.add((tuple(bond), bond_counter, ()))
        bond_counter += 1
    
    return stored_positions, transformation_tree

        


        



    

