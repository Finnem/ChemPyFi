import numpy as np
import os
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

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
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


def create_query_mol(mol):
    """
    Creates a molecule that has no defined bond orders to query as a substructure.
    """
    mol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        bond.SetBondType(Chem.BondType.UNSPECIFIED)
    return mol

def keep_atoms(mol, atoms, rebuild_threshold=1000, disable_rebuild=False, sanitize_rebuild=False):
    """Remove all atoms except the given atoms from the molecule.
    
    Args:
        mol (RDKit.Mol): Molecule to remove the atoms from.
        atoms (list(RDKit.Atom or int)): List of atoms to keep. If list of ints, the atoms with the given indices will be kept.
            Passing indices will generally be faster.
        rebuild_threshold (int): If the molecule has at least this many atoms, rebuild from kept atoms for speed.
        disable_rebuild (bool): Defaults to False. If True, always use removal instead of reconstruction.
        sanitize_rebuild (bool): Defaults to False. If True, sanitize the rebuilt molecule.
            
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
    if (not disable_rebuild) and mol.GetNumAtoms() >= rebuild_threshold:
        kept_indices = sorted(indices)
        if not kept_indices:
            return Chem.Mol()
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
        positions = position(mol)[kept_indices]
        elements = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in kept_indices]
        formal_charges = [mol.GetAtomWithIdx(idx).GetFormalCharge() for idx in kept_indices]
        hybridizations = [str(mol.GetAtomWithIdx(idx).GetHybridization()) for idx in kept_indices]
        bond_indices = []
        bond_orders = []
        kept_bonds = []
        # Prefer neighbor traversal when keeping a small subset of atoms
        if len(indices) < max(2000, int(0.2 * mol.GetNumAtoms())):
            for old_idx in kept_indices:
                atom = mol.GetAtomWithIdx(old_idx)
                for neighbor in atom.GetNeighbors():
                    neighbor_idx = neighbor.GetIdx()
                    if neighbor_idx in indices and neighbor_idx > old_idx:
                        bond = mol.GetBondBetweenAtoms(int(old_idx), int(neighbor_idx))
                        if bond is None:
                            continue
                        bond_indices.append((index_map[old_idx], index_map[neighbor_idx]))
                        bond_orders.append(bond.GetBondType())
                        kept_bonds.append(bond)
        else:
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                if a1 in indices and a2 in indices:
                    bond_indices.append((index_map[a1], index_map[a2]))
                    bond_orders.append(bond.GetBondType())
                    kept_bonds.append(bond)
        new_mol = build_molecule(
            positions,
            elements,
            bond_indices,
            bond_orders,
            formal_charges=formal_charges,
            hybridizations=hybridizations,
            sanitize=sanitize_rebuild,
        )
        try:
            copy_props(mol, new_mol)
            for old_idx, new_idx in index_map.items():
                old_atom = mol.GetAtomWithIdx(old_idx)
                new_atom = new_mol.GetAtomWithIdx(new_idx)
                for prop_name in old_atom.GetPropNames():
                    new_atom.SetProp(prop_name, old_atom.GetProp(prop_name))
                new_atom.SetAtomMapNum(old_atom.GetAtomMapNum())
                new_atom.SetIsotope(old_atom.GetIsotope())
                new_atom.SetChiralTag(old_atom.GetChiralTag())
                new_atom.SetNoImplicit(old_atom.GetNoImplicit())
                new_atom.SetIsAromatic(old_atom.GetIsAromatic())
            for bond in kept_bonds:
                a1 = bond.GetBeginAtomIdx()
                a2 = bond.GetEndAtomIdx()
                new_bond = new_mol.GetBondBetweenAtoms(index_map[a1], index_map[a2])
                if new_bond is None:
                    continue
                new_bond.SetIsAromatic(bond.GetIsAromatic())
                new_bond.SetBondDir(bond.GetBondDir())
                new_bond.SetStereo(bond.GetStereo())
                stereo_atoms = bond.GetStereoAtoms()
                if stereo_atoms and len(stereo_atoms) == 2:
                    new_bond.SetStereoAtoms(
                        index_map[stereo_atoms[0]],
                        index_map[stereo_atoms[1]],
                    )
                for prop_name in bond.GetPropNames():
                    new_bond.SetProp(prop_name, bond.GetProp(prop_name))
        except Exception:
            pass
        return new_mol
    # Default path: remove atoms in-place (fast for smaller molecules)
    rw = Chem.RWMol(mol)
    rw.BeginBatchEdit()
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(int(idx))
    rw.CommitBatchEdit()
    return rw.GetMol()

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
    
    # Map bond order representations to RDKit bond types
    bond_order_mapping = {
        "SINGLE": Chem.BondType.SINGLE,
        "DOUBLE": Chem.BondType.DOUBLE,
        "TRIPLE": Chem.BondType.TRIPLE,
        "AROMATIC": Chem.BondType.AROMATIC,
        "ONEANDAHALF": Chem.BondType.AROMATIC  # Map ONEANDAHALF to AROMATIC bond type
    }
    
    # Add bonds to the molecule
    for (start_idx, end_idx), bond_order in zip(bond_indices, bond_orders):
        if isinstance(bond_order, Chem.BondType):
            bond_type = bond_order
        elif np.issubdtype(type(bond_order), np.number):
            if float(bond_order) == 1.0:
                bond_type = Chem.BondType.SINGLE
            elif float(bond_order) == 2.0:
                bond_type = Chem.BondType.DOUBLE
            elif float(bond_order) == 3.0:
                bond_type = Chem.BondType.TRIPLE
            elif float(bond_order) == 1.5:
                bond_type = Chem.BondType.AROMATIC
            else:
                bond_type = Chem.BondType.SINGLE
        else:
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

def proximity_bond(mol, correct_bond_orders = True, allow_hydrogen_gas = False, try_full = True, as_property_mol = False, verbose = "auto"):
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
        verbose (bool or "auto"): Defaults to "auto". If "auto", verbosity is enabled for large/complex bond inference.

    Returns:
        RDKit.Mol: Molecule with inferred bonds.

    """
    from rdkit.Chem import rdchem
    from rdkit.Chem import rdMolTransforms
    from rdkit.Chem import rdmolops
    from .util import bond_lengths, ideal_bond_lengths
    from .geometry import position
    from scipy.spatial import cKDTree

    verbose_mode = False if verbose == "auto" else bool(verbose)
    if verbose_mode:
        logger.info("Proximity bond inference started")
    # get the positions of the atoms
    positions = position(mol)
    periodic_table = Chem.GetPeriodicTable()
    elements = np.array([atom.GetSymbol() for atom in mol.GetAtoms()])
    max_distance = 2.5
    # create a kdtree for the atoms
    if verbose_mode: logger.info("Running approximate bonding")
    tree = cKDTree(positions)
    # query the tree for atoms that are closer than maximum sensible bond length
    d_matrix = tree.sparse_distance_matrix(tree, max_distance = max_distance, output_type = 'coo_matrix')
    # Pre-filter to only upper triangle to reduce work
    upper_triangle_mask = d_matrix.row < d_matrix.col
    row_filtered = d_matrix.row[upper_triangle_mask]
    col_filtered = d_matrix.col[upper_triangle_mask]
    data_filtered = d_matrix.data[upper_triangle_mask]

    bond_distances = np.array([bond_lengths[elements[i]][elements[j]] for i, j in zip(row_filtered, col_filtered)])
    
    # Filter by distance threshold
    cond = data_filtered < bond_distances[:,0] * 1.15
    close_atoms = np.vstack([row_filtered[cond], col_filtered[cond]]).T
    close_distances = data_filtered[cond]
    close_bond_distances = bond_distances[cond]
    
    mol = Chem.RWMol(mol)
    # Track existing bonds to avoid repeated lookups
    existing_bonds = set()
    if verbose_mode: logger.info("Assigning bond orders on approximate distances")
    bond_indices = []
    # create a bond for each pair of close atoms
    for idx, (i, j) in enumerate(close_atoms):
        if not allow_hydrogen_gas:
            if (elements[i] == "H") and (elements[j] == "H"):
                continue
        
        dist = close_distances[idx]
        bond_dist = close_bond_distances[idx]
        bond_key = (int(i), int(j))
        bond_indices.append(bond_key)
        
        # Check if bond exists (cache to avoid repeated lookups)
        if bond_key in existing_bonds:
            bond_exists = True
        else:
            bond = mol.GetBondBetweenAtoms(int(i), int(j))
            bond_exists = bond is not None
            if bond_exists:
                existing_bonds.add(bond_key)
        
        if dist < bond_dist[3]:
            if not bond_exists:
                mol.AddBond(int(i), int(j), rdchem.BondType.TRIPLE)
                existing_bonds.add(bond_key)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.TRIPLE)
        elif dist < bond_dist[2]:
            if not bond_exists:
                mol.AddBond(int(i), int(j), rdchem.BondType.DOUBLE)
                existing_bonds.add(bond_key)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.DOUBLE)
        elif dist < bond_dist[1]:
            bond_order = rdchem.BondType.SINGLE#ONEANDAHALF
            if not bond_exists:
                mol.AddBond(int(i), int(j), bond_order)
                existing_bonds.add(bond_key)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(bond_order)
        else:
            if not bond_exists:
                mol.AddBond(int(i), int(j), rdchem.BondType.SINGLE)
                existing_bonds.add(bond_key)
            else:
                mol.GetBondBetweenAtoms(int(i), int(j)).SetBondType(rdchem.BondType.SINGLE)

    bond_indices = np.asarray(bond_indices)
    # check for valency violations
    if verbose_mode: logger.info("Running SAT based bond order correction")
    if correct_bond_orders:
        try:
            rdmolops.FastFindRings(mol)
            ring_count = mol.GetRingInfo().NumRings()
        except Exception:
            ring_count = 0
        n_contacts = len(bond_indices)
        n_existing_bonds = mol.GetNumBonds()
        if ring_count > 0 and n_contacts >= n_existing_bonds:
            logger.warning(
                "proximity_bond: skipping Z3 bond-order correction for ring-containing "
                "structure with ambiguous proximity graph (%s contacts, %s bonds). "
                "Use explicit bonding or correct_bond_orders=False if this is intentional.",
                n_contacts,
                n_existing_bonds,
            )
        else:
            mol = correct_bond_orders_func(
                mol, try_full=try_full, verbose=verbose, bond_indices=bond_indices
            )

    mol.UpdatePropertyCache(strict=False) 

    return PropertyMol(mol) if as_property_mol else mol.GetMol()


def resolve_small_clashes(ligand, receptor, distance_threshold = 1.3, max_depth = 4):
    from .select import get_kinematic_graph
    from .geometry import rotate_around_bond
    rigid_components, adjacency_matrix, bonds_to_atoms = get_kinematic_graph(ligand)
    kinematic_weights = kinematic_chain_weights(ligand, rigid_components, return_node_weights=False)
    atom_to_component = {}
    for i, rigid_component in enumerate(rigid_components):
        for atom in rigid_component:
            atom_to_component[atom] = i

    from scipy.spatial import cKDTree
    receptor_coords = position(receptor)
    receptor_kd_tree = cKDTree(receptor_coords)

    ligand_positions = position(ligand)
    clashing_indices = np.argwhere(np.array(receptor_kd_tree.query_ball_point(ligand_positions, distance_threshold, return_length=True)) > 0).flatten()

    clashing_components = set()
    for clashing_index in clashing_indices:
        clashing_components.add(atom_to_component[clashing_index])

    # mark clashing components
    clashing_atoms = [ligand.GetAtomWithIdx(int(i)) for clashing_component in clashing_components for i in rigid_components[clashing_component]]
    clashing_positions = position(clashing_atoms)
    

    heavier_adjacency_matrix = np.zeros_like(adjacency_matrix)
    for i in range(kinematic_weights.shape[0]):
        for j in range(kinematic_weights.shape[1]):
            if i < j:
                continue
            if adjacency_matrix[i, j] == 1:
                if kinematic_weights[i, j] >= kinematic_weights[j, i]:
                    heavier_adjacency_matrix[j, i] = 1
                else:
                    heavier_adjacency_matrix[i, j] = 1

    ligand_adjacency_matrix = Chem.GetAdjacencyMatrix(ligand)
    rotated_mol = Chem.Mol(ligand)
    for clashing_component in clashing_components:
        next_matrix = np.array(heavier_adjacency_matrix)
        cur_depth = 0
        cur_other_component = next_matrix[clashing_component]
        path = []
        path.append(cur_other_component)
        while(cur_depth < max_depth) and (np.sum(adjacency_matrix[cur_other_component.astype(bool)]) < 3):
            next_matrix = next_matrix @ heavier_adjacency_matrix
            cur_depth += 1
            cur_other_component = next_matrix[clashing_component]
            path.append(cur_other_component)
        # we dont care for all other clashing atoms at the moment:
        ignored_mask = np.zeros(ligand.GetNumAtoms(), dtype=bool)
        ignored_mask[np.setdiff1d(clashing_indices, rigid_components[clashing_component])] = True
        # now we start sampling until we hit our component again
        def rotate_mol(mol, other_component_mask_index, angle):
            other_component_mask = path[other_component_mask_index]
            if other_component_mask_index == 0:
                # check for clashes
                mol_positions = position(mol)
                if np.any((np.array(receptor_kd_tree.query_ball_point(mol_positions, distance_threshold, return_length=True)) > 0)[~ignored_mask]):
                    return None
                else:
                    # check self clashes
                    distances = np.linalg.norm(mol_positions[:, None] - mol_positions[None, :], axis=-1)
                    np.fill_diagonal(distances, np.inf)
                    # mask adjacent atoms
                    distances[ligand_adjacency_matrix.astype(bool)] = np.inf
                    distances[ligand_adjacency_matrix.T.astype(bool)] = np.inf
                    if np.any(distances < 0.5):
                        print("detected self clash")
                        return None
                    return mol
            else:
                # try rotations until we hit our component:
                # next component is determined by moving heavier_adjacency_matrix backwards
                next_component_idx = other_component_mask_index - 1
                next_component = path[next_component_idx]
                bond_indices = bonds_to_atoms[(np.argwhere(other_component_mask)[0][0], np.argwhere(next_component)[0][0])]
                for angle in np.linspace(0, 2*np.pi, 36):
                    rot_mol = rotate_around_bond(mol, *bond_indices, angle)
                    rot_mol = rotate_mol(rot_mol, next_component_idx, angle)
                    if rot_mol is not None:
                        return rot_mol
                return None
        next_mol = rotate_mol(rotated_mol, len(path) -1, 0)
        if next_mol is None:
            print("no rotation found")
        else:
            print("rotation found")
            rotated_mol = next_mol

    return rotated_mol

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


def seed_align(aligned_mol, target_mol, aligned_seed_indices, target_seed_indices, up_to_depth = None):
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
                            rotatable_bond_to_bond_idx[(j, i)] = (neighbor.GetIdx(), atom_idx)
                            break
    # next we assign edge weights based on the number of atoms connected via them
    # for these we run a bottom up BFS through the rigid component tree
    aligned_atom_size_weights = kinematic_chain_weights(aligned_mol, aligned_chain)
    target_chain = get_kinematic_chain(target_mol)
    target_atom_size_weights = kinematic_chain_weights(target_mol, target_chain)

    def handle_bond(bond, parent_node, seen_bonds, seen_components, forbidden_angles, cur_depth = 0):
        print(f"Handling bond {rotatable_bond_to_bond_idx[bond]} at depth {cur_depth}")
        next_bonds = set()
        parent_transform = parent_node["transformation"]
        orig_seen_bonds = set(seen_bonds)
        orig_seen_components = set(seen_components)
        seen_bonds.add(bond)
        seen_bonds.add((bond[1], bond[0]))

        # here we only check the 1 since we only add bonds where 0 is already seen
        if bond[1] in seen_components:
            print(f"Undetected Cycle in molecule for bond {rotatable_bond_to_bond_idx[bond]}, This will probably lead to distorted poses.")
            return
        seen_components.add(bond[1])

        if True: # gather relevant atom indices (if just for lazy collapsing)
            affected_atom_indices = aligned_chain[bond[1]]
            considered_atom_indices = list(affected_atom_indices)
            for b in np.argwhere(rotatable_bonds[bond[1]] == 1):
                considered_atom_indices.extend([rotatable_bond_to_bond_idx[(bond[1], i)][1] for i in b])
            bond_start_atom_idx = rotatable_bond_to_bond_idx[bond][0]
            bond_end_atom_idx = rotatable_bond_to_bond_idx[bond][1]
        
        if True: # gather relevant coodinates (if just for lazy collapsing)
            base_position = aligned_positions[rotatable_bond_to_bond_idx[bond][0]]
            base_position = base_position @ parent_transform[:3, :3].T + parent_transform[:3, 3]
            end_bond_position = aligned_positions[rotatable_bond_to_bond_idx[bond][1]]
            end_bond_position = end_bond_position @ parent_transform[:3, :3].T + parent_transform[:3, 3]
            considered_positions = aligned_positions[considered_atom_indices]
            considered_positions = considered_positions @ parent_transform[:3, :3].T + parent_transform[:3, 3]
            center = np.mean(considered_positions, axis=0)
            radius = np.max(np.linalg.norm(considered_positions - center, axis=1))

        # get all atoms that are close to the affected atoms
        target_close_atom_indices = target_tree.query_ball_point(center, radius)
        
        if len(target_close_atom_indices) > 0: # get affected and target weights
            considered_weights = np.array([aligned_atom_size_weights.get(i, np.zeros(8)) for i in considered_atom_indices])
            target_weights = np.array([target_atom_size_weights.get(i, np.zeros(8)) for i in target_close_atom_indices])
            size_differences = np.sum(np.abs((considered_weights[:, None, :] - target_weights[None, :, :])), axis = -1)
            element_differences = np.array([[1 if aligned_mol.GetAtomWithIdx(i).GetSymbol() != target_mol.GetAtomWithIdx(j).GetSymbol() else 0 for j in target_close_atom_indices] for i in considered_atom_indices])
            # move both coordinate sets to origin
            target_positions = position([target_mol.GetAtomWithIdx(i) for i in target_close_atom_indices]) - base_position

        considered_positions = considered_positions - base_position
        bond_direction = (end_bond_position - base_position); bond_direction /= np.linalg.norm(bond_direction)
        
        if True: # set up collision tree
            seen_atom_indices = set(sum([list(aligned_chain[i]) for i in seen_components], []))
            self_other_positions = aligned_positions[[i for i in seen_atom_indices if i not in [bond_start_atom_idx, bond_end_atom_idx] + list(considered_atom_indices)]]
            self_other_positions = self_other_positions @ parent_transform[:3, :3].T + parent_transform[:3, 3]
            self_other_kd_tree = cKDTree(self_other_positions)
        
        if True: # next we try rotations around the bond
            best_score = 1e6
            best_match = None
            best_weights = None
            best_angle = None
            curr_collision_threshold = 1.5
            while (best_score > 1e5) and (curr_collision_threshold > 0.7):
                rotation_count = 36
                for i in range(rotation_count):
                    angle = i * np.pi / (rotation_count/2)
                    if np.any([np.abs(angle - forbidden_angle) < np.pi/(rotation_count/2) for forbidden_angle in forbidden_angles]):
                        continue
                    rotation = Rotation.from_rotvec(bond_direction * angle)
                    rotated_positions = rotation.apply(considered_positions)
                    # check for collisions 
                    if np.any(self_other_kd_tree.query_ball_point(rotated_positions + base_position, curr_collision_threshold)):
                       continue
                    if len(target_close_atom_indices) == 0:
                        best_score = -1
                        best_angle = angle
                        break
                    else:
                        # to determine the score, we assign atoms by distance
                        distance_metric = np.linalg.norm(rotated_positions[:, None] - target_positions[None, :], axis = -1)
                        # scale size_differences down to be equal to 1 A distance
                        total = distance_metric
                        best_matching = linear_sum_assignment(total)
                        weights = (element_differences + size_differences)[best_matching]
                        score = np.sum(weights)
                        #nodes with larger size and low size difference should be rewarded
                        weights = 1/(weights + 1) * weights
                        if score < best_score:
                            best_score = score
                            best_match = best_matching
                            best_weights = weights
                            best_angle = angle
                curr_collision_threshold -= 0.1
            if best_score > 1e5:
                return None
            if best_score == -1:
                # no target atoms, just rotate
                theta = best_angle
            else:
                if True: # compute analytical rotation
                    # get the best matching
                    base_positions = considered_positions[best_match[0]]
                    target_positions = target_positions[best_match[1]]
                    # get the optimal rotation
                    theta = find_optimal_rotation(bond_direction, base_positions, target_positions, np.ones_like(best_weights))
                
            if True: # compute transformation
                rotation = Rotation.from_rotvec(bond_direction * theta)

                transformation_to_origin = np.eye(4)
                transformation_to_origin[:3, 3] = -base_position
                transformation = np.eye(4)
                transformation[:3, :3] = rotation.as_matrix().T
                transformation_from_origin = np.eye(4)
                transformation_from_origin[:3, 3] = base_position
                transformation = transformation_from_origin @ transformation @ transformation_to_origin
        if cur_depth >= up_to_depth:
            transformation = parent_transform
        else:
            transformation = transformation @ parent_transform
        if True: # gather child transformations
            # consider all outgoing edges
            child_results = []
            for i in np.argwhere(rotatable_bonds[bond[1]] == 1):
                if i[0] in seen_components:
                    continue
                child_result = handle_bond(
                (bond[1], i[0]),
                {"transformation": transformation,
                    "affected_indices": affected_atom_indices,
                    "bond": bond,
                    "depth" : cur_depth
                }, seen_bonds, seen_components, [], cur_depth + 1)
                if child_result is not None:
                    child_results.append(child_result)
                else:
                    forbidden_angles = forbidden_angles + [best_angle]
                    print(f"Child bond failed, backtracking with forbidden angles: {len(forbidden_angles)} on depth {cur_depth}")
                    return handle_bond(bond, parent_node, orig_seen_bonds, orig_seen_components, forbidden_angles, cur_depth)
        return {"transformation": transformation, "affected_indices": affected_atom_indices, "bond": bond, "children": child_results, "depth" : cur_depth}

    # now we start the recursive tree building
    seen_components = set(aligned_seed_chain_indices)
    seen_bonds = set()
    next_bonds = set()
    for bond in np.argwhere(rotatable_bonds == 1):
        if bond[0] in aligned_seed_chain_indices:
            if bond[1] in aligned_seed_chain_indices:
                seen_bonds.add(bond)
                seen_bonds.add((bond[1], bond[0]))
            else:
                next_bonds.add((tuple(bond), 0, ()))
    children = []
    transformation_tree = {
        "transformation": np.eye(4),
        "affected_indices": [],
        "bond": None,
        "children": children,
        "used_thetas": [],
        "depth" : 0
    }
    for bond in next_bonds:
        children.append(handle_bond(bond[0], transformation_tree, seen_bonds, seen_components, []))

    # now traverse transformation tree and actually apply transformations
    def traverse_tree(node):
        transformation = node["transformation"]
        for child in node["children"]:
            traverse_tree(child)
        if node["bond"] is not None:
            # apply transformation to all affected atoms
            for atom_idx in node["affected_indices"]:
                aligned_positions[atom_idx] = aligned_positions[atom_idx] @ transformation[:3, :3].T + transformation[:3, 3]
        return
    traverse_tree(transformation_tree)
    return aligned_positions