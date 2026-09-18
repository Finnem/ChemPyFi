from .Base import Receptor
from scipy.spatial import cKDTree
from rdkit.Chem import AllChem as Chem
from ..rdutil import position, check_occlusion
import numpy as np


class CationReceptor(Receptor):
    """ Class to represent a receptor molecule that represents Cations in a Receptor.

    Parameters
    ----------
    mol : RDKit Mol
        The receptor molecule.
    distance_threshold : float
        The distance threshold for hydrogen bond acceptor interactions.

    Attributes
    ----------  
    query_mols : list
        A list of query molecules that represent hydrogen bond acceptors.
    distance_threshold : float
        The distance threshold for hydrogen bond acceptor interactions.
    """

    def __init__(self, mol, distance_threshold = 6, occlusion_check = True):
        self.query_mols = [
            Chem.MolFromSmarts("[+]"), # generally all positively charged atoms
            # in a guanidinium all nitrogens can act as ionic centers
            Chem.MolFromSmarts("[NH](C)C(=[N;H2])([N;H2])"), # guanidinium
            Chem.MolFromSmarts("[N;H2]=C([NH](C))([N;H2])"), # guanidinium
            Chem.MolFromSmarts("[N;H2]C([NH](C))(=[N;H2])"), # guanidinium
            Chem.MolFromSmarts("[nH](C)C(=[N;H2])([N;H2])"), # guanidinium
            Chem.MolFromSmarts("[n;H2]=C([nH](C))([n;H2])"), # guanidinium
            Chem.MolFromSmarts("[n;H2]C([nH](C))(=[n;H2])"), # guanidinium
            Chem.MolFromSmarts("[NX4;H3]"), # guanidinium
            Chem.MolFromSmarts("[nX4;H3]"), # guanidinium
            
        ]
        self.blacklist_mols = [
            # Schrodinger produces weird histidines
            Chem.MolFromSmarts("[NH]1=[CH]-[N]-[CH]=C1"), # generally all negatively charged atoms
        ]
        self.occlusion_check = occlusion_check
        self.distance_threshold = distance_threshold
        super().__init__(mol)

    def _compute_query_positions_and_information(self, mol):
        """ Compute the positions of the atoms in the query molecule. 
        
        Parameters
        ----------
        mol : RDKit Mol
            The receptor molecule.

        """
        query_positions = []
        query_indices = []
        found_indices = set()

        blacklist_indices = set()
        for blacklist_mol in self.blacklist_mols:
            for match in mol.GetSubstructMatches(blacklist_mol):
                blacklist_indices.add(match[0])

        # occlusion should ignore neighboring hydrogens.
        occlusion_ignore = []
        mol_length = mol.GetNumAtoms()
        for query_mol in self.query_mols:
            for match in mol.GetSubstructMatches(query_mol):
                # make sure we don't repeat our findings
                if match[0] in found_indices:
                    continue
                if match[0] in blacklist_indices:
                    continue
                matched_atom = mol.GetAtomWithIdx(match[0])
                occlusion_ignore_mask = np.zeros(mol_length, dtype=bool)
                for neighbor in matched_atom.GetNeighbors():
                    occlusion_ignore_mask[neighbor.GetIdx()] = True
                occlusion_ignore.append(occlusion_ignore_mask)
                found_indices.add(match[0])
                matched_position = position(matched_atom)
                query_positions.append(matched_position)
                query_indices.append(match[0])
        self.information = {
            "occlusion_ignore_masks": np.array(occlusion_ignore)
        }
        self.query_positions = np.array(query_positions)
        self.query_indices = np.array(query_indices)


    def detect_interactions(self, ligand, distance_threshold = None):
        """ Ligand should here be a HydrogenBondAcceptorLigand object!

        """
        # detect if hydrogens should be ignored
        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        # Find all systems within the distance threshold
        if self.kdtree is None:
            return []
        if len(ligand.query_positions) == 0:
            return []
        donor_indices = self.kdtree.query_ball_point(ligand.query_positions, distance_threshold)
        acceptor_indices = np.concatenate([np.full(len(indices), i, dtype=int) for i, indices in enumerate(donor_indices)])
        donor_indices = np.concatenate(donor_indices).astype(int)

        # compute the angle between the hydrogen bond acceptor and the hydrogen bond donor
        differences = ligand.query_positions[acceptor_indices] - self.query_positions[donor_indices]
        distances = np.linalg.norm(differences, axis = 1)
        acceptor_directions = differences / distances[:, np.newaxis]
        acceptor_positions = ligand.query_positions[acceptor_indices]
        donor_positions = self.query_positions[donor_indices]

        if len(acceptor_positions) == 0:
            return []

        if self.occlusion_check:
            occlusion_mask = self.check_occlusion(ligand, donor_indices, acceptor_indices, distance_threshold)
            """
            # finally run occlusion check for all found interactions
            # determine potentially occluding atoms of ligand
            potential_occluders = []
            # assume all ligand atoms to potentially by occluding
            potential_occluders.extend(ligand.mol.GetAtoms())
            # also any atoms that are within the distance threshold of the interaction
            # lazily construct kd tree of own atoms
            if not hasattr(self, "kdtree_own"):
                self.kdtree_own = cKDTree(position(self.mol))
            # find all atoms within the distance threshold
            occluder_indices = self.kdtree_own.query_ball_point(np.concatenate([acceptor_positions, donor_positions]), distance_threshold)
            occluder_indices = np.concatenate(occluder_indices).astype(int)
            potential_occluders.extend([self.mol.GetAtomWithIdx(int(i)) for i in occluder_indices])

            # apply occlusion results
            occlusion_mask = ~check_occlusion(donor_positions, acceptor_positions, potential_occluders)
            """
            donor_indices = donor_indices[occlusion_mask]
            acceptor_indices = acceptor_indices[occlusion_mask]
            donor_positions = donor_positions[occlusion_mask]
            acceptor_positions = acceptor_positions[occlusion_mask]
            distances = distances[occlusion_mask]

        return list(zip(self.query_indices[donor_indices], ligand.query_indices[acceptor_indices], donor_positions, acceptor_positions, distances))


class CationLigand():
    """ Class to represent a receptor molecule that represents Cations in a Ligand.

    Parameters
    ----------
    mol : RDKit Mol
        The receptor molecule.

    Attributes
    ----------  
    query_mols : list
        A list of query molecules that represent hydrogen bond acceptors.
    distance_threshold : float
        The distance threshold for hydrogen bond acceptor interactions.
    """

    def __init__(self, mol):
        self.query_mols = [
            Chem.MolFromSmarts("[+]"), # generally all positively charged atoms
            # in a guanidinium all nitrogens can act as ionic centers
            Chem.MolFromSmarts("[NH](C)C(=[N;H2])([N;H2])"), # guanidinium
            Chem.MolFromSmarts("[N;H2]=C([NH](C))([N;H2])"), # guanidinium
            Chem.MolFromSmarts("[N;H2]C([NH](C))(=[N;H2])"), # guanidinium
            Chem.MolFromSmarts("[NX4;H3]"), # guanidinium
            
        ]
        self.blacklist_mols = [
            # Schrodinger produces weird histidines
            Chem.MolFromSmarts("[NH]1=[CH]-[N]-[CH]=C1"), # generally all negatively charged atoms
        ]
        self.ignore_Hs = False
        self.mol = mol
        self._compute_query_positions_and_information(mol)

    def _compute_query_positions_and_information(self, mol):
        """ Compute the positions of the atoms in the query molecule. 
        
        Parameters
        ----------
        mol : RDKit Mol
            The receptor molecule.

        """
        query_positions = []
        query_indices = []
        found_indices = set()

        blacklist_indices = set()
        for blacklist_mol in self.blacklist_mols:
            for match in mol.GetSubstructMatches(blacklist_mol):
                blacklist_indices.add(match[0])

        occlusion_ignore = []
        mol_length = mol.GetNumAtoms()
        for query_mol in self.query_mols:
            for match in mol.GetSubstructMatches(query_mol):
                # make sure we don't repeat our findings
                if match[0] in found_indices:
                    continue
                if match[0] in blacklist_indices:
                    continue
                matched_atom = mol.GetAtomWithIdx(match[0])
                occlusion_ignore_mask = np.zeros(mol_length, dtype=bool)
                for neighbor in matched_atom.GetNeighbors():
                    occlusion_ignore_mask[neighbor.GetIdx()] = True
                occlusion_ignore.append(occlusion_ignore_mask)
                found_indices.add(match[0])
                matched_position = position(matched_atom)
                query_positions.append(matched_position)
                query_indices.append(match[0])
        self.information = {
            "occlusion_ignore_masks": np.array(occlusion_ignore)
        }
        self.query_positions = np.array(query_positions)
        self.query_indices = np.array(query_indices)
