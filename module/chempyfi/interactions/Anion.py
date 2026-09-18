from .Base import Receptor
from scipy.spatial import cKDTree
from rdkit.Chem import AllChem as Chem
from ..rdutil import position, check_occlusion
import numpy as np


class AnionReceptor(Receptor):
    """ Class to represent a receptor molecule that represents Anions in a Receptor.

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
            Chem.MolFromSmarts("[-]"), # generally all negatively charged atoms
            Chem.MolFromSmarts("[O]~*~[O-]"), # both oxygens in carboxylate
            Chem.MolFromSmarts("[OX1]=*~[OX1]"), # one oxygen in carboxylate
            Chem.MolFromSmarts("[OX1]-*~[OX1]"), # one oxygen in carboxylate
            Chem.MolFromSmarts("[OX1]~C-[OX2]"), # one oxygen in carboxylate, for some stupid reason rdkit always matches hydrogens at the carboxylate, even though if they are not part of the structure
            Chem.MolFromSmarts("[OX2]~C~[OX1]"), # one oxygen in carboxylate, for some stupid reason rdkit always matches hydrogens at the carboxylate, even though if they are not part of the structure
            
        ]
        self.blacklist_mols = [
            # Schrodinger produces weird histidines
            Chem.MolFromSmarts("[N]1-[CH]=C-[NH]=[CH]1"), 
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

        for query_mol in self.query_mols:
            for match in mol.GetSubstructMatches(query_mol):
                # make sure we don't repeat our findings
                if match[0] in found_indices:
                    continue
                if match[0] in blacklist_indices:
                    continue
                matched_atom = mol.GetAtomWithIdx(match[0])
                found_indices.add(match[0])
                matched_position = position(matched_atom)
                query_positions.append(matched_position)
                query_indices.append(match[0])
        self.information = {
        }
        self.query_positions = np.array(query_positions)
        self.query_indices = np.array(query_indices)


    def detect_interactions(self, ligand, distance_threshold = None):
        """ Ligand should here be a HydrogenBondDonorLigand object!

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
            
            donor_indices = donor_indices[occlusion_mask]
            acceptor_indices = acceptor_indices[occlusion_mask]
            donor_positions = donor_positions[occlusion_mask]
            acceptor_positions = acceptor_positions[occlusion_mask]
            distances = distances[occlusion_mask]

        return list(zip(self.query_indices[donor_indices], ligand.query_indices[acceptor_indices], donor_positions, acceptor_positions, distances))


class AnionLigand():
    """ Class to represent a receptor molecule that represents Anions in a Ligand.

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
            Chem.MolFromSmarts("[-]"), # generally all negatively charged atoms
            Chem.MolFromSmarts("[O]~*~[O-]"), # both oxygens in carboxylate
            Chem.MolFromSmarts("[OX1]~*~[OX1]"), # one oxygen in carboxylate
            Chem.MolFromSmarts("[OX1]=*~[OX1]"), # one oxygen in carboxylate
            Chem.MolFromSmarts("[OX1]-*~[OX1]"), # one oxygen in carboxylate
            
        ]
        self.blacklist_mols = [
            # Schrodinger produces weird histidines
            Chem.MolFromSmarts("[N]1-[CH]=C-[NH]=[CH]1"), 
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

        occlusion_masks = []
        for query_mol in self.query_mols:
            for match in mol.GetSubstructMatches(query_mol):
                # make sure we don't repeat our findings
                if match[0] in found_indices:
                    continue
                if match[0] in blacklist_indices:
                    continue
                matched_atom = mol.GetAtomWithIdx(match[0])
                found_indices.add(match[0])
                matched_position = position(matched_atom)
                query_positions.append(matched_position)
                query_indices.append(match[0])
        self.information = {
        }
        self.query_positions = np.array(query_positions)
        self.query_indices = np.array(query_indices)

