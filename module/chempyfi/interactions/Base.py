from scipy.spatial import cKDTree
from rdkit.Chem import AllChem as Chem
from ..rdutil import position, check_occlusion
import numpy as np

class Receptor():
    """ Abstract class to represent an interacting receptor molecule.

    Parameters
    ----------
    mol : RDKit Mol
        The receptor molecule.

    Attributes
    ----------
    mol : RDKit Mol
        The receptor molecule.
    information : dict
        Information about the query molecule.
    query_positions : np.ndarray
        The positions of the atoms in the query molecule.
    kdtree : cKDTree
        A KDTree of the query positions.    

    """
    def __init__(self, mol):
        self.mol = mol
        self.information = []
        self.query_positions = []
        self.query_indices = []

        self._compute_query_positions_and_information(mol)
        if any([self.query_positions is None]):
            self.kdtree = None
        else:
            self._construct_kd_tree()
        
    def _compute_query_positions_and_information(mol):
        """ Compute the positions of the atoms in the query molecule. """
        raise NotImplementedError("This method must be implemented in a subclass.")

    def _construct_kd_tree(self):
        """ Construct a KDTree from the query positions. """
        if len(self.query_positions.shape) == 1:
            self.query_positions = self.query_positions.reshape(-1, 3)
        self.kdtree = cKDTree(self.query_positions)

    def check_occlusion(self, ligand, this_indices, ligand_indices, distance_threshold):
        """ Check for occlusion of the receptor by the ligand.

        Parameters
        ----------
        ligand : RDKit Mol
            The ligand molecule.

        Returns
        -------
        bool
            True if the receptor is occluded by the ligand, False otherwise.

        """

        start_positions = self.query_positions[this_indices]
        end_positions = ligand.query_positions[ligand_indices]
        potential_occluders = []
        # assume all ligand atoms to potentially by occluding
        potential_occluders.extend(ligand.mol.GetAtoms())
        # also any atoms that are within the distance threshold of the interaction
        # lazily construct kd tree of own atoms
        if not hasattr(self, "kdtree_own"):
            self.kdtree_own = cKDTree(position(self.mol))
        # find all atoms within the distance threshold
        if len(start_positions) == 0:
            return []
        occluder_indices = self.kdtree_own.query_ball_point(start_positions, distance_threshold)
        occluder_indices = np.concatenate(occluder_indices).astype(int)
        potential_occluders.extend([self.mol.GetAtomWithIdx(int(i)) for i in occluder_indices])

        # check own occlusion_ignore_masks
        if self.information.get("occlusion_ignore_masks", None) is not None:
            this_occlusion_ignore_masks = self.information["occlusion_ignore_masks"][:, occluder_indices][this_indices]
            # pad by in front by ligand atoms
            this_occlusion_ignore_masks = np.pad(this_occlusion_ignore_masks, ((0, 0), (len(ligand.mol.GetAtoms()), 0)), mode='constant', constant_values=False)
        else:
            this_occlusion_ignore_masks = np.zeros((len(start_positions), len(ligand.mol.GetAtoms()) + len(occluder_indices)), dtype=bool)
        
        # check other occlusion_ignore_masks
        if ligand.information.get("occlusion_ignore_masks", None) is not None:
            other_occlusion_ignore_masks = ligand.information["occlusion_ignore_masks"][ligand_indices]
            # pad in back by this occluder atoms
            this_occlusion_ignore_masks |= np.pad(other_occlusion_ignore_masks, ((0, 0), (0, len(occluder_indices))), mode='constant', constant_values=False)


        # compute occlusion results
        occlusion_mask = check_occlusion(start_positions, end_positions, potential_occluders, return_occluders=True)
        # apply ignore mask
        
        occlusion_mask = occlusion_mask & ~this_occlusion_ignore_masks
        # collapse occlusion mask
        occlusion_mask = np.any(occlusion_mask, axis = 1)
        return ~occlusion_mask
