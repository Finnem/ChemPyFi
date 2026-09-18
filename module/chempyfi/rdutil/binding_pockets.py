from .select import get_small_molecules
from .geometry import position, check_occlusion, AtomKDTree
from scipy.spatial import cKDTree
import numpy as np
import logging
from collections import defaultdict

PROTEINOGENIC_AA = set(["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"])
class BindingPocket():
    def __init__(self, center, radius, atoms, protein, ignore_aa = True):
        self.center = center
        self.radius = radius
        self.atoms = atoms
        self.protein = protein

        self.waters = []
        self.solutes = []



        small_molecules = get_small_molecules(protein, min_atoms = 1)
        self.ligands = []
    
        atom_ids = set([atom.GetIdx() for atom in atoms])
        for small_molecule in small_molecules:
            if (ignore_aa or (ignore_aa is None)) and all(atom.GetPDBResidueInfo().GetResidueName() in PROTEINOGENIC_AA for atom in small_molecule):
                if ignore_aa is None: logging.warning("Skipping proteinogenic amino acid as small molecule. If you want to include it, please pass ignore_aa = False.")
                continue
            positions = position(small_molecule)
            if len(small_molecule) < 6:
                if sum([(atom.GetAtomicNum() not in [1,8]) for atom in small_molecule]) == 0:
                    if len([atom for atom in small_molecule if atom.GetIdx() in atom_ids]) > 0:
                        self.waters.append(small_molecule)
                else:
                    if len([atom for atom in small_molecule if atom.GetIdx() in atom_ids]) > 0:
                        self.solutes.append(small_molecule)
            else:
                if np.min(np.linalg.norm(positions - center, axis=1)) < radius:
                    self.ligands.append(small_molecule)
            
        self.pocket_atoms = [a for a in atoms if a.GetIdx() not in sum([[atom.GetIdx() for atom in ligand] for ligand in self.ligands + self.waters], [])]

    def get_grid(self, spacing):
        # Get the grid
        grid = np.arange(-self.radius, self.radius, spacing)
        grid = np.meshgrid(grid, grid, grid)
        grid = np.stack(grid, axis=-1)
        grid = grid + self.center
        grid = grid.reshape(-1, 3)
        return grid
    


    def get_accessible_points(self, spacing, around = 1, neighbor_count = 4, distance_threshold = 0.5):
        """Returns points that are accessible in the binding pocket.
        
        Args:
            spacing (float): Spacing of the grid.
            around (int): Defaults to 1. Number of grid points to check around each point.
            neighbor_count (int): Defaults to 4. Number of neighbors that need to be accessible for a point to be considered accessable.

        Returns:
            np.ndarray: Nx3 array of points that are accessible.
        """
        grid = self.get_grid(spacing)

        kdtree = AtomKDTree(self.pocket_atoms)
        collision_indices = kdtree.query_ball_point(grid, distance_threshold)
        unobstructed_indices = [i for i, indices in enumerate(collision_indices) if len(indices) == 0]

        # check if they lie within the sphere
        unobstructed_indices = [i for i in unobstructed_indices if np.linalg.norm(grid[i] - self.center) < self.radius]

        center_index = unobstructed_indices[np.argmin(np.linalg.norm(grid[unobstructed_indices] - self.center, axis=1))]

        # propagate close unobstructed points from center
        to_check = set(get_grid_neighbors(center_index, spacing, self.radius, range = around)).intersection(set(unobstructed_indices))
        accessable_indices = set(to_check)
        view_counter = defaultdict(int)
        while len(to_check) > 0:
            point_idx = to_check.pop()
            for neighbor_idx in get_grid_neighbors(point_idx, spacing, self.radius, around):
                if (neighbor_idx not in accessable_indices) and (neighbor_idx not in to_check) and (neighbor_idx in unobstructed_indices):
                    view_counter[neighbor_idx] += 1
                    if view_counter[neighbor_idx] > neighbor_count:
                        accessable_indices.add(neighbor_idx)
                        to_check.add(neighbor_idx)
        
        return grid[list(accessable_indices)]

def get_grid_neighbors(grid_idx, spacing, radius, range = 1):
    grid_dimensions = np.floor(np.array([radius * 2 / spacing] * 3)).astype(int) + 1
    grid_idx = np.array([grid_idx % grid_dimensions[0], (grid_idx // grid_dimensions[0]) % grid_dimensions[1], grid_idx // (grid_dimensions[0] * grid_dimensions[1])])
    grid_idx = grid_idx.reshape(-1, 1, 3)
    
    neighbor_offsets = np.arange(-range, range + 1)
    neighbor_offsets = np.meshgrid(neighbor_offsets, neighbor_offsets, neighbor_offsets)
    neighbor_offsets = np.stack(neighbor_offsets, axis=-1)
    neighbor_offsets = neighbor_offsets.reshape(1, -1, 3)
    grid_idx = grid_idx + neighbor_offsets
    grid_idx = grid_idx.reshape(-1, 3)
    grid_idx = grid_idx[(grid_idx >= 0).all(axis=1)]
    grid_idx = grid_idx[:, 0] + grid_idx[:, 1] * grid_dimensions[0] + grid_idx[:, 2] * grid_dimensions[0] * grid_dimensions[1]
    return grid_idx


def get_binding_pockets_by_ligand(protein, margin = 4.0, ignore_aa = None):
    small_molecules = get_small_molecules(protein)
    binding_pockets = []
    for small_molecule in small_molecules:
        if (ignore_aa or (ignore_aa is None)) and all(atom.GetPDBResidueInfo().GetResidueName() in PROTEINOGENIC_AA for atom in small_molecule):
            if ignore_aa is None: logging.warning("Skipping proteinogenic amino acid as small molecule. If you want to include it, please pass ignore_aa = False. To ignore it, pass ignore_aa = True.")
            continue
        # determine center of molecule
        atom_positions = np.array([position(atom) for atom in small_molecule])
        center = np.mean(atom_positions, axis=0)
        # determine radius of molecule
        radius = np.max(np.linalg.norm(atom_positions - center, axis=1))
        # determine atoms in the binding pocket
        atoms = []
        for atom in protein.GetAtoms():
            if np.linalg.norm(position(atom) - center) < radius + margin:
                atoms.append(atom)
        binding_pockets.append(BindingPocket(center, radius, atoms, protein, ignore_aa = ignore_aa))
    return binding_pockets
