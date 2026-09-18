from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree

from ..rdutil import position


_PROTEIN_APOLAR_INCLUDE_SMARTS = [
    "[#6]",   # carbon
    "[#16]",  # sulfur
]

_PROTEIN_APOLAR_EXCLUDE_SMARTS = [
    "[CX3](=[OX1])",  # carbonyl carbons
    "[SX4](=[OX1])",  # sulfoxide/sulfone sulfur
    "[SX6](=[OX1])",  # sulfone/sulfate sulfur
]


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """Generate deterministic unit sphere points using a Fibonacci lattice."""
    if n_points <= 0:
        return np.zeros((0, 3), dtype=float)
    if n_points == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)

    indices = np.arange(n_points, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (indices / (n_points - 1.0))
    radius = np.sqrt(np.clip(1.0 - y * y, 0.0, 1.0))
    theta = phi * indices
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    points = np.stack([x, y, z], axis=1)
    return points


def _residue_key(atom: Chem.Atom) -> str:
    """Create a stable residue key for decomposition."""
    info = atom.GetPDBResidueInfo()
    if info is None:
        return f"UNK:{atom.GetIdx()}"
    chain_id = info.GetChainId().strip() or "_"
    resn = info.GetResidueName().strip() or "UNK"
    resi = str(info.GetResidueNumber())
    insertion = info.GetInsertionCode().strip()
    if insertion:
        return f"{chain_id}:{resn}:{resi}{insertion}"
    return f"{chain_id}:{resn}:{resi}"


class ApolarSurfaceReceptor:
    """
    Compute ligand apolar SASA burial by protein and by apolar protein atoms.

    Ligand apolar typing:
    - simple element rule: C/S/halogens and neutral atoms

    Protein apolar typing:
    - SMARTS include/exclude rules.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        probe_radius: float = 1.4,
        n_sphere_points: int = 960,
        pocket_cutoff: float = 6.0,
        epsilon: float = 1e-8,
        zero_denominator: str = "zero",
    ) -> None:
        self.mol = mol
        self.probe_radius = float(probe_radius)
        self.n_sphere_points = int(n_sphere_points)
        self.pocket_cutoff = float(pocket_cutoff)
        self.epsilon = float(epsilon)
        self.zero_denominator = zero_denominator

        self._periodic_table = Chem.GetPeriodicTable()
        self._unit_sphere_points = _fibonacci_sphere(self.n_sphere_points)
        self._protein_data = self._extract_atom_data(self.mol)
        self._protein_apolar_mask = self._protein_apolar_mask_smarts(self.mol)
        self._protein_residue_keys = [
            _residue_key(atom) for atom in self.mol.GetAtoms()
        ]
        self._protein_tree = (
            cKDTree(self._protein_data["positions"])
            if self._protein_data["positions"].shape[0] > 0
            else None
        )

    def _extract_atom_data(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return {
                "positions": np.zeros((0, 3), dtype=float),
                "radii": np.zeros((0,), dtype=float),
                "atomic_numbers": np.zeros((0,), dtype=int),
            }

        positions = np.asarray(position(mol), dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(-1, 3)
        atomic_numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
        radii = np.array(
            [self._periodic_table.GetRvdw(int(z)) for z in atomic_numbers],
            dtype=float,
        )
        return {
            "positions": positions,
            "radii": radii,
            "atomic_numbers": atomic_numbers,
        }

    @staticmethod
    def _ligand_apolar_mask_simple(ligand: Chem.Mol) -> np.ndarray:
        apolar_atomic_nums = {6, 16, 9, 17, 35, 53}
        mask = np.zeros((ligand.GetNumAtoms(),), dtype=bool)
        for atom in ligand.GetAtoms():
            is_apolar = (
                atom.GetAtomicNum() in apolar_atomic_nums
                and atom.GetFormalCharge() == 0
            )
            mask[atom.GetIdx()] = is_apolar
        return mask

    def _protein_apolar_mask_smarts(self, protein: Chem.Mol) -> np.ndarray:
        mask = np.zeros((protein.GetNumAtoms(),), dtype=bool)

        for smarts in _PROTEIN_APOLAR_INCLUDE_SMARTS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in protein.GetSubstructMatches(patt):
                for idx in match:
                    mask[idx] = True

        # remove charged atoms from apolar class
        for atom in protein.GetAtoms():
            if atom.GetFormalCharge() != 0:
                mask[atom.GetIdx()] = False

        for smarts in _PROTEIN_APOLAR_EXCLUDE_SMARTS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None:
                continue
            for match in protein.GetSubstructMatches(patt):
                for idx in match:
                    mask[idx] = False

        return mask

    def _candidate_indices_for_ligand_atom(
        self,
        center: np.ndarray,
        expanded_radius_i: float,
        tree: Optional[cKDTree],
        atom_radii: np.ndarray,
        margin: float = 1e-3,
    ) -> np.ndarray:
        if tree is None or atom_radii.shape[0] == 0:
            return np.zeros((0,), dtype=int)
        # center-distance cutoff: ri + rj + 2*probe + margin
        max_r = float(np.max(atom_radii))
        query_radius = expanded_radius_i + (max_r + self.probe_radius) + margin
        idxs = tree.query_ball_point(center, query_radius)
        if len(idxs) == 0:
            return np.zeros((0,), dtype=int)
        return np.asarray(idxs, dtype=int)

    @staticmethod
    def _occlusion_mask_for_points(
        points: np.ndarray,
        blocker_positions: np.ndarray,
        blocker_expanded_radii: np.ndarray,
        epsilon: float,
    ) -> np.ndarray:
        """
        Return point x blocker occlusion mask.
        point is occluded by blocker j if ||p-cj|| < (rj_expanded - epsilon).
        """
        if points.shape[0] == 0 or blocker_positions.shape[0] == 0:
            return np.zeros((points.shape[0], blocker_positions.shape[0]), dtype=bool)
        deltas = points[:, None, :] - blocker_positions[None, :, :]
        d2 = np.einsum("...i,...i->...", deltas, deltas)
        thresholds = np.clip(blocker_expanded_radii - epsilon, 0.0, None)
        return d2 < (thresholds[None, :] ** 2)

    def detect_interactions(self, ligand: Chem.Mol) -> Dict[str, object]:
        ligand_data = self._extract_atom_data(ligand)
        ligand_positions = ligand_data["positions"]
        ligand_radii = ligand_data["radii"]
        n_lig_atoms = ligand_positions.shape[0]
        lig_apolar_mask = self._ligand_apolar_mask_simple(ligand)
        lig_apolar_indices = np.where(lig_apolar_mask)[0]

        if n_lig_atoms == 0 or lig_apolar_indices.size == 0:
            frac = 0.0 if self.zero_denominator == "zero" else np.nan
            return {
                "lig_apolar_sasa_free": 0.0,
                "lig_apolar_sasa_bound": 0.0,
                "lig_apolar_buried_total": 0.0,
                "lig_apolar_buried_by_apolar_protein": 0.0,
                "fraction_buried_by_apolar_protein": frac,
                "per_ligand_atom": {},
                "per_protein_residue": {},
                "free_accessible_masks": {},
                "n_apolar_ligand_atoms": int(lig_apolar_indices.size),
                "n_points_used": int(self.n_sphere_points),
                "params": {
                    "probe_radius": self.probe_radius,
                    "n_sphere_points": self.n_sphere_points,
                    "pocket_cutoff": self.pocket_cutoff,
                    "epsilon": self.epsilon,
                },
            }

        lig_tree = cKDTree(ligand_positions)
        protein_positions = self._protein_data["positions"]
        protein_radii = self._protein_data["radii"]

        # optional pocket prefilter for protein candidates
        pocket_protein_mask = np.ones((protein_positions.shape[0],), dtype=bool)
        if protein_positions.shape[0] > 0 and self.pocket_cutoff > 0.0:
            lig_neighbor_tree = cKDTree(ligand_positions[lig_apolar_indices])
            if lig_neighbor_tree.n > 0:
                max_prot_r = float(np.max(protein_radii)) if protein_radii.size else 0.0
                search_r = self.pocket_cutoff + self.probe_radius + max_prot_r
                nearby = lig_neighbor_tree.query_ball_point(protein_positions, search_r)
                pocket_protein_mask = np.array([len(hit) > 0 for hit in nearby], dtype=bool)

        lig_apolar_sasa_free = 0.0
        lig_apolar_sasa_bound = 0.0
        lig_apolar_buried_by_apolar = 0.0

        per_ligand_atom = {
            int(idx): {
                "sasa_free": 0.0,
                "sasa_bound": 0.0,
                "buried_total": 0.0,
                "buried_by_apolar_protein": 0.0,
            }
            for idx in lig_apolar_indices
        }
        per_protein_residue = defaultdict(float)
        free_accessible_masks = {}

        for lig_idx in lig_apolar_indices:
            center = ligand_positions[lig_idx]
            ri_expanded = ligand_radii[lig_idx] + self.probe_radius
            points = center[None, :] + self._unit_sphere_points * ri_expanded
            area_per_point = 4.0 * np.pi * (ri_expanded ** 2) / float(self.n_sphere_points)

            # Step B: free ligand apolar SASA (occlusion by other ligand atoms)
            lig_candidate_idxs = self._candidate_indices_for_ligand_atom(
                center=center,
                expanded_radius_i=ri_expanded,
                tree=lig_tree,
                atom_radii=ligand_radii,
            )
            lig_candidate_idxs = lig_candidate_idxs[lig_candidate_idxs != lig_idx]
            if lig_candidate_idxs.size > 0:
                blocker_pos = ligand_positions[lig_candidate_idxs]
                blocker_expanded_r = ligand_radii[lig_candidate_idxs] + self.probe_radius
                lig_occluded_mask = self._occlusion_mask_for_points(
                    points=points,
                    blocker_positions=blocker_pos,
                    blocker_expanded_radii=blocker_expanded_r,
                    epsilon=self.epsilon,
                )
                free_mask = ~np.any(lig_occluded_mask, axis=1)
            else:
                free_mask = np.ones((points.shape[0],), dtype=bool)

            free_accessible_masks[int(lig_idx)] = free_mask
            n_free = int(np.count_nonzero(free_mask))
            sasa_free_atom = n_free * area_per_point
            lig_apolar_sasa_free += sasa_free_atom
            per_ligand_atom[int(lig_idx)]["sasa_free"] = float(sasa_free_atom)

            # Step C/E: bound SASA + apolar burial attribution on free points
            free_points = points[free_mask]
            if free_points.shape[0] == 0:
                continue

            prot_candidate_idxs = self._candidate_indices_for_ligand_atom(
                center=center,
                expanded_radius_i=ri_expanded,
                tree=self._protein_tree,
                atom_radii=protein_radii if protein_radii.size else np.zeros((0,)),
            )
            if prot_candidate_idxs.size > 0:
                prot_candidate_idxs = prot_candidate_idxs[pocket_protein_mask[prot_candidate_idxs]]

            if prot_candidate_idxs.size == 0:
                sasa_bound_atom = free_points.shape[0] * area_per_point
                lig_apolar_sasa_bound += sasa_bound_atom
                per_ligand_atom[int(lig_idx)]["sasa_bound"] = float(sasa_bound_atom)
                continue

            prot_pos = protein_positions[prot_candidate_idxs]
            prot_expanded_r = protein_radii[prot_candidate_idxs] + self.probe_radius
            prot_occluded_mask = self._occlusion_mask_for_points(
                points=free_points,
                blocker_positions=prot_pos,
                blocker_expanded_radii=prot_expanded_r,
                epsilon=self.epsilon,
            )
            blocked_by_any = np.any(prot_occluded_mask, axis=1)

            # Bound SASA counts points not blocked by protein.
            n_bound = int(np.count_nonzero(~blocked_by_any))
            sasa_bound_atom = n_bound * area_per_point
            lig_apolar_sasa_bound += sasa_bound_atom
            per_ligand_atom[int(lig_idx)]["sasa_bound"] = float(sasa_bound_atom)

            # Step E: buried by apolar protein (simple convention)
            # point counts if blocked by >=1 protein atom and any blocker is apolar
            if blocked_by_any.any():
                apolar_candidate_mask = self._protein_apolar_mask[prot_candidate_idxs]
                apolar_blocker_mask = prot_occluded_mask & apolar_candidate_mask[None, :]
                blocked_by_apolar = np.any(apolar_blocker_mask, axis=1)
                n_blocked_by_apolar = int(np.count_nonzero(blocked_by_apolar))
                buried_by_apolar_atom = n_blocked_by_apolar * area_per_point
                lig_apolar_buried_by_apolar += buried_by_apolar_atom
                per_ligand_atom[int(lig_idx)]["buried_by_apolar_protein"] = float(
                    buried_by_apolar_atom
                )

                # per-residue decomposition: equal split among unique apolar blocking residues
                for point_idx in np.where(blocked_by_apolar)[0]:
                    blocking_local_idxs = np.where(apolar_blocker_mask[point_idx])[0]
                    if blocking_local_idxs.size == 0:
                        continue
                    residue_keys = {
                        self._protein_residue_keys[int(prot_candidate_idxs[local_idx])]
                        for local_idx in blocking_local_idxs
                    }
                    if len(residue_keys) == 0:
                        continue
                    split = area_per_point / float(len(residue_keys))
                    for res_key in residue_keys:
                        per_protein_residue[res_key] += split

        # Step D: buried total
        lig_apolar_buried_total = lig_apolar_sasa_free - lig_apolar_sasa_bound
        for lig_idx, atom_entry in per_ligand_atom.items():
            atom_entry["buried_total"] = atom_entry["sasa_free"] - atom_entry["sasa_bound"]

        # Step F: normalization
        if lig_apolar_sasa_free <= self.epsilon:
            fraction = 0.0 if self.zero_denominator == "zero" else np.nan
        else:
            fraction = lig_apolar_buried_by_apolar / lig_apolar_sasa_free

        return {
            "lig_apolar_sasa_free": float(lig_apolar_sasa_free),
            "lig_apolar_sasa_bound": float(lig_apolar_sasa_bound),
            "lig_apolar_buried_total": float(lig_apolar_buried_total),
            "lig_apolar_buried_by_apolar_protein": float(lig_apolar_buried_by_apolar),
            "fraction_buried_by_apolar_protein": float(fraction)
            if not np.isnan(fraction)
            else np.nan,
            "per_ligand_atom": per_ligand_atom,
            "per_protein_residue": dict(per_protein_residue),
            "free_accessible_masks": free_accessible_masks,
            "n_apolar_ligand_atoms": int(lig_apolar_indices.size),
            "n_points_used": int(self.n_sphere_points),
            "params": {
                "probe_radius": self.probe_radius,
                "n_sphere_points": self.n_sphere_points,
                "pocket_cutoff": self.pocket_cutoff,
                "epsilon": self.epsilon,
            },
        }
