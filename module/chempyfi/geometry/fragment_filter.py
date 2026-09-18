"""Occlusion-based pruning of molecular fragments (model-building helper)."""

import logging

import numpy as np
from scipy.spatial import cKDTree

from .occlusion import check_occlusion

logger = logging.getLogger(__name__)


def filter_fragments_by_occlusion(
    mol, fragments, center_atom_index, additional_indices=None, verbose=False
):
    """Drop fragments fully occluded from the center fragment along all sight lines."""
    from ..rdutil.geometry import position

    central_frag = None
    for frag in fragments:
        if center_atom_index in frag:
            central_frag = frag
            break
    if central_frag is None:
        if verbose:
            logger.info("No central fragment found for occlusion filter")
        return fragments

    central_atoms = [mol.GetAtomWithIdx(int(i)) for i in central_frag]
    central_positions = position(central_atoms)
    tree = cKDTree(central_positions)
    all_frag_atoms = set()
    for frag in fragments:
        all_frag_atoms.update(frag)

    additional_set = set(additional_indices or [])
    kept = []
    removed = 0
    for frag in fragments:
        if frag is central_frag:
            kept.append(frag)
            continue
        if additional_set.intersection(frag):
            kept.append(frag)
            continue
        frag_atoms = [mol.GetAtomWithIdx(int(i)) for i in frag]
        frag_positions = position(frag_atoms)
        _, nearest_idxs = tree.query(frag_positions)
        to_atoms = [central_atoms[int(i)] for i in nearest_idxs]
        occluder_indices = all_frag_atoms - set(frag) - set(central_frag)
        if not occluder_indices:
            kept.append(frag)
            continue
        occluder_atoms = [mol.GetAtomWithIdx(int(i)) for i in occluder_indices]
        occluded = check_occlusion(frag_atoms, to_atoms, occluder_atoms, ignore_outside=True)
        if np.all(occluded):
            removed += 1
            if verbose:
                logger.info(f"Occlusion removed fragment of size {len(frag)}")
            continue
        kept.append(frag)
    if verbose:
        logger.info(f"Occlusion filter removed {removed} fragments")
    return kept
