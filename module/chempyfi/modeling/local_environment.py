"""Construct chemically meaningful local molecular models (local-environment extraction)."""

import logging

from rdkit.Chem import rdmolops

from ..fragmentation.bookkeeping import (
    compute_replacement_pairs,
    get_fragment_connection_bonds,
    reorder_fragments_consecutive,
)
from ..fragmentation.capping import create_molecule_cutout
from ..fragmentation.local_region import extend_connected_indices, get_surrounding_indices
from ..fragmentation.peptide import split_fragments_by_peptide_bonds
from ..geometry.fragment_filter import filter_fragments_by_occlusion
from ..rdutil import get_connected_component_indices
from .molecular_model import MolecularModel

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def extract_local_environment(
    mol,
    center_atom_index,
    verbose=False,
    filter_occluded_fragments=True,
    additional_indices=None,
) -> MolecularModel:
    """
    Build a capped local model around ``center_atom_index``.

    Assumes the molecule is already proximity bonded.
    """
    all_components = get_connected_component_indices(mol)
    if verbose:
        logger.info(f"Connected components: {len(all_components)}")
    center_component_indices = None
    for component_indices in all_components:
        if center_atom_index in component_indices:
            center_component_indices = component_indices
            break
    if verbose:
        logger.info(f"Found center component {center_component_indices}")
    if center_component_indices is None:
        raise ValueError(f"Center atom index {center_atom_index} not found in molecule")

    surrounding_indices = get_surrounding_indices(mol, center_component_indices)
    if additional_indices:
        surrounding_indices = list(set(surrounding_indices).union(set(additional_indices)))
    if verbose:
        logger.info(f"Surrounding indices: {len(surrounding_indices)}")

    all_used_components = []
    all_to_replace = []
    ring_info = None
    try:
        rdmolops.FastFindRings(mol)
        ring_info = mol.GetRingInfo()
    except Exception:
        ring_info = None

    for component_indices in all_components:
        current_selected = set(component_indices).intersection(surrounding_indices)
        current_selected, to_replace = extend_connected_indices(
            mol, current_selected, component_indices, ring_info=ring_info, max_ring_size=12
        )
        if len(current_selected) != 0:
            all_used_components.append(current_selected)
            if verbose:
                logger.info(f"Extended component {len(component_indices)} to {len(current_selected)}")
            all_to_replace.extend(to_replace)
    if verbose:
        logger.info(f"Extended components total: {len(all_used_components)}")

    split_fragments = split_fragments_by_peptide_bonds(all_used_components, mol, verbose=verbose)
    if verbose:
        logger.info(f"Split fragments: {len(split_fragments)}")

    if filter_occluded_fragments:
        split_fragments = filter_fragments_by_occlusion(
            mol,
            split_fragments,
            center_atom_index,
            additional_indices=additional_indices,
            verbose=verbose,
        )
        if verbose:
            logger.info(f"Occlusion filter: {len(split_fragments)}")

    kept_atoms = set()
    for frag in split_fragments:
        kept_atoms.update(frag)
    to_replace = compute_replacement_pairs(mol, split_fragments, kept_atoms)
    if verbose:
        logger.info(f"Replacement pairs: {len(to_replace)}")

    cutout_mol, index_map, added_hydrogens = create_molecule_cutout(
        mol, split_fragments, to_replace, verbose=verbose
    )
    if verbose:
        logger.info(f"Cutout mol atoms: {cutout_mol.GetNumAtoms()}")
    split_covalent_bonds = get_fragment_connection_bonds(mol, split_fragments, index_map)
    split_fragments = [
        sorted(index_map[idx] for idx in frag if idx in index_map)
        for frag in split_fragments
        if frag
    ]
    if added_hydrogens:
        atom_to_fragment = {
            atom_idx: frag_idx
            for frag_idx, frag in enumerate(split_fragments)
            for atom_idx in frag
        }
        for base_old_idx, h_new_idx in added_hydrogens:
            base_new_idx = index_map.get(base_old_idx)
            if base_new_idx is None:
                continue
            frag_idx = atom_to_fragment.get(base_new_idx)
            if frag_idx is None:
                continue
            split_fragments[frag_idx].append(h_new_idx)

    center_new_idx = index_map.get(center_atom_index)
    cutout_mol, split_fragments, split_covalent_bonds, renumber_map = reorder_fragments_consecutive(
        cutout_mol,
        split_fragments,
        split_covalent_bonds,
        center_new_idx,
        return_renumber_map=True,
    )

    return MolecularModel.from_pipeline(
        cutout_mol,
        split_fragments,
        split_covalent_bonds,
        cutout_source_to_precut=index_map,
        cap_source_heavy_to_precut_h=added_hydrogens,
        cut_bond_pairs=to_replace,
        renumber_precut_to_final=renumber_map,
    )


def fragment_molecule(
    mol,
    center_atom_index,
    verbose=False,
    filter_occluded_fragments=True,
    additional_indices=None,
):
    """
    Build a capped local cutout around ``center_atom_index`` for downstream QC models.

    Legacy API; prefer :func:`extract_local_environment` for explicit index mappings.
    """
    return extract_local_environment(
        mol,
        center_atom_index,
        verbose=verbose,
        filter_occluded_fragments=filter_occluded_fragments,
        additional_indices=additional_indices,
    ).to_legacy_tuple()
