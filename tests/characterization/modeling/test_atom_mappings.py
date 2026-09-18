"""Document atom-index behavior of the public fragment_molecule API."""

import pytest

from chempyfi.orcautil.led_fragment import fragment_molecule
from chempyfi.rdutil.rw import proximity_bond
from rdkit import Chem


@pytest.mark.characterization
def test_cutout_indices_are_local_to_cutout_mol(build_mol):
    mol = build_mol(
        ["C"] * 8,
        [(i, i + 1, Chem.BondType.SINGLE) for i in range(7)],
        [(i * 1.54, 0.0, 0.0) for i in range(8)],
    )
    proximity_bond(mol)
    center = 4
    cutout, frags, _ = fragment_molecule(
        mol, center, filter_occluded_fragments=False, verbose=False
    )
    max_idx = max(i for f in frags for i in f)
    assert max_idx < cutout.GetNumAtoms()
    assert min(i for f in frags for i in f) >= 0


@pytest.mark.characterization
def test_public_api_does_not_return_source_index_map(build_mol):
    """index_map from create_molecule_cutout is not exposed by fragment_molecule."""
    mol = build_mol(
        ["C"] * 6,
        [(i, i + 1, Chem.BondType.SINGLE) for i in range(5)],
        [(i * 1.54, 0.0, 0.0) for i in range(6)],
    )
    proximity_bond(mol)
    result = fragment_molecule(mol, 3, filter_occluded_fragments=False)
    assert len(result) == 3
    cutout, frags, bonds = result
    assert cutout is not None
    assert frags is not None
    assert bonds is not None
