"""Characterization tests for fragment_molecule (local-environment extraction)."""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.orcautil.led_fragment import fragment_molecule
from chempyfi.rdutil.rw import proximity_bond
from chempyfi.rdutil.select import get_connected_component_indices


def _alkane_chain_mol(n=10):
    rw = Chem.RWMol()
    for _ in range(n):
        rw.AddAtom(Chem.Atom("C"))
    for i in range(n - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(i * 1.54, 0.0, 0.0))
    mol.AddConformer(conf)
    proximity_bond(mol)
    return mol


@pytest.mark.characterization
def test_fragment_molecule_returns_cutout_and_fragments():
    mol = _alkane_chain_mol(12)
    center = 6
    cutout, frags, cov_bonds = fragment_molecule(
        mol, center, filter_occluded_fragments=False, verbose=False
    )
    assert cutout.GetNumAtoms() > 0
    assert len(frags) >= 1
    assert isinstance(cov_bonds, list)


@pytest.mark.characterization
def test_center_fragment_is_last():
    mol = _alkane_chain_mol(12)
    center = 6
    cutout, frags, _ = fragment_molecule(
        mol, center, filter_occluded_fragments=False, verbose=False
    )
    assert center in frags[-1]
    all_indices = sorted(i for f in frags for i in f)
    assert all_indices == list(range(cutout.GetNumAtoms()))


@pytest.mark.characterization
def test_single_connected_component_in_cutout():
    mol = _alkane_chain_mol(12)
    center = 6
    cutout, _, _ = fragment_molecule(
        mol, center, filter_occluded_fragments=False, verbose=False
    )
    components = get_connected_component_indices(cutout)
    assert len(components) == 1


@pytest.mark.characterization
def test_occlusion_filter_can_remove_fragments():
    mol = _alkane_chain_mol(12)
    center = 6
    _, frags_on, _ = fragment_molecule(
        mol, center, filter_occluded_fragments=True, verbose=False
    )
    _, frags_off, _ = fragment_molecule(
        mol, center, filter_occluded_fragments=False, verbose=False
    )
    assert len(frags_on) <= len(frags_off)
