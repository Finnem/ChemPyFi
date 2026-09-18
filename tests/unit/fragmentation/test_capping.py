"""Unit tests for boundary hydrogen capping."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.orcautil.led_fragment import create_molecule_cutout
from chempyfi.rdutil.geometry import position


def _two_fragment_mol():
    rw = Chem.RWMol()
    for sym in ["C", "C", "C", "C"]:
        rw.AddAtom(Chem.Atom(sym))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(2, 3, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(4)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    conf.SetAtomPosition(1, Point3D(1.5, 0.0, 0.0))
    conf.SetAtomPosition(2, Point3D(4.0, 0.0, 0.0))
    conf.SetAtomPosition(3, Point3D(5.5, 0.0, 0.0))
    mol.AddConformer(conf)
    return mol


@pytest.mark.unit
def test_capping_adds_hydrogen_at_cut():
    mol = _two_fragment_mol()
    frags = [[0, 1], [2, 3]]
    to_replace = [(1, 2)]
    cutout, index_map, added = create_molecule_cutout(mol, frags, to_replace)
    assert cutout.GetNumAtoms() == 5
    assert len(added) == 1
    h_idx = added[0][1]
    assert cutout.GetAtomWithIdx(h_idx).GetAtomicNum() == 1
    assert cutout.GetBondBetweenAtoms(index_map[1], h_idx) is not None


@pytest.mark.unit
def test_cap_distance_is_approximately_1_1_angstrom():
    mol = _two_fragment_mol()
    frags = [[0, 1], [2, 3]]
    to_replace = [(1, 2)]
    cutout, index_map, added = create_molecule_cutout(mol, frags, to_replace)
    heavy = index_map[1]
    h_idx = added[0][1]
    dist = np.linalg.norm(
        np.array(cutout.GetConformer().GetAtomPosition(heavy))
        - np.array(cutout.GetConformer().GetAtomPosition(h_idx))
    )
    assert dist == pytest.approx(1.1, abs=0.05)
