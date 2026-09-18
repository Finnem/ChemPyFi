"""Unit tests for rdutil.geometry.check_occlusion."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.rdutil.geometry import check_occlusion


def _atom(symbol, xyz):
    mol = Chem.RWMol()
    a = Chem.Atom(symbol)
    mol.AddAtom(a)
    mol = mol.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol.GetAtomWithIdx(0)


@pytest.mark.unit
def test_occlusion_clear_line_of_sight():
    from_atoms = np.array([[0.0, 0.0, 0.0]])
    to_atoms = np.array([[5.0, 0.0, 0.0]])
    # Occluder far off the segment
    occluders = np.array([[2.5, 5.0, 0.0]])
    assert not check_occlusion(from_atoms, to_atoms, occluders)[0]


@pytest.mark.unit
def test_occlusion_blocked_by_sphere_on_segment():
    from_atoms = np.array([[0.0, 0.0, 0.0]])
    to_atoms = np.array([[5.0, 0.0, 0.0]])
    occluders = np.array([[2.5, 0.0, 0.0]])
    assert check_occlusion(from_atoms, to_atoms, occluders, default_radius=1.5)[0]


@pytest.mark.unit
def test_occlusion_on_segment_blocked_with_both_ignore_modes():
    from_atoms = np.array([[0.0, 0.0, 0.0]])
    to_atoms = np.array([[5.0, 0.0, 0.0]])
    occluders = np.array([[2.5, 0.0, 0.0]])
    assert check_occlusion(
        from_atoms, to_atoms, occluders, ignore_outside=True, default_radius=1.5
    )[0]
    assert check_occlusion(
        from_atoms, to_atoms, occluders, ignore_outside=False, default_radius=1.5
    )[0]


@pytest.mark.unit
def test_occlusion_rdkit_atoms():
    a = _atom("C", (0.0, 0.0, 0.0))
    b = _atom("C", (4.0, 0.0, 0.0))
    oc = _atom("O", (2.0, 0.0, 0.0))
    assert check_occlusion([a], [b], [oc])[0]


@pytest.mark.unit
def test_occlusion_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        check_occlusion(np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((1, 3)))


@pytest.mark.unit
def test_occlusion_empty_from_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        check_occlusion(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((1, 3)))
