"""Ionic interaction detection."""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.interactions.Anion import AnionReceptor
from chempyfi.interactions.Cation import CationLigand


def _charged_mol(center_symbol, charge, xyz):
    rw = Chem.RWMol()
    a = Chem.Atom(center_symbol)
    a.SetFormalCharge(charge)
    rw.AddAtom(a)
    mol = rw.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(*xyz))
    mol.AddConformer(conf)
    return mol


@pytest.mark.unit
def test_anion_cation_contact():
    anion = _charged_mol("Cl", -1, (0.0, 0.0, 0.0))
    cation = _charged_mol("Na", 1, (3.0, 0.0, 0.0))
    receptor = AnionReceptor(anion, occlusion_check=False)
    ligand = CationLigand(cation)
    hits = receptor.detect_interactions(ligand, distance_threshold=6.0)
    assert len(hits) >= 1
    assert hits[0][4] < 6.0


@pytest.mark.unit
def test_anion_cation_no_contact_when_far():
    anion = _charged_mol("Cl", -1, (0.0, 0.0, 0.0))
    cation = _charged_mol("Na", 1, (30.0, 0.0, 0.0))
    receptor = AnionReceptor(anion, occlusion_check=False)
    ligand = CationLigand(cation)
    hits = receptor.detect_interactions(ligand, distance_threshold=6.0)
    assert hits == []
