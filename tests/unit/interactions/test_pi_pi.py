"""Aromatic proximity (pi-stacking) detection."""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.interactions.PiPiInteractions import (
    AromaticProximityLigand,
    AromaticProximityReceptor,
)


def _benzene_at(origin, normal_offset=0.0):
    mol = Chem.MolFromSmiles("c1ccccc1")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, atom in enumerate(mol.GetAtoms()):
        conf.SetAtomPosition(
            atom.GetIdx(),
            Point3D(origin[0] + i * 0.1, origin[1] + normal_offset, origin[2]),
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol


@pytest.mark.unit
def test_stacked_rings_detected():
    receptor = AromaticProximityReceptor(_benzene_at((0.0, 0.0, 0.0)), occlusion_check=False)
    ligand = AromaticProximityLigand(_benzene_at((0.0, 3.8, 0.0)))
    hits = receptor.detect_interactions(ligand, distance_threshold=5.0)
    assert len(hits) >= 1


@pytest.mark.unit
def test_distant_rings_not_detected():
    receptor = AromaticProximityReceptor(_benzene_at((0.0, 0.0, 0.0)), occlusion_check=False)
    ligand = AromaticProximityLigand(_benzene_at((0.0, 20.0, 0.0)))
    hits = receptor.detect_interactions(ligand, distance_threshold=5.0)
    assert hits == []
