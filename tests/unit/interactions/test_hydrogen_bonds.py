"""Hydrogen-bond detection without importing HydrogenBondDonor (pymolviz leak)."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chempyfi.interactions.HydrogenBondAcceptor import (
    HydrogenBondAcceptorReceptor,
    HydrogenBondDonorLigand,
)


def _methanol_at(dx=0.0):
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(
            i, Chem.rdGeometry.Point3D(pos.x + dx, pos.y, pos.z)
        )
    mol.UpdatePropertyCache()
    return mol


@pytest.mark.unit
def test_methanol_acceptor_and_donor_sites_exist():
    mol = _methanol_at(0.0)
    receptor = HydrogenBondAcceptorReceptor(mol, occlusion_check=False)
    ligand = HydrogenBondDonorLigand(mol, ignore_Hs=True)
    assert len(receptor.query_positions) >= 1
    assert len(ligand.query_positions) >= 1


@pytest.mark.unit
def test_hbond_detected_between_close_methanol_molecules():
    acceptor = _methanol_at(0.0)
    donor = _methanol_at(2.7)
    receptor = HydrogenBondAcceptorReceptor(acceptor, occlusion_check=False)
    ligand = HydrogenBondDonorLigand(donor, ignore_Hs=True)
    hits = receptor.detect_interactions(ligand)
    assert len(hits) >= 1
    assert hits[0][4] < 30.0


@pytest.mark.unit
def test_hbond_absent_when_methanol_molecules_far_apart():
    acceptor = _methanol_at(0.0)
    donor = _methanol_at(20.0)
    receptor = HydrogenBondAcceptorReceptor(acceptor, occlusion_check=False)
    ligand = HydrogenBondDonorLigand(donor, ignore_Hs=True)
    hits = receptor.detect_interactions(ligand)
    assert hits == []
