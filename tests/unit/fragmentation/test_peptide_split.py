"""Unit tests for peptide-aware fragment splitting."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from chempyfi.orcautil.led_fragment import split_fragments_by_peptide_bonds
from chempyfi.rdutil.rw import proximity_bond


def _dipeptide_mol():
    mol = Chem.MolFromSmiles("NCC(=O)NCC(=O)O")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=0)
    mol.UpdatePropertyCache()
    proximity_bond(mol)
    return mol


@pytest.mark.unit
def test_peptide_split_increases_fragment_count():
    mol = _dipeptide_mol()
    component = list(range(mol.GetNumAtoms()))
    frags = split_fragments_by_peptide_bonds([component], mol)
    assert len(frags) >= 2
    assert sum(len(f) for f in frags) >= len(component)


@pytest.mark.unit
def test_small_organic_does_not_peptide_split(build_mol):
    mol = build_mol(
        ["C", "C", "C"],
        [(0, 1, Chem.BondType.SINGLE), (1, 2, Chem.BondType.SINGLE)],
        [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (3.0, 0.0, 0.0)],
    )
    proximity_bond(mol)
    frags = split_fragments_by_peptide_bonds([list(range(3))], mol)
    assert len(frags) == 1
    assert frags[0] == [0, 1, 2]
