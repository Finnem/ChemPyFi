"""Shared pytest fixtures for the official ChemPyFi test suite."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D


@pytest.fixture
def periodic_table():
    return Chem.GetPeriodicTable()


def build_molecule(symbols, bonds, positions, aromatic_atoms=()):
    """Build a molecule with explicit hydrogens and one conformer."""
    aromatic_atoms = set(aromatic_atoms)
    mol = Chem.RWMol()
    for i, symbol in enumerate(symbols):
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(i + 1)
        atom.SetIsAromatic(i in aromatic_atoms)
        mol.AddAtom(atom)
    for begin, end, bond_type in bonds:
        mol.AddBond(begin, end, bond_type)

    mol = mol.GetMol()
    conformer = Chem.Conformer(len(symbols))
    for i, point in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*(float(value) for value in point)))
    mol.AddConformer(conformer)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


@pytest.fixture
def build_mol():
    return build_molecule
