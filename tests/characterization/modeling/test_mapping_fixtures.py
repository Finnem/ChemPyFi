"""Stage 8: richer source↔output mapping characterization."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.fragmentation.bookkeeping import reorder_fragments_consecutive
from chempyfi.fragmentation.capping import create_molecule_cutout
from chempyfi.fragmentation.peptide import split_fragments_by_peptide_bonds
from chempyfi.modeling.local_environment import fragment_molecule
from chempyfi.rdutil.rw import proximity_bond


def _chain(n, spacing=1.54):
    rw = Chem.RWMol()
    for _ in range(n):
        rw.AddAtom(Chem.Atom("C"))
    for i in range(n - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(i * spacing, 0.0, 0.0))
    mol.AddConformer(conf)
    proximity_bond(mol)
    return mol


def _ring_with_tail():
    """Six-membered ring + tail; explicit bonds only (avoid proximity_bond on rings — can be very slow)."""
    rw = Chem.RWMol()
    for _ in range(6):
        rw.AddAtom(Chem.Atom("C"))
    rw.AddAtom(Chem.Atom("C"))
    ring_bonds = [(i, (i + 1) % 6, Chem.BondType.SINGLE) for i in range(6)]
    for a, b, t in ring_bonds:
        rw.AddBond(a, b, t)
    rw.AddBond(0, 6, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(7)
    for i in range(6):
        angle = i * np.pi / 3
        conf.SetAtomPosition(i, Point3D(np.cos(angle), np.sin(angle), 0.0))
    conf.SetAtomPosition(6, Point3D(2.5, 0.0, 0.0))
    mol.AddConformer(conf)
    return mol


@pytest.mark.characterization
def test_disconnected_components_fragment_membership():
    mol = _chain(4)
    rw = Chem.RWMol(mol)
    rw.AddAtom(Chem.Atom("C"))
    rw.AddAtom(Chem.Atom("C"))
    rw.AddBond(4, 5, Chem.BondType.SINGLE)
    conf = rw.GetConformer()
    conf.SetAtomPosition(4, Point3D(10.0, 0.0, 0.0))
    conf.SetAtomPosition(5, Point3D(11.54, 0.0, 0.0))
    mol2 = rw.GetMol()
    proximity_bond(mol2)
    cutout, frags, _ = fragment_molecule(mol2, 1, filter_occluded_fragments=False)
    covered = sorted(i for f in frags for i in f)
    assert covered == list(range(cutout.GetNumAtoms()))


@pytest.mark.characterization
def test_reorder_fragments_consecutive_local_ranges():
    mol = _chain(8)
    frags = [[0, 1], [2, 3, 4], [5, 6, 7]]
    bonds = [(1, 2), (4, 5)]
    reordered, new_frags, new_bonds = reorder_fragments_consecutive(mol, frags, bonds, center_atom_index=4)
    assert reordered.GetNumAtoms() == mol.GetNumAtoms()
    assert sum(len(f) for f in new_frags) == reordered.GetNumAtoms()
    offset = 0
    for frag in new_frags:
        assert frag == list(range(offset, offset + len(frag)))
        offset += len(frag)
    assert new_frags[-1]  # center fragment moved last


@pytest.mark.characterization
def test_cutout_coordinate_preservation_via_index_map():
    mol = _chain(6)
    frags = [[0, 1, 2], [3, 4, 5]]
    cutout, index_map, caps = create_molecule_cutout(mol, frags, [(2, 3)])
    for src, dst in index_map.items():
        assert mol.GetConformer().GetAtomPosition(src).Distance(
            cutout.GetConformer().GetAtomPosition(dst)
        ) < 1e-6
    for heavy_src, h_dst in caps:
        assert index_map[heavy_src] < cutout.GetNumAtoms() - 1


@pytest.mark.characterization
def test_peptide_split_produces_multiple_fragments():
    smiles = "CC(=O)NC(CC(=O)O)C(=O)NC(C)C(=O)O"
    mol = Chem.MolFromSmiles(smiles)
    Chem.AllChem.EmbedMolecule(mol, randomSeed=1)
    proximity_bond(mol)
    components = [list(range(mol.GetNumAtoms()))]
    frags = split_fragments_by_peptide_bonds(components, mol)
    assert len(frags) >= 2


@pytest.mark.characterization
def test_occlusion_pruning_reduces_fragment_count():
    mol = _chain(20)
    cutout_all, frags_all, _ = fragment_molecule(mol, 10, filter_occluded_fragments=False)
    cutout_filt, frags_filt, _ = fragment_molecule(mol, 10, filter_occluded_fragments=True)
    assert len(frags_filt) <= len(frags_all)
    assert cutout_filt.GetNumAtoms() <= cutout_all.GetNumAtoms()


@pytest.mark.characterization
def test_ring_system_maps_into_cutout_with_local_indices():
    mol = _ring_with_tail()
    frags = [list(range(6)), [6]]
    cutout, index_map, _ = create_molecule_cutout(mol, frags, [(5, 6)])
    ring_local = [index_map[i] for i in range(6)]
    assert len(set(ring_local)) == 6
    assert max(ring_local) < cutout.GetNumAtoms() - 1  # last atom is cap H
    for src in range(6):
        assert mol.GetConformer().GetAtomPosition(src).Distance(
            cutout.GetConformer().GetAtomPosition(index_map[src])
        ) < 1e-6
