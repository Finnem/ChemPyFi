"""Index-mapping invariants for local-model fragmentation."""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.fragmentation.capping import create_molecule_cutout
from chempyfi.modeling.local_environment import fragment_molecule
from chempyfi.rdutil.rw import proximity_bond


def _chain(n):
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
def test_cutout_reorder_preserves_total_atom_count():
    mol = _chain(10)
    cutout, frags, _ = fragment_molecule(mol, 5, filter_occluded_fragments=False)
    assert sum(len(f) for f in frags) == cutout.GetNumAtoms()


@pytest.mark.characterization
def test_capping_index_map_maps_source_heavy_to_cutout():
    mol = _chain(6)
    frags = [[0, 1, 2], [3, 4, 5]]
    to_replace = [(2, 3)]
    cutout, index_map, added = create_molecule_cutout(mol, frags, to_replace)
    assert 2 in index_map
    assert len(added) == 1
    heavy_old, h_new = added[0]
    assert index_map[heavy_old] < cutout.GetNumAtoms()
    assert h_new == cutout.GetNumAtoms() - 1


@pytest.mark.characterization
def test_center_atom_maps_into_last_fragment_after_reorder():
    mol = _chain(12)
    center = 6
    cutout, frags, _ = fragment_molecule(mol, center, filter_occluded_fragments=False)
    center_frag = frags[-1]
    assert any(idx < cutout.GetNumAtoms() for idx in center_frag)
