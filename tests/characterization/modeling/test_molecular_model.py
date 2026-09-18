"""MolecularModel API vs legacy fragment_molecule equivalence."""

import pytest
from rdkit.Geometry import Point3D

from chempyfi.modeling import extract_local_environment, fragment_molecule
from chempyfi.rdutil.rw import proximity_bond


def _chain(n):
    from rdkit import Chem

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
def test_legacy_tuple_matches_extract_local_environment():
    mol = _chain(10)
    center = 5
    legacy_mol, legacy_frags, legacy_bonds = fragment_molecule(
        mol, center, filter_occluded_fragments=False
    )
    model = extract_local_environment(mol, center, filter_occluded_fragments=False)
    model_mol, model_frags, model_bonds = model.to_legacy_tuple()
    assert legacy_frags == model_frags
    assert legacy_bonds == model_bonds
    assert legacy_mol.GetNumAtoms() == model_mol.GetNumAtoms()


@pytest.mark.characterization
def test_model_to_source_covers_heavy_atoms(build_mol):
    from rdkit import Chem

    mol = build_mol(
        ["C"] * 6,
        [(i, i + 1, Chem.BondType.SINGLE) for i in range(5)],
        [(i * 1.54, 0.0, 0.0) for i in range(6)],
    )
    proximity_bond(mol)
    model = extract_local_environment(mol, 3, filter_occluded_fragments=False)
    heavy_mapped = [s for s in model.model_to_source if s is not None]
    assert len(heavy_mapped) >= 4
    assert all(isinstance(x, int) for x in heavy_mapped)


@pytest.mark.characterization
def test_cap_atoms_have_none_source_mapping():
    from chempyfi.fragmentation.bookkeeping import reorder_fragments_consecutive
    from chempyfi.fragmentation.capping import create_molecule_cutout
    from chempyfi.modeling.molecular_model import MolecularModel

    mol = _chain(6)
    source_frags = [[0, 1, 2], [3, 4, 5]]
    cut_bond = [(2, 3)]
    cutout, index_map, added = create_molecule_cutout(mol, source_frags, cut_bond)
    precut_frags = [
        sorted(index_map[i] for i in frag) for frag in source_frags
    ]
    reordered, new_frags, bonds, renum = reorder_fragments_consecutive(
        cutout, precut_frags, [], center_atom_index=None, return_renumber_map=True
    )
    model = MolecularModel.from_pipeline(
        reordered,
        new_frags,
        bonds,
        cutout_source_to_precut=index_map,
        cap_source_heavy_to_precut_h=added,
        cut_bond_pairs=cut_bond,
        renumber_precut_to_final=renum,
    )
    assert any(s is None for s in model.model_to_source)
    assert model.boundaries
    assert any(b.cap_model_atom is not None for b in model.boundaries)


@pytest.mark.characterization
def test_fragment_membership_covers_all_model_atoms():
    mol = _chain(8)
    model = extract_local_environment(mol, 4, filter_occluded_fragments=False)
    covered = sorted(i for frag in model.fragments for i in frag.model_atom_indices)
    assert covered == list(range(model.molecule.GetNumAtoms()))
