"""Parametric semantic intent equivalence (legacy vs MolecularModel)."""

import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.fragmentation.bookkeeping import reorder_fragments_consecutive
from chempyfi.fragmentation.capping import create_molecule_cutout
from chempyfi.modeling import extract_local_environment
from chempyfi.modeling.molecular_model import MolecularModel
from chempyfi.orcautil.orca_intent import (
    build_intent_from_legacy_args,
    build_intent_from_model,
    intents_semantically_equal,
)
from chempyfi.rdutil.rw import proximity_bond

pytest.importorskip("opi", reason="orca-pi not installed (pip install '.[opi]')")


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


def _disconnected_two_fragments():
    rw = Chem.RWMol()
    for _ in range(6):
        rw.AddAtom(Chem.Atom("C"))
    rw.AddBond(0, 1, Chem.BondType.SINGLE)
    rw.AddBond(1, 2, Chem.BondType.SINGLE)
    rw.AddBond(3, 4, Chem.BondType.SINGLE)
    rw.AddBond(4, 5, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(6)
    for i in range(3):
        conf.SetAtomPosition(i, Point3D(i * 1.54, 0.0, 0.0))
    for j in range(3):
        conf.SetAtomPosition(3 + j, Point3D(10.0 + j * 1.54, 5.0, 0.0))
    mol.AddConformer(conf)
    proximity_bond(mol)
    return extract_local_environment(mol, 1, filter_occluded_fragments=False)


def _capped_local_model():
    mol = _chain(6)
    source_frags = [[0, 1, 2], [3, 4, 5]]
    cut_bond = [(2, 3)]
    cutout, index_map, added = create_molecule_cutout(mol, source_frags, cut_bond)
    precut_frags = [sorted(index_map[i] for i in frag) for frag in source_frags]
    reordered, new_frags, bonds, renum = reorder_fragments_consecutive(
        cutout, precut_frags, [], center_atom_index=None, return_renumber_map=True
    )
    return MolecularModel.from_pipeline(
        reordered,
        new_frags,
        bonds,
        cutout_source_to_precut=index_map,
        cap_source_heavy_to_precut_h=added,
        cut_bond_pairs=cut_bond,
        renumber_precut_to_final=renum,
    )


@pytest.mark.opi
@pytest.mark.integration
@pytest.mark.parametrize(
    "model_factory,center_last",
    [
        (lambda: extract_local_environment(_chain(8), 4, filter_occluded_fragments=False), True),
        (lambda: extract_local_environment(_chain(12), 6, filter_occluded_fragments=False), True),
        (_disconnected_two_fragments, False),
        (_capped_local_model, False),
    ],
    ids=["covalent_chain", "multi_fragment_chain", "disconnected", "capped_cutout"],
)
def test_legacy_and_model_intents_match(model_factory, center_last):
    model = model_factory()
    header = "! DLPNO-CCSD(T) def2-SVP"
    mdci = "LED true\nPrintLevel 2"
    frags = [list(f.model_atom_indices) for f in model.fragments]
    legacy = build_intent_from_legacy_args(
        model.molecule,
        frags,
        model.fragment_connection_bonds,
        header,
        mdci,
        center_fragment_last=center_last,
    )
    from_model = build_intent_from_model(
        model,
        header,
        mdci,
        model.fragment_connection_bonds,
        center_fragment_last=center_last,
    )
    assert intents_semantically_equal(legacy, from_model)
