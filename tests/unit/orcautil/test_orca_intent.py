"""Semantic ORCA intent (no OPI required)."""

import pytest
from rdkit.Geometry import Point3D

from chempyfi.modeling import extract_local_environment
from chempyfi.orcautil.orca_intent import build_intent_from_legacy_args, intents_semantically_equal
from chempyfi.rdutil.rw import proximity_bond
from rdkit import Chem


def _two_fragment_cutout():
    rw = Chem.RWMol()
    for _ in range(6):
        rw.AddAtom(Chem.Atom("C"))
    for i in range(5):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(6)
    for i in range(6):
        conf.SetAtomPosition(i, Point3D(i * 1.54, 0.0, 0.0))
    mol.AddConformer(conf)
    proximity_bond(mol)
    return extract_local_environment(mol, 2, filter_occluded_fragments=False)


@pytest.mark.unit
def test_model_and_legacy_intents_agree():
    model = _two_fragment_cutout()
    header = "! TEST"
    frags = [list(f.model_atom_indices) for f in model.fragments]
    a = build_intent_from_legacy_args(
        model.molecule, frags, model.fragment_connection_bonds, header, "", center_fragment_last=True
    )
    from chempyfi.orcautil.orca_intent import build_intent_from_model

    b = build_intent_from_model(
        model, header, "", model.fragment_connection_bonds, center_fragment_last=True
    )
    assert intents_semantically_equal(a, b)
