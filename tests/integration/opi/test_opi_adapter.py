"""OPI adapter integration (skipped without orca-pi)."""

import pytest

opi = pytest.importorskip("opi", reason="orca-pi not installed (pip install '.[opi]')")

from chempyfi.modeling import extract_local_environment
from chempyfi.orcautil.opi_adapter import intent_from_model, is_opi_available, to_opi_structure
from chempyfi.orcautil.orca_intent import (
    build_intent_from_legacy_args,
    intents_semantically_equal,
)
from chempyfi.rdutil.rw import proximity_bond
from rdkit import Chem
from rdkit.Geometry import Point3D


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


@pytest.mark.opi
@pytest.mark.integration
def test_opi_available():
    assert is_opi_available()


@pytest.mark.opi
@pytest.mark.integration
def test_handwritten_intent_matches_model_intent(tmp_path):
    mol = _chain(8)
    model = extract_local_environment(mol, 4, filter_occluded_fragments=False)
    header = "! DLPNO-CCSD(T) def2-SVP"
    mdci = "LED true"
    frags = [list(f.model_atom_indices) for f in model.fragments]
    legacy_intent = build_intent_from_legacy_args(
        model.molecule,
        frags,
        model.fragment_connection_bonds,
        header,
        mdci,
        center_fragment_last=True,
    )
    model_intent = intent_from_model(model, header, mdci, center_fragment_last=True)
    assert intents_semantically_equal(legacy_intent, model_intent)


@pytest.mark.opi
@pytest.mark.integration
def test_to_opi_structure_roundtrip_smoke():
    mol = _chain(6)
    model = extract_local_environment(mol, 3, filter_occluded_fragments=False)
    structure = to_opi_structure(model)
    assert structure is not None
    assert len(structure.atoms) == model.molecule.GetNumAtoms()
