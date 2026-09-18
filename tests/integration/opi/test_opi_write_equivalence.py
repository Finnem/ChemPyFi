"""OPI-written ORCA inputs match semantic intent (OPI-only writer)."""

from pathlib import Path

import pytest

pytest.importorskip("opi", reason="orca-pi not installed (pip install 'chempyfi-covaled[opi]')")

from chempyfi.modeling import extract_local_environment
from chempyfi_covaled.opi import (
    build_intent_from_inp_file,
    intent_from_model,
    intents_semantically_equal,
    write_fragment_input,
    write_orca_input_opi,
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
def test_write_fragment_input_matches_semantic_intent(tmp_path):
    mol = _chain(8)
    model = extract_local_environment(mol, 4, filter_occluded_fragments=False)
    header = "! TEST-HEADER"
    mdci = "LED true"
    frags = [list(f.model_atom_indices) for f in model.fragments]
    bonds = list(model.fragment_connection_bonds)
    intent = intent_from_model(model, header, mdci, center_fragment_last=True)
    inp_path = tmp_path / "opi.inp"
    write_fragment_input(mol, frags, bonds, header, mdci, str(inp_path), center_fragment_last=True)
    parsed = build_intent_from_inp_file(inp_path)
    assert intents_semantically_equal(intent, parsed)


@pytest.mark.opi
@pytest.mark.integration
def test_write_orca_input_opi_matches_semantic_intent(tmp_path):
    mol = _chain(8)
    model = extract_local_environment(mol, 4, filter_occluded_fragments=False)
    header = "! TEST-HEADER"
    mdci = "LED true"
    intent = intent_from_model(model, header, mdci, center_fragment_last=True)
    inp_path = tmp_path / "direct.inp"
    write_orca_input_opi(model, inp_path, header, mdci, center_fragment_last=True)
    assert inp_path.exists()
    parsed = build_intent_from_inp_file(inp_path)
    assert intents_semantically_equal(intent, parsed)
