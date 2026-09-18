"""Renderer-neutral interaction point specs (Stage 10 parity)."""

import pytest

from chempyfi_pymol.display import interaction_point_specs


@pytest.mark.unit
def test_interaction_point_specs_from_hbond_rows():
    interactions = {
        "hydrogen_bond_donor_interactions": [
            (0, 1, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 10.0),
        ],
        "hydrogen_bond_acceptor_interactions": [],
    }
    specs = interaction_point_specs(interactions)
    assert len(specs) == 2
    assert specs[0][1].endswith("_receptor")
    assert specs[1][1].endswith("_ligand")


@pytest.mark.unit
def test_chempyfi_pymol_root_import_is_lightweight():
    import sys

    for key in list(sys.modules):
        if key == "chempyfi_pymol" or key.startswith("chempyfi_pymol."):
            del sys.modules[key]
    import chempyfi_pymol  # noqa: F401

    assert "pymolviz" not in sys.modules
    assert "pymol" not in sys.modules
