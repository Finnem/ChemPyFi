"""Architectural tests for dependency isolation (desired vs current)."""

import ast
from pathlib import Path
from typing import Set

import pytest


def _module_top_level_imports(relative_path: str) -> Set[str]:
    root = Path(__file__).resolve().parents[2]
    source = (root / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.unit
def test_hydrogen_bond_acceptor_module_has_no_pymolviz():
    imports = _module_top_level_imports("module/chempyfi/interactions/HydrogenBondAcceptor.py")
    assert "pymolviz" not in imports


@pytest.mark.unit
def test_led_fragment_has_no_pymol_or_opi():
    imports = _module_top_level_imports("module/chempyfi/orcautil/led_fragment.py")
    assert "pymol" not in imports
    assert "opi" not in imports


@pytest.mark.unit
def test_hydrogen_bond_donor_module_has_no_pymolviz():
    imports = _module_top_level_imports("module/chempyfi/interactions/HydrogenBondDonor.py")
    assert "pymolviz" not in imports


@pytest.mark.unit
def test_interactions_init_reexports_donor_module():
    """Package __init__ star-imports HydrogenBondDonor (headless detection path)."""
    assert "HydrogenBondDonor" in open(
        Path(__file__).resolve().parents[2] / "module/chempyfi/interactions/__init__.py"
    ).read()


@pytest.mark.unit
def test_detection_module_defers_pymol_rendering():
    imports = _module_top_level_imports("module/chempyfi/interactions/Detection.py")
    assert "pymolviz" not in imports
    assert "pymol" not in imports

