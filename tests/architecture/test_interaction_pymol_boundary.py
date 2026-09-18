"""Stage 7: detection vs PyMOL rendering boundary."""

import ast
import subprocess
import sys
from pathlib import Path
from typing import Set

import pytest
from rdkit import Chem


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

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = str(REPO_ROOT / "module")

_BLOCKER = """
import sys

class _BlockImports:
    BLOCK = frozenset({"pymol", "pymolviz", "PySide6"})

    def find_module(self, fullname, path=None):
        root = fullname.split(".", 1)[0]
        if root in self.BLOCK:
            return self

    def load_module(self, fullname):
        raise ImportError(f"blocked: {fullname}")

sys.meta_path.insert(0, _BlockImports())
"""


def _run(body: str):
    env = {**dict(__import__("os").environ), "PYTHONPATH": MODULE_PATH}
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKER + body],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.unit
def test_import_interactions_does_not_load_pymol_stack():
    _run(
        """
import sys
import chempyfi.interactions
assert "pymol" not in sys.modules
assert "pymolviz" not in sys.modules
"""
    )


@pytest.mark.unit
def test_detect_interactions_with_pymol_blocked():
    _run(
        """
from rdkit import Chem
from rdkit.Geometry import Point3D
from chempyfi.interactions import detect_interactions
from chempyfi.rdutil.rw import proximity_bond

def chain(n):
    rw = Chem.RWMol()
    for _ in range(n):
        rw.AddAtom(Chem.Atom("C"))
    for i in range(n - 1):
        rw.AddBond(i, i + 1, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(i * 1.5, 0.0, 0.0))
    mol.AddConformer(conf)
    proximity_bond(mol)
    return mol

receptor = chain(4)
ligand = chain(2)
result = detect_interactions(receptor, ligand)
assert isinstance(result, dict)
assert "hydrogen_bond_donor_interactions" in result
"""
    )


@pytest.mark.unit
def test_detection_module_has_no_top_level_pymolviz():
    imports = _module_top_level_imports("module/chempyfi/interactions/Detection.py")
    assert "pymolviz" not in imports
    assert "chempyfi_pymol" not in imports


def _methanol_at(dx=0.0):
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    Chem.AllChem.EmbedMolecule(mol, randomSeed=0)
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(pos.x + dx, pos.y, pos.z))
    mol.UpdatePropertyCache()
    return mol


@pytest.mark.unit
def test_interaction_result_has_display_geometry():
    """When interactions exist, tuples include 3D endpoints for pymolviz adapters."""
    from chempyfi.interactions.HydrogenBondAcceptor import (
        HydrogenBondAcceptorReceptor,
        HydrogenBondDonorLigand,
    )

    acceptor = _methanol_at(0.0)
    donor = _methanol_at(2.7)
    receptor = HydrogenBondAcceptorReceptor(acceptor, occlusion_check=False)
    ligand = HydrogenBondDonorLigand(donor, ignore_Hs=True)
    hits = receptor.detect_interactions(ligand)
    assert len(hits) >= 1
    for row in hits:
        assert len(row) >= 4
        assert row[2] is not None and row[3] is not None


@pytest.mark.unit
def test_chempyfi_pymol_import_without_pymol_until_display():
    _run(
        """
import sys
import chempyfi_pymol
assert "pymol" not in sys.modules
assert "pymolviz" not in sys.modules
from chempyfi_pymol.display import _merge_self_lists
merged = _merge_self_lists({"hydrogen_bond_donor_interactions": [(0, 1, (0,0,0), (1,1,1))]}, None, "hydrogen_bond_donor_interactions")
assert len(merged) == 1
"""
    )


@pytest.mark.unit
def test_interactions_create_display_lazy_imports_pymol_package(monkeypatch):
    """Compat wrapper delegates to chempyfi_pymol only when called."""
    calls = []

    class FakeVisual:
        def write(self, path):
            calls.append(("write", path))

    def fake_create(*args, **kwargs):
        calls.append("create")
        return FakeVisual()

    import chempyfi_pymol.display as display_mod

    monkeypatch.setattr(display_mod, "create_interaction_display", fake_create)
    from chempyfi.interactions import Interactions

    receptor = _methanol_at(0.0)
    ligand = _methanol_at(2.7)
    Interactions(receptor).create_interaction_display(ligand, prefix="t")
    assert calls == ["create"]
