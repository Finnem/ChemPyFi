"""Stage 11: proximity_bond must not hang on ring+tail fixture."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = str(REPO_ROOT / "module")


def _ring_with_tail_mol():
    rw = Chem.RWMol()
    for _ in range(6):
        rw.AddAtom(Chem.Atom("C"))
    rw.AddAtom(Chem.Atom("C"))
    for i in range(6):
        rw.AddBond(i, (i + 1) % 6, Chem.BondType.SINGLE)
    rw.AddBond(0, 6, Chem.BondType.SINGLE)
    mol = rw.GetMol()
    conf = Chem.Conformer(7)
    for i in range(6):
        angle = i * np.pi / 3
        conf.SetAtomPosition(i, Point3D(np.cos(angle), np.sin(angle), 0.0))
    conf.SetAtomPosition(6, Point3D(2.5, 0.0, 0.0))
    mol.AddConformer(conf)
    return mol


@pytest.mark.unit
def test_ring_with_tail_proximity_bond_completes_in_subprocess():
    script = """
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from chempyfi.rdutil.rw import proximity_bond

rw = Chem.RWMol()
for _ in range(6):
    rw.AddAtom(Chem.Atom("C"))
rw.AddAtom(Chem.Atom("C"))
for i in range(6):
    rw.AddBond(i, (i + 1) % 6, Chem.BondType.SINGLE)
rw.AddBond(0, 6, Chem.BondType.SINGLE)
mol = rw.GetMol()
conf = Chem.Conformer(7)
for i in range(6):
    angle = i * np.pi / 3
    conf.SetAtomPosition(i, Point3D(np.cos(angle), np.sin(angle), 0.0))
conf.SetAtomPosition(6, Point3D(2.5, 0.0, 0.0))
mol.AddConformer(conf)
proximity_bond(mol)
print("ok")
"""
    env = {**dict(__import__("os").environ), "PYTHONPATH": MODULE_PATH}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "ok" in result.stdout
