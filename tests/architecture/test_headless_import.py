"""Regression tests: core ChemPyFi import without PyMOL/pymolviz."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_import_chempyfi_does_not_load_pymolviz():
    for key in list(sys.modules):
        if key == "chempyfi" or key.startswith("chempyfi."):
            del sys.modules[key]
    if "pymolviz" in sys.modules:
        pytest.skip("pymolviz already imported in this process")
    import chempyfi  # noqa: F401

    assert "pymolviz" not in sys.modules


@pytest.mark.unit
def test_import_chempyfi_subprocess_with_blocked_pymol_stack():
    script = """
import sys

class _BlockImports:
    BLOCK = frozenset({"pymol", "pymolviz", "PySide6"})

    def find_module(self, fullname, path=None):
        root = fullname.split(".", 1)[0]
        if root in self.BLOCK:
            return self

    def load_module(self, fullname):
        raise ImportError(f"blocked optional integration: {fullname}")

sys.meta_path.insert(0, _BlockImports())
import chempyfi
assert "fragment_molecule" in dir(chempyfi) or hasattr(chempyfi, "fragment_molecule")
"""
    env = {**dict(__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "module")}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
