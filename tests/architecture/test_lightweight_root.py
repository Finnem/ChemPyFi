"""Stage 6: ``import chempyfi`` must not eagerly load heavy integrations."""

import subprocess
import sys
from pathlib import Path

import pytest

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


def _run_snippet(body: str):
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
def test_import_chempyfi_only_exposes_version_in_process():
    for key in list(sys.modules):
        if key == "chempyfi" or key.startswith("chempyfi."):
            del sys.modules[key]
    import chempyfi

    assert chempyfi.__version__ == "0.1"
    assert "chempyfi.orcautil" not in sys.modules
    assert "chempyfi.pymolutil" not in sys.modules
    assert "pymol" not in sys.modules
    assert "pymolviz" not in sys.modules


@pytest.mark.unit
def test_import_chempyfi_subprocess_heavy_modules_not_loaded():
    _run_snippet(
        """
import sys
import chempyfi
assert chempyfi.__version__
for name in ("chempyfi.orcautil", "chempyfi.pymolutil", "pymol", "pymolviz"):
    assert name not in sys.modules
for name in sys.modules:
    assert "ledaw" not in name.lower()
assert "pandas" not in sys.modules
"""
    )


@pytest.mark.unit
def test_lazy_submodule_rdutil():
    _run_snippet(
        """
import sys
import chempyfi
rd = chempyfi.rdutil
assert "chempyfi.rdutil" in sys.modules
assert "chempyfi.orcautil" not in sys.modules
"""
    )


@pytest.mark.unit
def test_compat_from_chempyfi_import_check_occlusion():
    _run_snippet(
        """
import sys
import chempyfi
from chempyfi import check_occlusion
assert callable(check_occlusion)
assert "chempyfi.geometry" in sys.modules
assert "chempyfi.orcautil" not in sys.modules
"""
    )


@pytest.mark.unit
def test_compat_fragment_molecule_loads_orcautil_only_when_requested():
    _run_snippet(
        """
import sys
import chempyfi
assert hasattr(chempyfi, "fragment_molecule")
from chempyfi import fragment_molecule
assert callable(fragment_molecule)
assert "chempyfi.orcautil" in sys.modules or "chempyfi.modeling" in sys.modules
"""
    )
