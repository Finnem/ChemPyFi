"""OPI must not load during core ChemPyFi imports."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MODULE = str(REPO / "module")


@pytest.mark.unit
def test_import_chempyfi_does_not_load_opi():
    """Isolated subprocess so earlier ``-m opi`` tests cannot populate ``sys.modules``."""
    script = """
import sys
import chempyfi
assert "opi" not in sys.modules
"""
    env = {**dict(__import__("os").environ), "PYTHONPATH": MODULE}
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout


@pytest.mark.unit
def test_extract_local_environment_subprocess_no_opi():
    script = """
import sys
import chempyfi
from chempyfi.modeling import extract_local_environment
assert "opi" not in sys.modules
"""
    env = {**dict(__import__("os").environ), "PYTHONPATH": MODULE}
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr or r.stdout
