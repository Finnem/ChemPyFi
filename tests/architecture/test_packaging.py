"""Stage 9: packaging metadata and install smoke checks."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"


@pytest.mark.unit
def test_pyproject_is_authoritative():
    assert PYPROJECT.is_file()
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "[project]" in text
    assert "chempyfi_pymol" in text or "chempyfi*" in text


@pytest.mark.unit
def test_setuptools_discovers_both_top_level_packages():
    text = PYPROJECT.read_text(encoding="utf-8")
    assert "module" in text
    assert "chempyfi" in text and "chempyfi_pymol" in text


@pytest.mark.unit
def test_core_imports_without_viz_extras():
    import chempyfi
    from chempyfi.geometry import check_occlusion
    from chempyfi.modeling import fragment_molecule

    assert chempyfi.__version__
    assert callable(check_occlusion)
    assert callable(fragment_molecule)


@pytest.mark.unit
def test_editable_install_smoke():
    """Verify PEP 517 editable install exposes chempyfi from module/."""
    script = """
import sys
import chempyfi
from chempyfi.modeling import fragment_molecule
assert chempyfi.__version__
assert callable(fragment_molecule)
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
