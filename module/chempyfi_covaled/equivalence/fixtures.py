"""Discover ORCA job paths in common ChemPyFi / CovaLED fixture layouts."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


def _first_orca_out(glob_root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(glob_root.glob(pattern))
    for p in matches:
        if p.is_file() and p.name == "orca.out":
            return p
    return None


def discover_fixture_orca_outputs(fixture_dir: Path) -> Dict[str, Path]:
    """
    Resolve supersystem, ligand, and receptor ``orca.out`` paths.

    Flat layout: ``super.out``, ``ligand.out``, ``receptor.out`` in ``fixture_dir``.

    Nested (5L4Q-style, matches ``extract_COVALED.py``):
      ``super/**/orca.out``, ``sub2/**/orca.out`` (ligand), ``sub1/**/orca.out`` (receptor/surrounding).
    """
    fixture_dir = Path(fixture_dir)
    flat = {
        "super": fixture_dir / "super.out",
        "ligand": fixture_dir / "ligand.out",
        "receptor": fixture_dir / "receptor.out",
    }
    if all(p.exists() for p in flat.values()):
        return flat

    super_out = _first_orca_out(fixture_dir, "super/**/orca.out")
    lig_out = _first_orca_out(fixture_dir, "sub2/**/orca.out")
    rec_out = _first_orca_out(fixture_dir, "sub1/**/orca.out")
    if super_out and lig_out and rec_out:
        return {"super": super_out, "ligand": lig_out, "receptor": rec_out}

    missing = [k for k, p in flat.items() if not p.exists()]
    raise FileNotFoundError(
        f"Could not discover orca.out triple under {fixture_dir}. "
        f"Missing flat files: {missing}. "
        "Expected flat super/ligand/receptor.out or nested super/, sub2/, sub1/."
    )


def property_json_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".property.json")


def standard_excel_for(out_path: Path) -> Path:
    return out_path.parent / "All_Standard_LED_matrices.xlsx"
