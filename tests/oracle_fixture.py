"""Optional on-disk ORCA / CovaLED regression fixtures (not shipped in the repo)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Basename of the historical 5L4Q bundle (when kept outside the repository).
DEFAULT_5L4Q_DIRNAME = "relax_5L4Q_full_pure_obj01_entry_00001_conf_01"


def orca_led_fixture_dir() -> Optional[Path]:
    """Directory containing super/ligand/receptor ORCA outputs (flat or 5L4Q nested layout)."""
    raw = os.environ.get("CHEMPYFI_ORCA_LED_FIXTURE_DIR")
    if not raw:
        return None
    return Path(raw)


def relax_5l4q_fixture_dir() -> Optional[Path]:
    """5L4Q Excel/ORCA bundle; same env var as :func:`orca_led_fixture_dir`."""
    return orca_led_fixture_dir()
