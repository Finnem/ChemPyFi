"""Bulk-solvation fp-LED redistribution must not exist in production chempyfi_covaled/orcautil code."""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SEARCH_ROOTS = [
    REPO / "module" / "chempyfi" / "orcautil",
    REPO / "module" / "chempyfi_covaled",
]
@pytest.mark.unit
def test_production_code_has_no_bulk_solvation_redistribution():
    hits = []
    for root in SEARCH_ROOTS:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "compute_bulk_solvation_contribution" in text:
                hits.append(str(path.relative_to(REPO)))
    assert hits == [], f"Forbidden bulk solvation references: {hits}"
