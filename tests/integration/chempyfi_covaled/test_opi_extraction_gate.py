"""OPI extraction gate (JSON-backed fixture; no excel_bridge)."""

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.opi]

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "opi" / "covaled_ethane_mini"


@pytest.fixture(scope="module", autouse=True)
def _ensure_fixture():
    gen = FIXTURE / "generate_fixture.py"
    if not (FIXTURE / "super" / "orca.out.property.json").exists():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(gen)])


def test_opi_extraction_gate_passes():
    pytest.importorskip("opi")
    from chempyfi_covaled.equivalence.opi_extraction import compare_opi_extraction_fixture

    report = compare_opi_extraction_fixture(FIXTURE)
    assert report["new_path_led_sources"] == {"super": "opi", "ligand": "opi", "receptor": "opi"}
    assert report["opi_extraction_gate"] == "PASS"
    assert report["parity_gates"]["OPI_RAW_LED_PARITY"] == "PASS"
    assert report["parity_gates"]["OPI_FRAGMENT_ORDER_PARITY"] == "PASS"
    assert report["parity_gates"]["OPI_EXTRACTION_PARITY"] == "PASS"
    assert report["parity_gates"]["STEP7_MATRIX_PARITY"] == "PASS"
    assert report["parity_gates"]["SCALAR_INTERACTION_ENERGY_PARITY"] == "PASS"
