"""OPI → LEDData extraction."""

from pathlib import Path

import pytest

pytest.importorskip("opi")

from chempyfi_covaled.led_data import KJMOL_PER_HARTREE
from chempyfi_covaled.opi_led import extract_led_data

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "opi" / "minimal_led.out"


@pytest.mark.opi
@pytest.mark.integration
def test_extract_led_data_fragment_count_and_units():
    led = extract_led_data(FIXTURE)
    assert led.n_fragments == 2
    assert led.fragment_ids == (1, 2)
    assert led.energy_unit == "kJ/mol"
    assert led.total.shape == (2, 2)
    assert led.total[0, 1] == pytest.approx(0.003807 * KJMOL_PER_HARTREE)


@pytest.mark.opi
@pytest.mark.integration
def test_extract_led_components_present():
    led = extract_led_data(FIXTURE)
    assert led.components.reference is not None
    assert led.components.correlation is not None
    assert led.components.electrostatics is not None
