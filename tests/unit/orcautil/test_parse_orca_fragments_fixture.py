"""Legacy ORCA INPUT FILE echo parser (no OPI)."""

from pathlib import Path

import pytest

from chempyfi.orcautil.led_extract import parse_orca_fragments

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "orca" / "minimal_input_echo.out"


@pytest.mark.unit
def test_parse_minimal_input_echo_fixture():
    frags, coords = parse_orca_fragments(FIXTURE)
    assert frags == [0, 0, 0, 1, 1, 1]
    assert len(coords) == 6
    assert coords[1][0] == pytest.approx(1.54)
