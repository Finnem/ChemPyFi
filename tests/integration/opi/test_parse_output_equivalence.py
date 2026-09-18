"""ORCA output parsing: legacy INPUT echo vs OPI (when fixture available)."""

import os
from pathlib import Path

import pytest

from chempyfi.orcautil.led_extract import parse_orca_fragments
from chempyfi.orcautil.opi_adapter import extract_fragment_mapping_from_output, is_opi_available

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "orca" / "minimal_input_echo.out"


@pytest.mark.unit
def test_legacy_parser_on_stored_fixture():
    frags, coords = parse_orca_fragments(FIXTURE)
    assert len(frags) == len(coords) == 6


@pytest.mark.opi
@pytest.mark.integration
def test_opi_parse_matches_legacy_when_property_json_present():
    if not is_opi_available():
        pytest.skip("orca-pi not installed")
    prop_json = FIXTURE.with_suffix(FIXTURE.suffix + ".property.json")
    custom = os.environ.get("CHEMPYFI_ORCA_OUT_FIXTURE")
    out_path = Path(custom) if custom else FIXTURE
    if custom:
        prop_json = out_path.with_suffix(out_path.suffix + ".property.json")
    if not prop_json.exists():
        pytest.skip(
            "No .property.json for OPI geometry parse; set CHEMPYFI_ORCA_OUT_FIXTURE to a real ORCA .out pair"
        )
    legacy_frags, legacy_coords = parse_orca_fragments(out_path)
    opi_frags, opi_coords = extract_fragment_mapping_from_output(out_path, use_opi=True)
    assert legacy_frags == opi_frags
    for a, b in zip(legacy_coords, opi_coords):
        for u, v in zip(a, b):
            assert u == pytest.approx(v, abs=1e-4)
