"""CovaLED requires ORCA .property.json sidecar."""

from pathlib import Path

import pytest

from chempyfi_covaled.exceptions import MissingLEDDataError


@pytest.mark.unit
def test_extract_led_data_requires_property_json(tmp_path):
    pytest.importorskip("opi")
    out = tmp_path / "job.out"
    out.write_text("****ORCA OUTPUT FILE****\n")
    from chempyfi_covaled.opi_led import extract_led_data

    with pytest.raises(MissingLEDDataError, match="property.json"):
        extract_led_data(out)
