"""Smoke imports for documented public APIs (no heavy execution)."""

import pytest


@pytest.mark.unit
def test_chempyfi_modeling_smoke():
    from chempyfi.modeling import extract_local_environment, fragment_molecule

    assert callable(extract_local_environment)
    assert callable(fragment_molecule)


@pytest.mark.unit
def test_chempyfi_interactions_smoke():
    from chempyfi.interactions import detect_interactions

    assert callable(detect_interactions)


@pytest.mark.unit
def test_chempyfi_covaled_pipeline_smoke():
    from chempyfi_covaled.led_data import LEDData
    from chempyfi_covaled import compute_covaled, compute_fp_covaled
    from chempyfi_covaled.pipeline import analyze_output

    assert LEDData is not None
    assert callable(analyze_output)
    assert callable(compute_covaled)
    assert callable(compute_fp_covaled)


@pytest.mark.unit
def test_chempyfi_covaled_opi_write_requires_opi_or_clear_error():
    pytest.importorskip("opi")
    from chempyfi_covaled.opi import write_fragment_input

    assert callable(write_fragment_input)
