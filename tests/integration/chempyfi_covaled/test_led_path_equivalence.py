"""Optional local oracle: legacy LEDAW vs OPI gas-phase CovaLED."""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.chempyfi_covaled_oracle]


def _fixture_dir():
    raw = os.environ.get("CHEMPYFI_ORCA_LED_FIXTURE_DIR")
    if not raw:
        pytest.skip("Set CHEMPYFI_ORCA_LED_FIXTURE_DIR to a super/ligand/receptor ORCA fixture directory")
    p = Path(raw)
    if not p.is_dir():
        pytest.skip(f"CHEMPYFI_ORCA_LED_FIXTURE_DIR is not a directory: {p}")
    return p


def _fragment_partition_from_env():
    lig = os.environ.get("CHEMPYFI_ORCA_LED_LIGAND_FRAGMENTS")
    rec = os.environ.get("CHEMPYFI_ORCA_LED_RECEPTOR_FRAGMENTS")
    if not lig or not rec:
        pytest.skip(
            "Set CHEMPYFI_ORCA_LED_LIGAND_FRAGMENTS and CHEMPYFI_ORCA_LED_RECEPTOR_FRAGMENTS "
            "(comma-separated 1-based IDs) for equivalence test"
        )
    return (
        tuple(int(x.strip()) for x in lig.split(",") if x.strip()),
        tuple(int(x.strip()) for x in rec.split(",") if x.strip()),
    )


@pytest.mark.integration
def test_legacy_and_opi_gas_phase_paths_match():
    pytest.importorskip("opi")
    from chempyfi_covaled.equivalence.harness import compare_gas_phase_led_paths

    fixture = _fixture_dir()
    lig, rec = _fragment_partition_from_env()
    method = os.environ.get("CHEMPYFI_ORCA_LED_METHOD", "DLPNO-CCSD(T)")
    report = compare_gas_phase_led_paths(
        fixture,
        lig,
        rec,
        method=method,
        force_ledaw=os.environ.get("CHEMPYFI_ORCA_LED_FORCE_LEDAW", "").lower() in ("1", "true", "yes"),
        verbose=False,
    )
    assert report.get("overall_verdict") in ("PASS", "PASS_WITH_GAPS"), report
