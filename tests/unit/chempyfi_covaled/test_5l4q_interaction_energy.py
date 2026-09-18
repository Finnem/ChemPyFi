"""5L4Q interaction-energy scalar vs historical extract_COVALED output."""

from pathlib import Path

import pandas as pd
import pytest

from chempyfi_covaled.analysis import (
    compute_covaled_interaction_matrix,
    interaction_energy_from_matrix,
)
from tests.oracle_fixture import DEFAULT_5L4Q_DIRNAME, relax_5l4q_fixture_dir

HISTORICAL_XLSX_BASENAME = f"{DEFAULT_5L4Q_DIRNAME}_LED_interaction.xlsx"
# extract_COVALED_run.log: TOTAL INTERACTION ENERGY: -182.5663 kJ/mol
HISTORICAL_E_INT_KJ_MOL = -182.56630608495888


def _require_fixture() -> Path:
    fixture = relax_5l4q_fixture_dir()
    if fixture is None or not fixture.is_dir():
        pytest.skip("Set CHEMPYFI_ORCA_LED_FIXTURE_DIR to the 5L4Q fixture directory")
    return fixture


def _load_total(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="TOTAL", index_col=0)
    df.index = [int(i) for i in df.index]
    df.columns = [int(c) for c in df.columns]
    return df


@pytest.mark.unit
def test_5l4q_historical_xlsx_nansum_is_documented_scalar():
    fixture = _require_fixture()
    historical_xlsx = fixture / HISTORICAL_XLSX_BASENAME
    if not historical_xlsx.exists():
        pytest.skip("5L4Q LED_interaction.xlsx not in fixture directory")
    hist = pd.read_excel(historical_xlsx, index_col=0)
    assert interaction_energy_from_matrix(hist) == pytest.approx(HISTORICAL_E_INT_KJ_MOL, abs=1e-8)


@pytest.mark.unit
def test_5l4q_step7_reproduces_historical_interaction_energy():
    fixture = _require_fixture()
    historical_xlsx = fixture / HISTORICAL_XLSX_BASENAME
    super_x = fixture / "super/orca_2111303/All_Standard_LED_matrices.xlsx"
    lig_x = fixture / "sub2/orca_2111304/All_Standard_LED_matrices.xlsx"
    rec_x = fixture / "sub1/orca_2111305/All_Standard_LED_matrices.xlsx"
    if not super_x.exists():
        pytest.skip("5L4Q All_Standard Excel not in fixture directory")
    led_int = compute_covaled_interaction_matrix(
        _load_total(super_x),
        _load_total(lig_x),
        _load_total(rec_x),
        [36],
        list(range(1, 36)),
    )
    e_int = interaction_energy_from_matrix(led_int)
    assert e_int == pytest.approx(HISTORICAL_E_INT_KJ_MOL, abs=1e-8)
    if historical_xlsx.exists():
        hist = pd.read_excel(historical_xlsx, index_col=0)
        hist.index = [int(i) for i in hist.index]
        hist.columns = [int(c) for c in hist.columns]
        diff = led_int.values.astype(float) - hist.values.astype(float)
        finite = pd.notna(led_int) & pd.notna(hist)
        assert float(diff[finite.values].max()) == pytest.approx(0.0, abs=1e-9)
