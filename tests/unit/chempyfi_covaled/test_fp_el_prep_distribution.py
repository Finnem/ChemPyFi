"""Regression tests for gas-phase fp REF-EL-PREP redistribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chempyfi_covaled.transforms import fp_el_prep_distribution


def _upper_triangle_sum(m: np.ndarray) -> float:
    u = np.triu(m, k=1)
    return float(np.nansum(u))


@pytest.mark.unit
def test_upper_triangle_with_nan_lower_not_all_nan():
    """LEDAW-style REF: finite upper triangle + diagonal, NaN lower triangle."""
    n = 4
    ref = np.full((n, n), np.nan)
    np.fill_diagonal(ref, [-100.0, -80.0, -60.0, -40.0])
    ref[0, 1] = -2.0
    ref[0, 2] = 3.0
    ref[0, 3] = -1.0
    ref[1, 2] = 4.0
    ref[1, 3] = -5.0
    ref[2, 3] = 2.0

    out = fp_el_prep_distribution(ref)
    assert np.isfinite(out).sum() == 6
    for i in range(n):
        for j in range(n):
            if j <= i:
                assert np.isnan(out[i, j]), f"expected NaN at ({i},{j}), got {out[i,j]}"
    assert _upper_triangle_sum(out) != 0.0


@pytest.mark.unit
def test_redistribution_matches_pandas_denominator_convention():
    """Same numeric result as LEDAW when denominators use skip-NaN sums."""
    ref = np.array(
        [
            [-10.0, -1.0, np.nan],
            [np.nan, -20.0, 2.0],
            [np.nan, np.nan, -30.0],
        ],
        dtype=float,
    )
    out = fp_el_prep_distribution(ref)
    # Manual LEDAW-style terms for (0,1)
    d0 = abs(-1.0)
    d1 = abs(-1.0) + abs(2.0)
    expected_01 = (-10.0 * abs(-1.0)) / d0 + (-20.0 * abs(-1.0)) / d1
    assert out[0, 1] == pytest.approx(expected_01)


@pytest.mark.unit
def test_zero_denominator_row_skips_pairs():
    ref = np.diag([-5.0, -5.0, -5.0]).astype(float)
    out = fp_el_prep_distribution(ref)
    assert not np.any(np.isfinite(out))


@pytest.mark.unit
def test_non_applicable_entries_remain_nan():
    ref = np.full((3, 3), np.nan)
    np.fill_diagonal(ref, [-1.0, -2.0, -3.0])
    ref[0, 2] = 1.5
    out = fp_el_prep_distribution(ref)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0]) and np.isnan(out[2, 0])
    assert np.isnan(out[1, 2])


@pytest.mark.unit
def test_positive_and_negative_coupling():
    ref = np.full((3, 3), np.nan)
    np.fill_diagonal(ref, [10.0, -20.0, 30.0])
    ref[0, 1] = -4.0
    ref[1, 2] = 5.0
    out = fp_el_prep_distribution(ref)
    assert np.isfinite(out[0, 1]) and np.isfinite(out[1, 2])
    assert out[0, 1] != out[1, 2]


@pytest.mark.unit
def test_regression_5l4q_super_ref_sheet():
    """Reproduce prior all-NaN failure: upper+diag REF from real fixture Excel."""
    from pathlib import Path

    excel = (
        Path(__file__).resolve().parents[3]
        / "test/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/super/orca_2111303/All_Standard_LED_matrices.xlsx"
    )
    if not excel.exists():
        pytest.skip("5L4Q fixture not present")
    ref = pd.read_excel(excel, sheet_name="REF", index_col=0).values.astype(float)
    out = fp_el_prep_distribution(ref)
    assert np.isfinite(out).sum() == 630

