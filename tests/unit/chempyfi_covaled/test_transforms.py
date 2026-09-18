"""Pure-Python fp-EL-PREP (replaces LEDAW compute_fp_el_prep)."""

import numpy as np
import pandas as pd
import pytest

from chempyfi_covaled.transforms import fp_el_prep_distribution


def _ledaw_denominator(df: pd.DataFrame, index: int) -> float:
    row_sum = abs(df.iloc[index, :]).sum() - abs(df.iloc[index, index])
    col_sum = abs(df.iloc[:, index]).sum() - abs(df.iloc[index, index])
    return row_sum + col_sum


def _ledaw_fp_el_prep(ref: np.ndarray) -> np.ndarray:
    """Independent pandas replica of LEDAW ``compute_fp_el_prep``."""
    df = pd.DataFrame(ref)
    n = df.shape[0]
    diagonal = np.diag(df.values)
    out = np.full((n, n), np.nan)
    denoms = np.array([_ledaw_denominator(df, i) for i in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            if denoms[i] == 0 or denoms[j] == 0:
                continue
            if not np.isfinite(denoms[i]) or not np.isfinite(denoms[j]):
                continue
            term_1 = (diagonal[i] * abs(df.iloc[i, j])) / denoms[i]
            term_2 = (diagonal[j] * abs(df.iloc[i, j])) / denoms[j]
            out[i, j] = term_1 + term_2
    return out


@pytest.mark.unit
def test_fp_el_prep_numerical_reference():
    """Matches LEDAW ``compute_fp_el_prep`` formula (hand-verified 2×2)."""
    ref = np.array([[1.0, 2.0], [2.0, 3.0]])
    got = fp_el_prep_distribution(ref)
    assert got[0, 1] == pytest.approx(2.0)


@pytest.mark.unit
def test_fp_el_prep_upper_triangle_only():
    ref = np.array([[1.0, 4.0], [4.0, 2.0]])
    out = fp_el_prep_distribution(ref)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 0]) and np.isnan(out[1, 1])
    assert np.isfinite(out[0, 1])


@pytest.mark.unit
def test_fp_el_prep_nan_lower_triangle_does_not_collapse():
    """Regression: NumPy ``.sum()`` on NaN-padded rows made every cell NaN."""
    ref = np.full((3, 3), np.nan)
    np.fill_diagonal(ref, [1.0, 4.0, 6.0])
    ref[0, 1], ref[0, 2], ref[1, 2] = 2.0, 3.0, 5.0
    poisoned = np.array([float(np.abs(ref[i, :]).sum() - abs(ref[i, i])) for i in range(3)])
    assert np.isnan(poisoned[1:]).all()
    out = fp_el_prep_distribution(ref)
    assert np.isfinite(out[0, 1]) and np.isfinite(out[0, 2]) and np.isfinite(out[1, 2])
    assert np.isnan(out[1, 0]) and np.isnan(out[2, 0]) and np.isnan(out[2, 1])
    np.testing.assert_allclose(out, _ledaw_fp_el_prep(ref), equal_nan=True)


@pytest.mark.unit
def test_fp_el_prep_redistribution_conserves_diagonal_total():
    ref = np.array(
        [
            [1.0, 2.0, -3.0],
            [np.nan, -4.0, 5.0],
            [np.nan, np.nan, 6.0],
        ]
    )
    out = fp_el_prep_distribution(ref)
    assert np.nansum(out) == pytest.approx(float(np.nansum(np.diag(ref))))


@pytest.mark.unit
def test_fp_el_prep_all_nan_non_applicable_slice_stays_nan():
    ref = np.array(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, 1.0, 2.0],
            [np.nan, np.nan, 3.0],
        ]
    )
    out = fp_el_prep_distribution(ref)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert np.isfinite(out[1, 2])


@pytest.mark.unit
def test_fp_el_prep_zero_sum_row_leaves_pairs_nan():
    ref = np.array(
        [
            [5.0, 0.0, 0.0],
            [np.nan, 1.0, 2.0],
            [np.nan, np.nan, 3.0],
        ]
    )
    out = fp_el_prep_distribution(ref)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert np.isfinite(out[1, 2])


@pytest.mark.unit
def test_fp_el_prep_positive_and_negative_preparation():
    ref = np.array(
        [
            [2.0, 1.0, 1.0],
            [np.nan, -8.0, 1.0],
            [np.nan, np.nan, 4.0],
        ]
    )
    out = fp_el_prep_distribution(ref)
    assert out[0, 1] != 0.0 and out[0, 2] != 0.0 and out[1, 2] != 0.0
    # Fragment 1 (negative diag) contributes a negative share to its pairs.
    assert out[1, 2] < 0.0
    assert np.nansum(out) == pytest.approx(float(np.nansum(np.diag(ref))))
    np.testing.assert_allclose(out, _ledaw_fp_el_prep(ref), equal_nan=True)
