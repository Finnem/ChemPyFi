"""CovaLED Step 7 matrix subtraction (pure pandas)."""

import numpy as np
import pandas as pd
import pytest

from chempyfi.orcautil.covaled_math import compute_covaled_interaction_matrix as _compute_covaled_interaction_matrix


@pytest.mark.unit
def test_intra_ligand_pair_subtracts_subsystem():
    df_super = pd.DataFrame(
        [[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    df_lig = pd.DataFrame([[0.0, 4.0], [4.0, 0.0]], index=[1, 2], columns=[1, 2])
    df_rec = pd.DataFrame([[0.0]], index=[3], columns=[3])
    result = _compute_covaled_interaction_matrix(df_super, df_lig, df_rec, [1, 2], [3])
    assert result.loc[1, 2] == pytest.approx(6.0)


@pytest.mark.unit
def test_inter_molecular_pair_unchanged():
    df_super = pd.DataFrame(
        [[0.0, -5.0], [-5.0, 0.0]], index=[1, 2], columns=[1, 2]
    )
    df_lig = pd.DataFrame([[0.0]], index=[1], columns=[1])
    df_rec = pd.DataFrame([[0.0]], index=[2], columns=[2])
    result = _compute_covaled_interaction_matrix(df_super, df_lig, df_rec, [1], [2])
    assert result.loc[1, 2] == pytest.approx(-5.0)


@pytest.mark.unit
def test_extract_interaction_energy_sums_upper_triangle():
    from chempyfi_covaled.analysis import interaction_energy_from_matrix

    df = pd.DataFrame([[0.0, 2.0], [2.0, 0.0]], index=[1, 2], columns=[1, 2])
    # Lower triangle is ignored even when populated (no double-count).
    assert interaction_energy_from_matrix(df) == pytest.approx(2.0)


@pytest.mark.unit
def test_step7_subtracts_intra_fragment_diagonal():
    df_super = pd.DataFrame(
        [[10.0, 1.0], [np.nan, 20.0]],
        index=[1, 2],
        columns=[1, 2],
    )
    df_lig = pd.DataFrame([[7.0]], index=[1], columns=[1])
    df_rec = pd.DataFrame([[4.0]], index=[1], columns=[1])
    result = _compute_covaled_interaction_matrix(df_super, df_lig, df_rec, [1], [2])
    assert result.loc[1, 1] == pytest.approx(3.0)
    assert result.loc[2, 2] == pytest.approx(16.0)
    assert result.loc[1, 2] == pytest.approx(1.0)
