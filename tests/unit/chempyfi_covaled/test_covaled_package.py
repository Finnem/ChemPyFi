"""chempyfi_covaled package runs without chempyfi.orcautil import."""

import pandas as pd
import pytest

from chempyfi_covaled.analysis import compute_covaled_interaction_matrix, compute_fp_covaled
from chempyfi_covaled.workflow import analyze_from_led_matrices


@pytest.mark.unit
def test_chempyfi_covaled_step7_via_package():
    df_super = pd.DataFrame(
        [[0.0, 10.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    df_lig = pd.DataFrame([[0.0, 4.0], [4.0, 0.0]], index=[1, 2], columns=[1, 2])
    df_rec = pd.DataFrame([[0.0]], index=[3], columns=[3])
    out = compute_covaled_interaction_matrix(df_super, df_lig, df_rec, [1, 2], [3])
    assert out.loc[1, 2] == pytest.approx(6.0)


@pytest.mark.unit
def test_workflow_analyze_from_matrices():
    df_super = pd.DataFrame([[0.0, -5.0], [-5.0, 0.0]], index=[1, 2], columns=[1, 2])
    df_lig = pd.DataFrame([[0.0]], index=[1], columns=[1])
    df_rec = pd.DataFrame([[0.0]], index=[2], columns=[2])
    result = analyze_from_led_matrices(df_super, df_lig, df_rec, [1], [2])
    # Upper-triangle convention (incl. diagonal): only (1,1), (1,2), (2,2) contribute.
    assert result.interaction_matrix.total_interaction_energy() == pytest.approx(-5.0)
