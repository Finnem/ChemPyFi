"""Scalar CovaLED interaction energy semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chempyfi_covaled.analysis import interaction_energy_from_matrix


@pytest.mark.unit
def test_upper_triangle_includes_diagonal_excludes_lower_nan_padding():
    m = pd.DataFrame(
        [
            [1.0, 2.0, np.nan],
            [99.0, 3.0, 4.0],
            [99.0, 99.0, 5.0],
        ],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    assert interaction_energy_from_matrix(m) == pytest.approx(1.0 + 2.0 + 3.0 + 4.0 + 5.0)


@pytest.mark.unit
def test_does_not_double_count_symmetric_fill():
    dense = np.array([[1.0, 2.0], [2.0, 3.0]])
    df = pd.DataFrame(dense, index=[1, 2], columns=[1, 2])
    assert interaction_energy_from_matrix(df) == pytest.approx(6.0)


@pytest.mark.unit
def test_regression_5l4q_historical_total_interaction_energy():
    """Matches extract_COVALED_run.log TOTAL INTERACTION ENERGY: -182.5663 kJ/mol."""
    from chempyfi_covaled.equivalence.harness import resolve_fixture_paths, run_legacy_gas_phase

    from tests.oracle_fixture import relax_5l4q_fixture_dir

    fixture = relax_5l4q_fixture_dir()
    if fixture is None or not fixture.is_dir():
        pytest.skip("Set CHEMPYFI_ORCA_LED_FIXTURE_DIR to the 5L4Q fixture directory")
    paths = resolve_fixture_paths(fixture)
    lig = [36]
    rec = list(range(1, 36))
    legacy = run_legacy_gas_phase(paths, lig, rec)
    e = legacy["total_interaction_energy"]
    assert e == pytest.approx(-182.5663, abs=0.001)
    historical = -182.5663
    assert abs(e - historical) <= 0.001
