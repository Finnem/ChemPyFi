"""Characterize unused fp-CovaLED Step 8 redistribution (not wired in production)."""

import pandas as pd
import pytest

from chempyfi.orcautil.covaled_math import compute_fp_covaled_interactions as _compute_fp_covaled_interactions


@pytest.mark.unit
def test_step8_redistributes_intra_to_inter_pairs():
    led_int = pd.DataFrame(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
        index=[1, 2, 3],
        columns=[1, 2, 3],
    )
    inter_pairs = {"1_3": -10.0, "2_3": -10.0}
    out = _compute_fp_covaled_interactions(led_int, [1, 2], [3], inter_pairs)
    assert set(out.keys()) == {"1_3", "2_3"}
    assert sum(out.values()) == pytest.approx(sum(inter_pairs.values()) + 3.0, rel=1e-6)


@pytest.mark.unit
def test_step8_no_op_without_intra():
    led_int = pd.DataFrame([[0.0]], index=[1], columns=[1])
    inter_pairs = {"1_2": -5.0}
    out = _compute_fp_covaled_interactions(led_int, [1], [2], inter_pairs)
    assert out == inter_pairs
