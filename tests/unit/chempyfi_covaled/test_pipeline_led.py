"""LEDData → CovaLED without Excel."""

import numpy as np
import pytest

from chempyfi_covaled.analysis import compute_covaled_from_system
from chempyfi_covaled.led_data import LEDData, LEDSystemData


def _led(ids, upper_val):
    n = len(ids)
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = upper_val
    return LEDData(fragment_ids=ids, total=m, energy_unit="kJ/mol")


@pytest.mark.unit
def test_compute_covaled_from_system_matches_pandas_path():
    sup = _led((1, 2, 3), 10.0)
    sup.total[0, 1] = sup.total[1, 0] = 10.0
    sup.total[0, 2] = sup.total[2, 0] = 0.0
    sup.total[1, 2] = sup.total[2, 1] = 0.0
    lig = _led((1, 2), 4.0)
    rec = _led((3,), 0.0)
    system = LEDSystemData(sup, lig, rec, (1, 2), (3,))
    out = compute_covaled_from_system(system)
    assert out.loc[1, 2] == pytest.approx(6.0)
