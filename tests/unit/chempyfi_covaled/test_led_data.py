"""LEDData validation and pandas bridge."""

import numpy as np
import pytest

from chempyfi_covaled.led_data import KJMOL_PER_HARTREE, LEDData, LEDSystemData, hartree_to_kjmol


@pytest.mark.unit
def test_led_data_validates_shape():
    total = np.zeros((2, 2))
    d = LEDData(fragment_ids=(1, 2), total=total, energy_unit="kJ/mol")
    assert d.n_fragments == 2


@pytest.mark.unit
def test_led_data_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        LEDData(fragment_ids=(1, 2, 3), total=np.zeros((2, 2)), energy_unit="kJ/mol")


@pytest.mark.unit
def test_hartree_conversion():
    assert hartree_to_kjmol(np.array([1.0]))[0] == pytest.approx(KJMOL_PER_HARTREE)


@pytest.mark.unit
def test_system_partition_validation():
    sup = LEDData(fragment_ids=(1, 2, 3), total=np.zeros((3, 3)), energy_unit="kJ/mol")
    lig = LEDData(fragment_ids=(1, 2), total=np.zeros((2, 2)), energy_unit="kJ/mol")
    rec = LEDData(fragment_ids=(1,), total=np.zeros((1, 1)), energy_unit="kJ/mol")
    sys = LEDSystemData(sup, lig, rec, (1, 2), (3,))
    sys.validate_partition()
    bad = LEDSystemData(sup, lig, rec, (1, 2), (99,))
    with pytest.raises(ValueError, match="not in supersystem"):
        bad.validate_partition()
