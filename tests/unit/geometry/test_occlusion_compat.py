"""Verify rdutil and geometry package expose the same occlusion primitive."""

import numpy as np
import pytest

from chempyfi.geometry.occlusion import check_occlusion as geom_check
from chempyfi.rdutil.geometry import check_occlusion as rdutil_check


@pytest.mark.unit
def test_geometry_and_rdutil_occlusion_match():
    from_atoms = np.array([[0.0, 0.0, 0.0]])
    to_atoms = np.array([[5.0, 0.0, 0.0]])
    occluders = np.array([[2.5, 0.0, 0.0]])
    assert bool(geom_check(from_atoms, to_atoms, occluders, default_radius=1.5)) == bool(
        rdutil_check(from_atoms, to_atoms, occluders, default_radius=1.5)
    )
