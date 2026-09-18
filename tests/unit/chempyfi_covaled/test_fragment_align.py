"""Fragment permutation for LED equivalence."""

import numpy as np
import pytest

from chempyfi_covaled.equivalence.fragment_align import (
    FragmentAlignmentError,
    align_square_matrix,
    apply_fragment_permutation,
    permutation_to_align,
)


@pytest.mark.unit
def test_identity_permutation():
    ids = (1, 2, 3)
    m = permutation_to_align(ids, ids)
    assert m.is_identity
    assert m.permutation == (0, 1, 2)


@pytest.mark.unit
def test_reorder_permutation():
    src = (2, 1, 3)
    tgt = (1, 2, 3)
    m = permutation_to_align(src, tgt)
    mat = np.arange(9, dtype=float).reshape(3, 3)
    aligned = apply_fragment_permutation(mat, m.permutation)
    assert aligned[0, 1] == mat[1, 0]


@pytest.mark.unit
def test_set_mismatch_raises():
    with pytest.raises(FragmentAlignmentError, match="set mismatch"):
        permutation_to_align((1, 2), (1, 3))


@pytest.mark.unit
def test_align_square_matrix():
    mat = np.eye(2)
    out, mapping = align_square_matrix(mat, (2, 1), (1, 2))
    assert mapping.permutation == (1, 0)
    assert out[0, 0] == mat[1, 1]
