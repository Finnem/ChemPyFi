"""Fragment ID ordering and matrix permutation for equivalence checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


class FragmentAlignmentError(ValueError):
    """Fragment sets or ordering cannot be reconciled."""


@dataclass(frozen=True)
class FragmentMapping:
    """Maps ``source_ids`` row/col order to ``target_ids`` order."""

    source_ids: Tuple[int, ...]
    target_ids: Tuple[int, ...]
    permutation: Tuple[int, ...]  # source index for each target position

    @property
    def is_identity(self) -> bool:
        return self.source_ids == self.target_ids


def permutation_to_align(source_ids: Sequence[int], target_ids: Sequence[int]) -> FragmentMapping:
    """
    Require equal sets of fragment IDs; return indices ``perm`` with
    ``source_ids[perm[k]] == target_ids[k]``.
    """
    src = tuple(int(x) for x in source_ids)
    tgt = tuple(int(x) for x in target_ids)
    if set(src) != set(tgt):
        only_src = sorted(set(src) - set(tgt))
        only_tgt = sorted(set(tgt) - set(src))
        raise FragmentAlignmentError(
            f"Fragment ID set mismatch. Only in source: {only_src}; only in target: {only_tgt}"
        )
    if len(src) != len(tgt):
        raise FragmentAlignmentError(f"Duplicate fragment IDs? source={src}, target={tgt}")
    perm = []
    for tid in tgt:
        if src.count(tid) != 1:
            raise FragmentAlignmentError(f"Fragment ID {tid} is not unique in source order {src}")
        perm.append(src.index(tid))
    return FragmentMapping(source_ids=src, target_ids=tgt, permutation=tuple(perm))


def apply_fragment_permutation(matrix: np.ndarray, permutation: Sequence[int]) -> np.ndarray:
    """Reorder rows and columns of a square matrix."""
    p = list(permutation)
    return np.asarray(matrix, dtype=float)[np.ix_(p, p)]


def align_square_matrix(
    matrix: np.ndarray,
    source_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Tuple[np.ndarray, FragmentMapping]:
    mapping = permutation_to_align(source_ids, target_ids)
    if mapping.is_identity:
        return np.asarray(matrix, dtype=float), mapping
    return apply_fragment_permutation(matrix, mapping.permutation), mapping
