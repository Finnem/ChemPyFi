"""Pure-Python LED transformations (replaces LEDAW post-processing, not vendored code)."""

from __future__ import annotations

import numpy as np


def _fp_el_prep_denominator(matrix: np.ndarray, index: int) -> float:
    """Off-diagonal absolute weight for fragment ``index``.

    LEDAW ``compute_denominator_for_fp_el_prep`` uses pandas ``DataFrame.sum``,
    which skips NaN. These matrices are stored as an **upper triangle**: the
    lower triangle is NaN because those entries are *not stored*, not because
    the interaction is scientifically unknown. NumPy ``.sum()`` would turn a
    valid row into NaN as soon as one unstored cell is present, which then
    poisons every redistribution denominator.

    ``np.nansum`` is used only for that absent-cell convention. A NaN
    *diagonal* still yields a NaN denominator, matching pandas
    ``sum() - abs(diag)`` when ``diag`` itself is NaN.
    """
    diag = matrix[index, index]
    row_sum = float(np.nansum(np.abs(matrix[index, :]))) - abs(diag)
    col_sum = float(np.nansum(np.abs(matrix[:, index]))) - abs(diag)
    return row_sum + col_sum


def fp_el_prep_distribution(ref_matrix: np.ndarray) -> np.ndarray:
    """
    Fragment-pair electronic preparation redistribution (LEDAW ``compute_fp_el_prep``).

    Input may be dense-symmetric or LEDAW upper-triangle (NaN below the
    diagonal). Returns upper-triangle values only (lower triangle and
    diagonal set to NaN), matching LEDAW Excel ``REF-EL-PREP``.
    """
    ref = np.asarray(ref_matrix, dtype=float)
    if ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError("ref_matrix must be square")
    n = ref.shape[0]
    diagonal = np.diag(ref)
    distributed = np.full_like(ref, np.nan)
    denominators = np.array([_fp_el_prep_denominator(ref, i) for i in range(n)])
    for i in range(n):
        for j in range(i + 1, n):
            if denominators[i] == 0 or denominators[j] == 0:
                continue
            if not np.isfinite(denominators[i]) or not np.isfinite(denominators[j]):
                continue
            coupling = ref[i, j]
            if not np.isfinite(coupling):
                coupling = ref[j, i]
            if not np.isfinite(coupling):
                continue
            term_1 = (diagonal[i] * abs(coupling)) / denominators[i]
            term_2 = (diagonal[j] * abs(coupling)) / denominators[j]
            distributed[i, j] = term_1 + term_2
    # Match LEDAW Excel: strict upper triangle only (i < j); diagonal + lower absent.
    keep = np.triu(np.ones_like(distributed, dtype=bool), k=1)
    distributed[~keep] = np.nan
    return distributed


def inter_molecular_pairs_from_dataframe(df, ligand_ids, receptor_ids) -> dict[str, float]:
    """Inter ligand–receptor entries from a Step 7 (or LED) DataFrame."""
    import pandas as pd

    lig = [int(x) for x in ligand_ids]
    rec = [int(x) for x in receptor_ids]
    out: dict[str, float] = {}
    for i in lig:
        for j in rec:
            a, b = (i, j) if i <= j else (j, i)
            key = f"{a}_{b}"
            val = None
            try:
                if a in df.index and b in df.columns:
                    val = df.loc[a, b]
                elif b in df.index and a in df.columns:
                    val = df.loc[b, a]
            except (KeyError, TypeError):
                val = None
            if val is not None and pd.notna(val):
                out[key] = float(val)
    return out


def inter_molecular_pairs(
    matrix: np.ndarray,
    fragment_ids: tuple[int, ...],
    ligand_ids: tuple[int, ...],
    receptor_ids: tuple[int, ...],
) -> dict[str, float]:
    """Upper-triangle inter pairs (one fragment ligand, one receptor)."""
    id_to_idx = {fid: i for i, fid in enumerate(fragment_ids)}
    out: dict[str, float] = {}
    for fi in ligand_ids:
        for fj in receptor_ids:
            if fi not in id_to_idx or fj not in id_to_idx:
                continue
            i, j = id_to_idx[fi], id_to_idx[fj]
            if i > j:
                i, j = j, i
            val = float(matrix[i, j])
            if np.isfinite(val):
                out[f"{fragment_ids[i]}_{fragment_ids[j]}"] = val
    return out
