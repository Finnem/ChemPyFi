"""Compare scalars and matrices with explicit tolerance reporting."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import tolerances as tol


def _finite_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isfinite(a) & np.isfinite(b)


def compare_matrices(
    legacy: np.ndarray,
    new: np.ndarray,
    *,
    atol: float = tol.TOL_MATRIX_ELEMENT_KJ_MOL,
    rtol: float = tol.TOL_RELATIVE,
    name: str = "matrix",
) -> Dict[str, Any]:
    la = np.asarray(legacy, dtype=float)
    nb = np.asarray(new, dtype=float)
    out: Dict[str, Any] = {"name": name, "shape_legacy": list(la.shape), "shape_new": list(nb.shape)}
    if la.shape != nb.shape:
        out["verdict"] = "FAIL"
        out["reason"] = "shape_mismatch"
        return out
    mask = _finite_mask(la, nb)
    comparison_region = "all_finite_overlap"
    if mask.sum() < la.size // 4:
        upper = np.triu(np.ones(la.shape, dtype=bool), k=1)
        mask = upper & _finite_mask(la, nb)
        comparison_region = "strict_upper_triangle_off_diagonal"
    if not mask.any():
        out["verdict"] = "NOT_TESTED"
        out["reason"] = "no_finite_overlap"
        out["comparison_region"] = comparison_region
        return out
    out["comparison_region"] = comparison_region
    out["n_elements_compared"] = int(mask.sum())
    diff = np.abs(la - nb)
    diff_masked = diff[mask]
    out["max_abs_diff"] = float(np.max(diff_masked))
    rel = diff_masked / np.maximum(np.abs(la[mask]), tol.MIN_DENOM_KJ_MOL)
    out["max_rel_diff"] = float(np.max(rel))
    out["sum_abs_diff"] = float(np.sum(diff_masked))
    out["tolerance_atol"] = atol
    out["tolerance_rtol"] = rtol
    out["verdict"] = "PASS" if np.allclose(la[mask], nb[mask], atol=atol, rtol=rtol) else "FAIL"
    return out


def compare_scalars(
    legacy: float,
    new: float,
    *,
    atol: float = tol.TOL_ENERGY_SCALAR_KJ_MOL,
    rtol: float = tol.TOL_RELATIVE,
    name: str = "scalar",
) -> Dict[str, Any]:
    lv, nv = float(legacy), float(new)
    diff = abs(lv - nv)
    rel = diff / max(abs(lv), tol.MIN_DENOM_KJ_MOL)
    verdict = "PASS" if np.isclose(lv, nv, atol=atol, rtol=rtol) else "FAIL"
    return {
        "name": name,
        "legacy": lv,
        "new": nv,
        "max_abs_diff": diff,
        "max_rel_diff": rel,
        "tolerance_atol": atol,
        "tolerance_rtol": rtol,
        "verdict": verdict,
    }


def compare_dataframes_by_labels(
    legacy_df,
    new_df,
    *,
    target_ids: Tuple[int, ...],
    atol: float = tol.TOL_MATRIX_ELEMENT_KJ_MOL,
    rtol: float = tol.TOL_RELATIVE,
    name: str = "matrix",
) -> Dict[str, Any]:
    """Align both DataFrames to ``target_ids`` index/columns then compare."""
    from .fragment_align import align_square_matrix

    leg = legacy_df.copy()
    leg.index = [int(i) for i in leg.index]
    leg.columns = [int(c) for c in leg.columns]
    new = new_df.copy()
    new.index = [int(i) for i in new.index]
    new.columns = [int(c) for c in new.columns]

    leg_ids = tuple(leg.index)
    new_ids = tuple(new.index)
    leg_arr, _ = align_square_matrix(leg.values, leg_ids, target_ids)
    new_arr, _ = align_square_matrix(new.values, new_ids, target_ids)
    return compare_matrices(leg_arr, new_arr, atol=atol, rtol=rtol, name=name)
