"""Load and compare LEDAW ``Summary_fp-LED_matrices.xlsx`` (gas-phase oracle)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# Retired bulk-solvation sheets — never compare.
FP_EXCLUDED_SHEETS = frozenset({"SOLV", "SOLV-STD", "SOLV-fp"})

# Gas-phase fp sheets written for DLPNO-CCSD(T) (matches ``process_fp_LED_matrices`` when e_solv=0).
FP_SHEETS_DLPNO_CCSDT: Tuple[str, ...] = (
    "TOTAL",
    "REF",
    "Electrostat",
    "Exchange",
    "REF-EL-PREP",
    "C-CCSD(T)",
    "Disp CCSD(T)",
    "Inter-NonDisp-C-CCSD(T)",
    "C-CCSD(T)-EL-PREP",
    "CCSD(T)-EL-PREP",
)

FP_SHEETS_HFLD: Tuple[str, ...] = (
    "TOTAL",
    "REF",
    "Electrostat",
    "Exchange",
    "REF-EL-PREP",
    "Disp HFLD",
)


def fp_summary_excel_for(out_path: Path) -> Path:
    return Path(out_path).parent / "Summary_fp-LED_matrices.xlsx"


def gas_phase_fp_sheet_names(method: str) -> Tuple[str, ...]:
    if method.lower() == "hfld":
        return FP_SHEETS_HFLD
    if method.lower() in ("dlpno-ccsd(t)", "dlpno-ccsd"):
        return FP_SHEETS_DLPNO_CCSDT
    raise ValueError(f"Unsupported method for fp Excel oracle: {method}")


def load_excel_fp_matrices(excel_path: Path, *, method: str) -> Dict[str, pd.DataFrame]:
    """Parse gas-phase fp sheets from LEDAW ``Summary_fp-LED_matrices.xlsx``."""
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Missing fp-LED summary Excel: {excel_path}")
    expected = set(gas_phase_fp_sheet_names(method))
    out: Dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(excel_path) as xl:
        for sheet in xl.sheet_names:
            if sheet in FP_EXCLUDED_SHEETS or sheet.upper().startswith("SOLV"):
                continue
            if sheet not in expected:
                continue
            df = pd.read_excel(xl, sheet_name=sheet, index_col=0)
            df.index = [int(i) for i in df.index]
            df.columns = [int(c) for c in df.columns]
            out[sheet] = df
    return out


# LEDAW fp Excel: values on strict upper triangle (i < j); diagonal and lower are NaN.
# Standard LED sheets may include diagonal REF self-energies; fp sheets do not.
FP_MATRIX_STORAGE = (
    "strict_upper_triangle_off_diagonal_only; "
    "diagonal_and_lower_triangle_are_NaN_absent_cells_not_unknowns"
)
