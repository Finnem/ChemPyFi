"""Gas-phase fp-LED matrix assembly (no bulk-solvation redistribution)."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from .transforms import fp_el_prep_distribution


def _fp_el_prep_dataframe(df_ref: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        fp_el_prep_distribution(df_ref.values),
        index=df_ref.index,
        columns=df_ref.columns,
    )
    return out


def _int_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [int(c) for c in df.columns]
    df.index = [int(i) for i in df.index]
    return df


def build_gas_phase_fp_led_matrices(
    standard_led_matrices: Dict[str, pd.DataFrame],
    method: str,
) -> Dict[str, pd.DataFrame]:
    """
    fp-LED reconstruction from standard LED component matrices (gas phase only).

    Bulk-solvation fp-LED redistribution is not supported.
    """
    df_ref = standard_led_matrices.get("REF", pd.DataFrame())
    if df_ref.empty:
        raise ValueError("REF matrix not found in standard LED matrices")
    df_ref = _int_index(df_ref)
    df_ref_distributed = _fp_el_prep_dataframe(df_ref)

    df_electrostat = _int_index(standard_led_matrices.get("Electrostat", pd.DataFrame()))
    df_exchange = _int_index(standard_led_matrices.get("Exchange", pd.DataFrame()))

    fp_led_matrices: Dict[str, pd.DataFrame] = {}
    method_l = method.lower()

    if method_l in ("dlpno-ccsd(t)", "dlpno-ccsd"):
        df_total = _int_index(standard_led_matrices.get("TOTAL", pd.DataFrame()))
        df_total_distributed = _fp_el_prep_dataframe(df_total)
        df_c_ccsd = df_total_distributed - df_ref_distributed

        disp_sheet = "Disp CCSD" if method_l == "dlpno-ccsd" else "Disp CCSD(T)"
        inter_nondisp_sheet = (
            "Inter-NonDisp-C-CCSD" if method_l == "dlpno-ccsd" else "Inter-NonDisp-C-CCSD(T)"
        )
        df_disp_ccsd = _int_index(standard_led_matrices.get(disp_sheet, pd.DataFrame()))
        df_inter_nondisp = _int_index(standard_led_matrices.get(inter_nondisp_sheet, pd.DataFrame()))

        df_ref_final = df_electrostat + df_exchange + df_ref_distributed
        df_c_ccsd_final = df_c_ccsd + df_disp_ccsd + df_inter_nondisp
        df_total_final = df_ref_final + df_c_ccsd_final

        fp_led_matrices["TOTAL"] = df_total_final
        fp_led_matrices["REF"] = df_ref_final
        fp_led_matrices["Electrostat"] = df_electrostat
        fp_led_matrices["Exchange"] = df_exchange
        fp_led_matrices["REF-EL-PREP"] = df_ref_distributed
        fp_led_matrices["C-CCSD(T)" if method_l == "dlpno-ccsd(t)" else "C-CCSD"] = df_c_ccsd_final
        fp_led_matrices[disp_sheet] = df_disp_ccsd
        fp_led_matrices[inter_nondisp_sheet] = df_inter_nondisp
        fp_led_matrices["C-CCSD(T)-EL-PREP" if method_l == "dlpno-ccsd(t)" else "C-CCSD-EL-PREP"] = (
            df_c_ccsd
        )
        fp_led_matrices["CCSD(T)-EL-PREP" if method_l == "dlpno-ccsd(t)" else "CCSD-EL-PREP"] = (
            df_total_distributed
        )

    elif method_l == "hfld":
        df_ref_final = df_electrostat + df_exchange + df_ref_distributed
        df_disp_hfld = _int_index(standard_led_matrices.get("Disp HFLD", pd.DataFrame()))
        df_total_hfld = df_ref_final + df_disp_hfld
        fp_led_matrices["TOTAL"] = df_total_hfld
        fp_led_matrices["REF"] = df_ref_final
        fp_led_matrices["Electrostat"] = df_electrostat
        fp_led_matrices["Exchange"] = df_exchange
        fp_led_matrices["REF-EL-PREP"] = df_ref_distributed
        fp_led_matrices["Disp HFLD"] = df_disp_hfld
    else:
        raise ValueError(f"Unsupported method for gas-phase fp-LED: {method}")

    return fp_led_matrices
