"""Legacy LEDAW vs OPI+``chempyfi_covaled`` gas-phase equivalence (developer oracle)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..analysis import (
    compute_covaled_interaction_matrix,
    compute_fp_covaled,
    interaction_energy_from_matrix,
)
from ..fp_led import build_gas_phase_fp_led_matrices
from ..led_data import LEDData, LEDSystemData
from ..opi_led import extract_led_data_for_out_file
from ..transforms import inter_molecular_pairs_from_dataframe
from .fixtures import discover_fixture_orca_outputs, property_json_for, standard_excel_for
from .fp_matrix import FP_MATRIX_STORAGE, fp_summary_excel_for, gas_phase_fp_sheet_names, load_excel_fp_matrices
from .fragment_align import FragmentAlignmentError, permutation_to_align
from . import metrics
from . import tolerances as tol

EXCLUDED_SHEETS = frozenset({"SOLV", "SOLV-STD", "SOLV-fp"})


def _parse_frag_list(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def resolve_fixture_paths(
    fixture_dir: Path,
    *,
    super_name: str = "super.out",
    ligand_name: str = "ligand.out",
    receptor_name: str = "receptor.out",
    auto_discover: bool = True,
) -> Dict[str, Path]:
    fixture_dir = Path(fixture_dir)
    flat = {
        "super": fixture_dir / super_name,
        "ligand": fixture_dir / ligand_name,
        "receptor": fixture_dir / receptor_name,
    }
    if all(p.exists() for p in flat.values()):
        paths = flat
    elif auto_discover:
        paths = discover_fixture_orca_outputs(fixture_dir)
    else:
        missing = [str(p) for p in flat.values() if not p.exists()]
        raise FileNotFoundError(f"Missing fixture ORCA output(s): {missing}")

    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing fixture ORCA output: {p}")
        if not property_json_for(p).exists() and not standard_excel_for(p).exists():
            raise FileNotFoundError(
                f"Missing OPI .property.json and All_Standard_LED_matrices.xlsx for {key}: {p}"
            )
    return paths


def load_excel_standard_matrices(excel_path: Path) -> Dict[str, pd.DataFrame]:
    """Load LEDAW standard Excel sheets; exclude solvation-related sheets."""
    out: Dict[str, pd.DataFrame] = {}
    with pd.ExcelFile(excel_path) as xl:
        for sheet in xl.sheet_names:
            if sheet.upper() in EXCLUDED_SHEETS or sheet.startswith("SOLV"):
                continue
            out[sheet] = pd.read_excel(xl, sheet_name=sheet, index_col=0)
    return out


def dataframe_to_led_data(df: pd.DataFrame, fragment_ids: Optional[Tuple[int, ...]] = None) -> LEDData:
    df = df.copy()
    df.index = [int(i) for i in df.index]
    df.columns = [int(c) for c in df.columns]
    ids = fragment_ids or tuple(df.index)
    return LEDData(fragment_ids=ids, total=df.values.astype(float), energy_unit="kJ/mol")


def led_data_to_standard_matrices(led: LEDData) -> Dict[str, pd.DataFrame]:
    from ..opi_led import led_data_to_standard_matrices as _fn

    return _fn(led)


def _excel_path_for_orca_out(out_path: Path) -> Path:
    """Reference Excel oracle (pre-generated; LEDAW no longer ships with ChemPyFi)."""
    out_path = Path(out_path)
    existing = out_path.parent / "All_Standard_LED_matrices.xlsx"
    if not existing.exists():
        raise FileNotFoundError(
            f"Missing All_Standard_LED_matrices.xlsx next to {out_path} (reference oracle only)."
        )
    return existing


def run_legacy_gas_phase(
    paths: Dict[str, Path],
    ligand_fragment_ids: Sequence[int],
    receptor_fragment_ids: Sequence[int],
    *,
    method: str = "DLPNO-CCSD(T)",
    conversion_factor: float = 2625.5,
) -> Dict[str, Any]:
    lig = list(ligand_fragment_ids)
    rec = list(receptor_fragment_ids)
    excel = {}
    for role in ("super", "ligand", "receptor"):
        excel[role] = _excel_path_for_orca_out(paths[role])

    std = {role: load_excel_standard_matrices(excel[role]) for role in excel}
    df_super = std["super"]["TOTAL"].copy()
    df_lig = std["ligand"]["TOTAL"].copy()
    df_rec = std["receptor"]["TOTAL"].copy()
    for df in (df_super, df_lig, df_rec):
        df.index = [int(i) for i in df.index]
        df.columns = [int(c) for c in df.columns]

    step7 = compute_covaled_interaction_matrix(df_super, df_lig, df_rec, lig, rec)
    fp_native = build_gas_phase_fp_led_matrices(std["super"], method)
    fp_native.pop("SOLV", None)
    fp_excel_path = fp_summary_excel_for(paths["super"])
    ledaw_fp = load_excel_fp_matrices(fp_excel_path, method=method) if fp_excel_path.exists() else {}

    super_ids = tuple(int(i) for i in df_super.index)
    inter_pairs = inter_molecular_pairs_from_dataframe(step7, lig, rec)
    step8 = compute_fp_covaled(step7, lig, rec, inter_pairs)

    return {
        "excel_paths": {k: str(v) for k, v in excel.items()},
        "fp_excel_path": str(fp_excel_path),
        "standard": std,
        "fragment_ids": {"super": super_ids, "ligand": tuple(df_lig.index), "receptor": tuple(df_rec.index)},
        "step7_matrix": step7,
        "fp_matrices_native": fp_native,
        "fp_matrices_ledaw_excel": ledaw_fp,
        "step8_pairs": step8,
        "inter_pairs_input": inter_pairs,
        "total_interaction_energy": interaction_energy_from_matrix(step7),
    }


def run_opi_gas_phase(
    paths: Dict[str, Path],
    ligand_fragment_ids: Sequence[int],
    receptor_fragment_ids: Sequence[int],
    *,
    method: str = "DLPNO-CCSD(T)",
) -> Dict[str, Any]:
    lig = tuple(int(x) for x in ligand_fragment_ids)
    rec = tuple(int(x) for x in receptor_fragment_ids)
    led_sources: Dict[str, str] = {}
    led_super, led_sources["super"] = extract_led_data_for_out_file(
        paths["super"], require_opi=False
    )
    led_lig, led_sources["ligand"] = extract_led_data_for_out_file(
        paths["ligand"], require_opi=False
    )
    led_rec, led_sources["receptor"] = extract_led_data_for_out_file(
        paths["receptor"], require_opi=False
    )

    system = LEDSystemData(
        supersystem=led_super,
        ligand=led_lig,
        receptor=led_rec,
        ligand_fragment_ids=lig,
        receptor_fragment_ids=rec,
    )
    system.validate_partition()

    from ..analysis import compute_covaled_from_system

    step7_df = compute_covaled_from_system(system)
    std_super = led_data_to_standard_matrices(led_super)
    fp_native = build_gas_phase_fp_led_matrices(std_super, method)
    fp_native.pop("SOLV", None)
    inter_pairs = inter_molecular_pairs_from_dataframe(step7_df, lig, rec)
    step8 = compute_fp_covaled(step7_df, list(lig), list(rec), inter_pairs)

    return {
        "led_sources": led_sources,
        "led_data": {"super": led_super, "ligand": led_lig, "receptor": led_rec},
        "standard": {"super": std_super},
        "fragment_ids": {
            "super": led_super.fragment_ids,
            "ligand": led_lig.fragment_ids,
            "receptor": led_rec.fragment_ids,
        },
        "step7_matrix": step7_df,
        "fp_matrices_native": fp_native,
        "step8_pairs": step8,
        "inter_pairs_input": inter_pairs,
        "total_interaction_energy": interaction_energy_from_matrix(step7_df),
    }


def _compare_step7(
    legacy: Dict[str, Any],
    new: Dict[str, Any],
    target_super_ids: Tuple[int, ...],
) -> Dict[str, Any]:
    step7: Dict[str, Any] = {}
    leg = legacy["standard"]
    nd = new["led_data"]

    step7["fragment_count"] = metrics.compare_scalars(
        len(legacy["fragment_ids"]["super"]),
        len(new["fragment_ids"]["super"]),
        atol=0,
        rtol=0,
        name="fragment_count_super",
    )

    for role, key in [("super", "TOTAL"), ("ligand", "TOTAL"), ("receptor", "TOTAL")]:
        leg_df = leg[role][key]
        if role == "super":
            step7[f"{role}_total_matrix"] = metrics.compare_dataframes_by_labels(
                leg_df, nd["super"].to_pandas(), target_ids=target_super_ids, name=f"{role}_TOTAL"
            )
        else:
            new_df = nd[role].to_pandas()
            step7[f"{role}_total_matrix"] = metrics.compare_matrices(
                leg_df.values.astype(float), new_df.values.astype(float), name=f"{role}_TOTAL"
            )

    step7["interaction_matrix"] = metrics.compare_dataframes_by_labels(
        legacy["step7_matrix"],
        new["step7_matrix"],
        target_ids=target_super_ids,
        name="step7_interaction_matrix",
    )
    step7["total_interaction_energy"] = metrics.compare_scalars(
        legacy["total_interaction_energy"],
        new["total_interaction_energy"],
        name="total_interaction_energy",
    )
    step7["interaction_energy_semantics"] = {
        "helper": "chempyfi_covaled.analysis.interaction_energy_from_matrix",
        "matrix": "CovaLED Step 7 interaction map (supersystem minus subsystems)",
        "pairs": "all stored entries with row index <= column index (upper triangle incl. diagonal)",
        "excludes": "lower triangle NaN padding (absent cells, not unknown energies)",
        "units": "kJ/mol",
        "legacy_value_kj_mol": legacy["total_interaction_energy"],
        "new_value_kj_mol": new["total_interaction_energy"],
        "historical_reference_5L4Q_kj_mol": -182.5663,
        "historical_source": "extract_COVALED_run.log (CHEMPYFI_ORCA_LED_FIXTURE_DIR bundle)",
    }

    component_map = [
        ("REF", "reference"),
        ("Electrostat", "electrostatics"),
        ("Exchange", "exchange"),
    ]
    for sheet, attr in component_map:
        if sheet in leg["super"] and getattr(nd["super"].components, attr) is not None:
            leg_df = leg["super"][sheet]
            comp_arr = getattr(nd["super"].components, attr)
            new_df = pd.DataFrame(
                comp_arr, index=nd["super"].fragment_ids, columns=nd["super"].fragment_ids
            )
            step7[f"component_{attr}"] = metrics.compare_dataframes_by_labels(
                leg_df, new_df, target_ids=target_super_ids, name=sheet
            )
    if "CORR" in leg["super"] and nd["super"].components.correlation is not None:
        new_df = pd.DataFrame(
            nd["super"].components.correlation,
            index=nd["super"].fragment_ids,
            columns=nd["super"].fragment_ids,
        )
        step7["component_correlation"] = metrics.compare_dataframes_by_labels(
            leg["super"]["CORR"], new_df, target_ids=target_super_ids, name="correlation"
        )
    elif nd["super"].components.correlation is not None:
        step7["component_correlation"] = {"verdict": "NOT_TESTED", "reason": "no_legacy_CORR_sheet"}

    return step7


def _compare_step8(
    legacy: Dict[str, Any],
    new: Dict[str, Any],
    target_super_ids: Tuple[int, ...],
    *,
    method: str,
) -> Dict[str, Any]:
    step8: Dict[str, Any] = {
        "fp_matrix_storage_convention": FP_MATRIX_STORAGE,
        "fp_excel_oracle": legacy.get("fp_excel_path"),
    }
    ledaw_fp = legacy.get("fp_matrices_ledaw_excel") or {}
    # Gas-phase fp rebuild must use the same standard LED sheets as LEDAW (Excel),
    # not the reduced component set from excel_bridge ``LEDData``.
    chempyfi_covaled_fp = build_gas_phase_fp_led_matrices(legacy["standard"]["super"], method)
    chempyfi_covaled_fp.pop("SOLV", None)
    new_fp = chempyfi_covaled_fp

    if not ledaw_fp:
        step8["ref_el_prep"] = {
            "verdict": "NOT_TESTED",
            "reason": "missing_Summary_fp-LED_matrices_xlsx",
        }
        step8["fp_total_matrix"] = step8["ref_el_prep"].copy()
    else:
        if "REF-EL-PREP" in ledaw_fp and "REF-EL-PREP" in new_fp:
            step8["ref_el_prep"] = metrics.compare_dataframes_by_labels(
                ledaw_fp["REF-EL-PREP"],
                new_fp["REF-EL-PREP"],
                target_ids=target_super_ids,
                atol=tol.TOL_FP_EL_PREP_KJ_MOL,
                name="REF-EL-PREP",
            )
            step8["ref_el_prep"]["legacy_source"] = "Summary_fp-LED_matrices.xlsx"
            step8["ref_el_prep"]["new_source"] = "chempyfi_covaled.fp_led.build_gas_phase_fp_led_matrices"
        else:
            step8["ref_el_prep"] = {"verdict": "NOT_TESTED", "reason": "missing_REF-EL-PREP_sheet"}
        if "TOTAL" in ledaw_fp and "TOTAL" in new_fp:
            step8["fp_total_matrix"] = metrics.compare_dataframes_by_labels(
                ledaw_fp["TOTAL"],
                new_fp["TOTAL"],
                target_ids=target_super_ids,
                name="fp_TOTAL",
            )
            step8["fp_total_matrix"]["legacy_source"] = "Summary_fp-LED_matrices.xlsx"
            step8["fp_total_matrix"]["new_source"] = "chempyfi_covaled.fp_led.build_gas_phase_fp_led_matrices"
        else:
            step8["fp_total_matrix"] = {"verdict": "NOT_TESTED", "reason": "missing_TOTAL_sheet"}

        step8["fp_gas_phase_sheets"] = {}
        for sheet in gas_phase_fp_sheet_names(method):
            if sheet in ("REF-EL-PREP", "TOTAL"):
                continue
            if sheet not in ledaw_fp or sheet not in new_fp:
                step8["fp_gas_phase_sheets"][sheet] = {
                    "verdict": "NOT_TESTED",
                    "reason": "sheet_missing_on_one_side",
                }
                continue
            leg_df, new_df = ledaw_fp[sheet], new_fp[sheet]
            if leg_df.empty or new_df.empty or len(leg_df.index) == 0 or len(new_df.index) == 0:
                step8["fp_gas_phase_sheets"][sheet] = {
                    "verdict": "NOT_TESTED",
                    "reason": "empty_matrix_on_one_side",
                }
                continue
            try:
                step8["fp_gas_phase_sheets"][sheet] = metrics.compare_dataframes_by_labels(
                    leg_df,
                    new_df,
                    target_ids=target_super_ids,
                    atol=tol.TOL_FP_EL_PREP_KJ_MOL if "EL-PREP" in sheet else tol.TOL_MATRIX_ELEMENT_KJ_MOL,
                    name=sheet,
                )
            except FragmentAlignmentError as exc:
                step8["fp_gas_phase_sheets"][sheet] = {
                    "verdict": "NOT_TESTED",
                    "reason": "fragment_alignment",
                    "detail": str(exc),
                }

    leg_pairs = legacy["step8_pairs"]
    new_pairs = new["step8_pairs"]
    common = sorted(set(leg_pairs) & set(new_pairs))
    pair_diffs = []
    for k in common:
        pair_diffs.append(abs(leg_pairs[k] - new_pairs[k]))
    max_pair = max(pair_diffs) if pair_diffs else 0.0
    step8["pairwise_fp"] = {
        "name": "step8_pairwise_dict",
        "n_pairs_compared": len(common),
        "max_abs_diff": max_pair,
        "tolerance_atol": tol.TOL_ENERGY_SCALAR_KJ_MOL,
        "verdict": "PASS" if max_pair <= tol.TOL_ENERGY_SCALAR_KJ_MOL else "FAIL",
    }
    only_leg = sorted(set(leg_pairs) - set(new_pairs))
    only_new = sorted(set(new_pairs) - set(leg_pairs))
    if only_leg or only_new:
        step8["pairwise_fp"]["verdict"] = "FAIL"
        step8["pairwise_fp"]["only_legacy_keys"] = only_leg
        step8["pairwise_fp"]["only_new_keys"] = only_new

    sum_before = sum(legacy["inter_pairs_input"].values())
    sum_after_leg = sum(leg_pairs.values())
    sum_after_new = sum(new_pairs.values())
    e7_leg = legacy["total_interaction_energy"]
    e7_new = new["total_interaction_energy"]
    step8["sum_before_redistribution"] = metrics.compare_scalars(sum_before, sum(new["inter_pairs_input"].values()), name="inter_pair_sum_input")
    step8["sum_after_redistribution"] = metrics.compare_scalars(sum_after_leg, sum_after_new, name="fp_pair_sum")
    step8["conservation_residual_legacy"] = {
        "name": "conservation_legacy",
        "residual": float(e7_leg - sum_after_leg),
        "tolerance_atol": tol.TOL_CONSERVATION_RESIDUAL_KJ_MOL,
        "verdict": "PASS" if abs(e7_leg - sum_after_leg) <= tol.TOL_CONSERVATION_RESIDUAL_KJ_MOL else "FAIL",
    }
    step8["conservation_residual_new"] = {
        "name": "conservation_new",
        "residual": float(e7_new - sum_after_new),
        "tolerance_atol": tol.TOL_CONSERVATION_RESIDUAL_KJ_MOL,
        "verdict": "PASS" if abs(e7_new - sum_after_new) <= tol.TOL_CONSERVATION_RESIDUAL_KJ_MOL else "FAIL",
    }
    return step8


def _verdict_from_item(item: Any) -> str:
    if isinstance(item, dict) and "verdict" in item:
        return str(item["verdict"])
    return "NOT_TESTED"


def _build_parity_gates(report: Dict[str, Any]) -> Dict[str, Any]:
    step7 = report.get("step7") or {}
    step8 = report.get("step8") or {}
    gates: Dict[str, Any] = {}

    gates["STEP7_MATRIX_PARITY"] = _verdict_from_item(step7.get("interaction_matrix"))

    gates["FP_EL_PREP_PARITY"] = _verdict_from_item(step8.get("ref_el_prep"))
    gates["FP_TOTAL_MATRIX_PARITY"] = _verdict_from_item(step8.get("fp_total_matrix"))
    gates["PAIRWISE_FP_PARITY"] = _verdict_from_item(step8.get("pairwise_fp"))

    cons_leg = _verdict_from_item(step8.get("conservation_residual_legacy"))
    cons_new = _verdict_from_item(step8.get("conservation_residual_new"))
    if cons_leg == "FAIL" or cons_new == "FAIL":
        gates["ENERGY_CONSERVATION"] = "FAIL"
    elif cons_leg == "PASS" and cons_new == "PASS":
        gates["ENERGY_CONSERVATION"] = "PASS"
    else:
        gates["ENERGY_CONSERVATION"] = "NOT_TESTED"

    gates["SCALAR_INTERACTION_ENERGY_PARITY"] = _verdict_from_item(step7.get("total_interaction_energy"))

    opi_sources = report.get("new_path_led_sources") or {}
    if all(v == "opi" for v in opi_sources.values()) and len(opi_sources) == 3:
        gates["OPI_RAW_LED_PARITY"] = report.get("opi_raw_led_parity_verdict", "NOT_TESTED")
        gates["OPI_FRAGMENT_ORDER_PARITY"] = report.get("opi_fragment_order_parity_verdict", "NOT_TESTED")
        gates["OPI_EXTRACTION_PARITY"] = report.get("opi_extraction_parity_verdict", "NOT_TESTED")
    else:
        gates["OPI_RAW_LED_PARITY"] = "NOT_TESTED"
        gates["OPI_FRAGMENT_ORDER_PARITY"] = "NOT_TESTED"
        gates["OPI_EXTRACTION_PARITY"] = "NOT_TESTED"
        gates["OPI_EXTRACTION_PARITY_note"] = (
            "Requires .property.json on super/ligand/receptor; excel_bridge does not satisfy extraction gate."
        )

    return gates


def final_removal_gates(scientific_gate: str, opi_extraction_gate: str) -> Dict[str, str]:
    """Combined LEDAW removal gate (A: 5L4Q science, B: OPI JSON fixture)."""
    ledaw = "PASS" if scientific_gate == "PASS" and opi_extraction_gate == "PASS" else "NOT_SATISFIED"
    return {
        "scientific_ledaw_removal_gate": scientific_gate,
        "opi_extraction_gate": opi_extraction_gate,
        "ledaw_removal_gate": ledaw,
    }


def _scientific_removal_gate(gates: Dict[str, Any]) -> str:
    required = (
        "STEP7_MATRIX_PARITY",
        "FP_EL_PREP_PARITY",
        "FP_TOTAL_MATRIX_PARITY",
        "PAIRWISE_FP_PARITY",
        "ENERGY_CONSERVATION",
        "SCALAR_INTERACTION_ENERGY_PARITY",
    )
    if any(gates.get(k) == "FAIL" for k in required):
        return "FAIL"
    if all(gates.get(k) == "PASS" for k in required):
        return "PASS"
    return "NOT_SATISFIED"


def _overall_verdict(sections: Dict[str, Dict[str, Any]]) -> str:
    fails = []
    for section, items in sections.items():
        for _k, v in items.items():
            if isinstance(v, dict) and v.get("verdict") == "FAIL":
                fails.append(f"{section}.{_k}")
    if fails:
        return "FAIL"
    not_tested = any(
        isinstance(v, dict) and v.get("verdict") == "NOT_TESTED"
        for items in sections.values()
        for v in items.values()
    )
    return "PASS_WITH_GAPS" if not_tested else "PASS"


def compare_gas_phase_led_paths(
    fixture_dir: Path,
    ligand_fragment_ids: Sequence[int],
    receptor_fragment_ids: Sequence[int],
    *,
    method: str = "DLPNO-CCSD(T)",
    **path_kw,
) -> Dict[str, Any]:
    paths = resolve_fixture_paths(fixture_dir, **path_kw)
    legacy = run_legacy_gas_phase(
        paths,
        ligand_fragment_ids,
        receptor_fragment_ids,
        method=method,
    )
    new = run_opi_gas_phase(paths, ligand_fragment_ids, receptor_fragment_ids, method=method)

    try:
        mapping = permutation_to_align(
            legacy["fragment_ids"]["super"], new["fragment_ids"]["super"]
        )
    except FragmentAlignmentError as e:
        return {
            "overall_verdict": "FAIL",
            "error": str(e),
            "fragment_mapping": None,
        }

    target_ids = mapping.target_ids
    report = {
        "fixture_dir": str(fixture_dir),
        "method": method,
        "fragment_mapping": {
            "legacy_super_ids": list(mapping.source_ids),
            "opi_super_ids": list(new["fragment_ids"]["super"]),
            "target_order": list(target_ids),
            "permutation": list(mapping.permutation),
            "is_identity": mapping.is_identity,
        },
        "ligand_fragment_ids": list(ligand_fragment_ids),
        "receptor_fragment_ids": list(receptor_fragment_ids),
        "orca_out_paths": {k: str(v) for k, v in paths.items()},
        "new_path_led_sources": new.get("led_sources"),
        "tolerances": {
            "TOL_ENERGY_SCALAR_KJ_MOL": tol.TOL_ENERGY_SCALAR_KJ_MOL,
            "TOL_MATRIX_ELEMENT_KJ_MOL": tol.TOL_MATRIX_ELEMENT_KJ_MOL,
            "TOL_CONSERVATION_RESIDUAL_KJ_MOL": tol.TOL_CONSERVATION_RESIDUAL_KJ_MOL,
            "TOL_FP_EL_PREP_KJ_MOL": tol.TOL_FP_EL_PREP_KJ_MOL,
            "TOL_RELATIVE": tol.TOL_RELATIVE,
        },
        "step7": _compare_step7(legacy, new, target_ids),
        "step8": _compare_step8(legacy, new, target_ids, method=method),
    }
    report["parity_gates"] = _build_parity_gates(report)
    report["scientific_ledaw_removal_gate"] = _scientific_removal_gate(report["parity_gates"])
    report["overall_verdict"] = _overall_verdict({"step7": report["step7"], "step8": report["step8"]})
    return report


def write_report(report: Dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")


def human_summary(report: Dict[str, Any]) -> str:
    lines = [
        f"Overall: {report.get('overall_verdict', 'UNKNOWN')}",
        f"Fixture: {report.get('fixture_dir', '')}",
    ]
    if report.get("error"):
        lines.append(f"Error: {report['error']}")
        return "\n".join(lines)
    fm = report.get("fragment_mapping") or {}
    lines.append(f"Fragment mapping identity: {fm.get('is_identity')}")
    pg = report.get("parity_gates") or {}
    if pg:
        lines.append("\n[parity_gates]")
        for name, verdict in pg.items():
            if name.endswith("_note"):
                continue
            lines.append(f"  {name}: {verdict}")
    lines.append(f"\nScientific LEDAW removal gate: {report.get('scientific_ledaw_removal_gate', '?')}")
    for section in ("step7", "step8"):
        lines.append(f"\n[{section}]")
        for name, item in (report.get(section) or {}).items():
            if isinstance(item, dict) and "verdict" in item:
                lines.append(f"  {name}: {item.get('verdict', '?')} max_abs_diff={item.get('max_abs_diff', '—')}")
    return "\n".join(lines)
