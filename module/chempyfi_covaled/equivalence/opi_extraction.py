"""OPI extraction parity vs trusted reference (no LEDAW, no excel_bridge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from ..analysis import compute_covaled_from_system, interaction_energy_from_matrix
from ..led_data import LEDData, LEDSystemData
from ..opi_led import extract_led_data
from . import metrics
from . import tolerances as tol
from .fixtures import discover_fixture_orca_outputs
from .fragment_align import FragmentAlignmentError, permutation_to_align


def _load_reference_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _reference_led_for_role(ref: Dict[str, Any], role: str) -> LEDData:
    key = {"super": "super", "ligand": "ligand", "receptor": "receptor"}[role]
    block = ref["roles"][key]
    ids = tuple(block["fragment_ids"])
    comps = block
    from ..led_data import LEDComponents

    def _mat(name):
        if name not in block:
            return None
        return np.asarray(block[name], dtype=float)

    components = LEDComponents(
        reference=_mat("refint"),
        correlation=_mat("corrint"),
        electrostatics=_mat("electrostref"),
        exchange=_mat("exchangeref"),
    )
    return LEDData(
        fragment_ids=ids,
        total=_mat("totint"),
        energy_unit=ref.get("energy_unit", "kJ/mol"),
        components=components,
    )


def _compare_led_data_to_reference(opi: LEDData, ref: LEDData, *, role: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": role, "checks": {}}
    out["checks"]["fragment_ids"] = {
        "opi": list(opi.fragment_ids),
        "reference": list(ref.fragment_ids),
        "verdict": "PASS" if opi.fragment_ids == ref.fragment_ids else "FAIL",
    }
    out["checks"]["n_fragments"] = metrics.compare_scalars(
        opi.n_fragments, ref.n_fragments, atol=0, rtol=0, name=f"{role}_n_fragments"
    )
    out["checks"]["energy_unit"] = {
        "opi": opi.energy_unit,
        "reference": ref.energy_unit,
        "verdict": "PASS" if opi.energy_unit == ref.energy_unit else "FAIL",
    }
    if opi.fragment_ids == ref.fragment_ids:
        out["checks"]["total_matrix"] = metrics.compare_matrices(
            ref.total, opi.total, name=f"{role}_totint"
        )
        for attr, ref_arr, opi_arr in [
            ("reference", ref.components.reference, opi.components.reference),
            ("correlation", ref.components.correlation, opi.components.correlation),
            ("electrostatics", ref.components.electrostatics, opi.components.electrostatics),
            ("exchange", ref.components.exchange, opi.components.exchange),
        ]:
            if ref_arr is None and opi_arr is None:
                continue
            if ref_arr is None or opi_arr is None:
                out["checks"][attr] = {"verdict": "FAIL", "reason": "component_missing_on_one_side"}
                continue
            out["checks"][attr] = metrics.compare_matrices(ref_arr, opi_arr, name=f"{role}_{attr}")
    else:
        out["checks"]["total_matrix"] = {"verdict": "NOT_TESTED", "reason": "fragment_id_mismatch"}
    fails = [k for k, v in out["checks"].items() if isinstance(v, dict) and v.get("verdict") == "FAIL"]
    out["verdict"] = "FAIL" if fails else "PASS"
    return out


def compare_opi_extraction_fixture(
    fixture_dir: Path,
    *,
    reference_json: Path | None = None,
    method: str = "DLPNO-CCSD(T)",
) -> Dict[str, Any]:
    """
    Force OPI-only extraction; compare raw ``LEDData`` and Step 7 to ``reference_led_matrices.json``.
    """
    fixture_dir = Path(fixture_dir)
    reference_json = reference_json or (fixture_dir / "reference_led_matrices.json")
    ref_doc = _load_reference_json(reference_json)
    paths = discover_fixture_orca_outputs(fixture_dir)

    for role, p in paths.items():
        prop = Path(str(p) + ".property.json")
        if not prop.exists():
            raise FileNotFoundError(f"OPI extraction gate requires .property.json for {role}: {prop}")

    led_sources = {}
    opi_led: Dict[str, LEDData] = {}
    for role, p in paths.items():
        opi_led[role] = extract_led_data(p)
        led_sources[role] = "opi"

    if any(v != "opi" for v in led_sources.values()):
        raise RuntimeError(f"Expected all sources 'opi', got {led_sources}")

    raw_parity = {}
    for role in ("super", "ligand", "receptor"):
        ref_led = _reference_led_for_role(ref_doc, role)
        raw_parity[role] = _compare_led_data_to_reference(opi_led[role], ref_led, role=role)

    part = ref_doc["covaled_partition"]
    lig = tuple(part["ligand_fragment_ids"])
    rec = tuple(part["receptor_fragment_ids"])
    system = LEDSystemData(
        supersystem=opi_led["super"],
        ligand=opi_led["ligand"],
        receptor=opi_led["receptor"],
        ligand_fragment_ids=lig,
        receptor_fragment_ids=rec,
    )
    system.validate_partition()
    step7_df = compute_covaled_from_system(system)

    # Reference Step 7 from reference LEDData (trusted, not OPI self-comparison).
    ref_system = LEDSystemData(
        supersystem=_reference_led_for_role(ref_doc, "super"),
        ligand=_reference_led_for_role(ref_doc, "ligand"),
        receptor=_reference_led_for_role(ref_doc, "receptor"),
        ligand_fragment_ids=lig,
        receptor_fragment_ids=rec,
    )
    ref_step7 = compute_covaled_from_system(ref_system)
    target_ids = tuple(ref_system.supersystem.fragment_ids)
    step7_report = {
        "interaction_matrix": metrics.compare_dataframes_by_labels(
            ref_step7, step7_df, target_ids=target_ids, name="step7_interaction_matrix"
        ),
        "total_interaction_energy": metrics.compare_scalars(
            interaction_energy_from_matrix(ref_step7),
            interaction_energy_from_matrix(step7_df),
            name="total_interaction_energy",
        ),
    }

    try:
        mapping = permutation_to_align(
            tuple(int(i) for i in ref_step7.index),
            tuple(int(i) for i in step7_df.index),
        )
        fragment_order_ok = mapping.is_identity
    except FragmentAlignmentError:
        fragment_order_ok = False

    raw_ok = all(raw_parity[r]["verdict"] == "PASS" for r in raw_parity)
    step7_ok = step7_report["interaction_matrix"]["verdict"] == "PASS" and step7_report[
        "total_interaction_energy"
    ]["verdict"] == "PASS"

    opi_version = None
    orca_pi_version = None
    try:
        import opi

        orca_pi_version = getattr(opi, "__version__", None)
    except ImportError:
        pass

    report = {
        "fixture_dir": str(fixture_dir),
        "method": method,
        "new_path_led_sources": led_sources,
        "reference_json": str(reference_json),
        "raw_led_parity": raw_parity,
        "step7": step7_report,
        "opi_raw_led_parity_verdict": "PASS" if raw_ok else "FAIL",
        "opi_fragment_order_parity_verdict": "PASS" if fragment_order_ok else "FAIL",
        "opi_extraction_parity_verdict": "PASS"
        if (raw_ok and fragment_order_ok and step7_ok and all(v == "opi" for v in led_sources.values()))
        else "FAIL",
        "orca_pi_version": orca_pi_version,
        "opi_version_note": "Record ORCA engine version separately when using live calculations",
    }
    report["parity_gates"] = {
        "OPI_RAW_LED_PARITY": report["opi_raw_led_parity_verdict"],
        "OPI_FRAGMENT_ORDER_PARITY": report["opi_fragment_order_parity_verdict"],
        "OPI_EXTRACTION_PARITY": report["opi_extraction_parity_verdict"],
        "STEP7_MATRIX_PARITY": step7_report["interaction_matrix"].get("verdict", "NOT_TESTED"),
        "SCALAR_INTERACTION_ENERGY_PARITY": step7_report["total_interaction_energy"].get(
            "verdict", "NOT_TESTED"
        ),
    }
    report["opi_extraction_gate"] = report["parity_gates"]["OPI_EXTRACTION_PARITY"]
    report["ledaw_removal_gate"] = (
        "PASS"
        if report["opi_extraction_gate"] == "PASS"
        else "NOT_SATISFIED (scientific gate on 5L4Q is separate)"
    )
    return report
