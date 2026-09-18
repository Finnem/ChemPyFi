"""OPI → LEDData (sole ``import opi`` site in chempyfi_covaled)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .exceptions import MissingLEDDataError
from .led_data import LEDComponents, LEDData, as_led_matrix


class OpiNotAvailableError(ImportError):
    pass


def _require_opi():
    try:
        import opi  # noqa: F401
    except ImportError as e:
        raise OpiNotAvailableError(
            "orca-pi is required for extract_led_data. Install: pip install 'chempyfi[opi]'"
        ) from e


def _led_from_opi_object(
    led,
    fragment_ids: Tuple[int, ...],
    *,
    energy_unit: str,
    source_path: Optional[str] = None,
) -> LEDData:
    if led is None:
        raise MissingLEDDataError("mdci_led", source=source_path)
    n = int(led.numoffragments or len(fragment_ids))
    if len(fragment_ids) != n:
        fragment_ids = tuple(range(1, n + 1))

    def _comp(name):
        raw = getattr(led, name, None)
        if raw is None:
            return None
        return as_led_matrix(raw, fragment_ids, energy_unit="hartree")

    total = _comp("totint")
    if total is None:
        raise MissingLEDDataError("mdci_led.totint", source=source_path)

    components = LEDComponents(
        reference=_comp("refint"),
        correlation=_comp("corrint"),
        electrostatics=_comp("electrostref"),
        exchange=_comp("exchangeref"),
        dispersion_strong=_comp("dispcontr"),
        dispersion_weak=_comp("dispweak"),
    )
    unit = "kJ/mol" if energy_unit == "hartree" else energy_unit
    return LEDData(
        fragment_ids=fragment_ids,
        total=total,
        energy_unit=unit,
        components=components,
        source_path=source_path,
    )


def fragment_ids_from_geometry(geometry) -> Tuple[int, ...]:
    """Build ordered 1-based fragment IDs from OPI geometry.fragments (per-atom)."""
    if geometry is None or not getattr(geometry, "fragments", None):
        return tuple()
    per_atom = [int(row[0]) if row else 0 for row in geometry.fragments]
    unique = sorted({f for f in per_atom if f > 0})
    if unique:
        return tuple(unique)
    return tuple(range(1, max(per_atom) + 1)) if per_atom else tuple()


def extract_led_data(
    output: Union[str, Path, object],
    *,
    geometry_index: int = -1,
    energy_unit: str = "hartree",
    source_path: Optional[str] = None,
) -> LEDData:
    """
    Parse OPI ``Output`` (or path to ``.out`` with sidecar ``.property.json``) into ``LEDData``.
    """
    _require_opi()
    from opi.output.core import Output

    if not isinstance(output, Output):
        path = Path(output)
        prop = Path(str(path) + ".property.json")
        if not prop.exists():
            raise MissingLEDDataError(
                "orca.out.property.json",
                source=str(path),
            )
        output = Output(str(path))
        source_path = source_path or str(path)
        output.parse(read_prop_json=True, read_gbw_json=False, do_create_property_json=False)
    else:
        if output.property_json_data is None:
            output.parse(read_prop_json=True, read_gbw_json=False, do_create_property_json=False)

    props = output.results_properties
    if not props or not props.geometries:
        raise ValueError("OPI output contains no geometries")
    geo_entry = props.geometries[geometry_index]
    led = geo_entry.mdci_led
    geom = geo_entry.geometry
    frag_ids = fragment_ids_from_geometry(geom)
    if led and led.numoffragments:
        n = int(led.numoffragments)
        if len(frag_ids) != n:
            frag_ids = tuple(range(1, n + 1))
    return _led_from_opi_object(
        led, frag_ids, energy_unit=energy_unit, source_path=source_path
    )


def led_data_to_standard_matrices(led: LEDData) -> dict:
    """Build sheet-name → DataFrame map for fp-LED assembly (kJ/mol)."""
    import pandas as pd

    labels = list(led.fragment_ids)

    def _df(arr, sheet: str):
        if arr is None:
            return None
        return pd.DataFrame(arr, index=labels, columns=labels)

    mats = {"TOTAL": _df(led.total, "TOTAL")}
    c = led.components
    if c.reference is not None:
        mats["REF"] = _df(c.reference, "REF")
    if c.electrostatics is not None:
        mats["Electrostat"] = _df(c.electrostatics, "Electrostat")
    if c.exchange is not None:
        mats["Exchange"] = _df(c.exchange, "Exchange")
    if c.correlation is not None:
        mats["CORR"] = _df(c.correlation, "CORR")
    if c.dispersion_strong is not None:
        disp = _df(c.dispersion_strong, "Disp")
        if disp is not None:
            mats["Disp CCSD(T)"] = disp
    return {k: v for k, v in mats.items() if v is not None}


def extract_led_data_from_standard_excel(
    excel_path: Union[str, Path],
    *,
    source_path: Optional[str] = None,
) -> LEDData:
    """
    Build ``LEDData`` from LEDAW ``All_Standard_LED_matrices.xlsx`` (kJ/mol).

    Oracle bridge when ``.property.json`` is unavailable; not a substitute for OPI in production.
    """
    import pandas as pd

    excel_path = Path(excel_path)
    sheets = {}
    with pd.ExcelFile(excel_path) as xl:
        for name in xl.sheet_names:
            if name.upper().startswith("SOLV"):
                continue
            sheets[name] = pd.read_excel(xl, sheet_name=name, index_col=0)

    if "TOTAL" not in sheets:
        raise ValueError(f"No TOTAL sheet in {excel_path}")
    total_df = sheets["TOTAL"].copy()
    total_df.index = [int(i) for i in total_df.index]
    total_df.columns = [int(c) for c in total_df.columns]
    fragment_ids = tuple(total_df.index)

    def _arr(sheet_name: str):
        if sheet_name not in sheets:
            return None
        df = sheets[sheet_name].copy()
        df.index = [int(i) for i in df.index]
        df.columns = [int(c) for c in df.columns]
        return df.values.astype(float)

    disp = _arr("Disp CCSD(T)")
    if disp is None:
        disp = _arr("Disp CCSD")
    components = LEDComponents(
        reference=_arr("REF"),
        correlation=_arr("CORR") if "CORR" in sheets else _arr("Correlation"),
        electrostatics=_arr("Electrostat"),
        exchange=_arr("Exchange"),
        dispersion_strong=disp,
        dispersion_weak=None,
    )
    return LEDData(
        fragment_ids=fragment_ids,
        total=total_df.values.astype(float),
        energy_unit="kJ/mol",
        components=components,
        source_path=source_path or str(excel_path),
    )


def extract_led_data_for_out_file(
    out_path: Union[str, Path],
    *,
    require_opi: bool = False,
) -> Tuple[LEDData, str]:
    """Prefer OPI ``.property.json``; else standard Excel next to ``orca.out`` (oracle bridge only)."""
    out_path = Path(out_path)
    prop = Path(str(out_path) + ".property.json")
    if prop.exists():
        return extract_led_data(out_path), "opi"
    if require_opi:
        raise MissingLEDDataError(
            "orca.out.property.json",
            source=str(out_path),
        )
    excel = out_path.parent / "All_Standard_LED_matrices.xlsx"
    if excel.exists():
        return extract_led_data_from_standard_excel(excel, source_path=str(out_path)), "excel_bridge"
    raise FileNotFoundError(
        f"No OPI sidecar or All_Standard_LED_matrices.xlsx for {out_path}. "
        "Re-run ORCA with JSON property output enabled."
    )
