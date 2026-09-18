#!/usr/bin/env python3
"""Generate covaled_ethane_mini OPI fixture (synthetic CovaLED tutorial-scale data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
KJ = 2625.5
STUB = "****ORCA OUTPUT FILE****\n(covaled_ethane_mini synthetic stub for OPI tests)\n"


def _prop(nfrag: int, tot, ref, corr, elec, exch, fragments):
    return {
        "geometries": [
            {
                "geometry": {"natoms": max(2, nfrag * 2), "fragments": fragments},
                "mdci_led": {
                    "numoffragments": nfrag,
                    "totint": tot,
                    "refint": ref,
                    "corrint": corr,
                    "electrostref": elec,
                    "exchangeref": exch,
                },
            }
        ]
    }


def _write_role(role_dir: Path, prop: dict) -> None:
    role_dir.mkdir(parents=True, exist_ok=True)
    out = role_dir / "orca.out"
    out.write_text(STUB)
    (Path(str(out) + ".property.json")).write_text(json.dumps(prop, indent=2) + "\n")


def main() -> None:
    # Supersystem: fragments 1 (ligand) and 2 (receptor) — Hartree, OPI tutorial scale.
    super_tot = [[-0.10, 0.003807], [0.003807, -0.20]]
    super_ref = [[-0.04, 0.001526], [0.001526, -0.08]]
    super_corr = [[-0.03, 0.002281], [0.002281, -0.06]]
    super_elec = [[-0.02, 0.001], [0.001, -0.03]]
    super_exch = [[-0.05, 0.0005], [0.0005, -0.11]]
    frags_super = [[1], [1], [2], [2]]
    _write_role(
        ROOT / "super",
        _prop(2, super_tot, super_ref, super_corr, super_elec, super_exch, frags_super),
    )

    _write_role(
        ROOT / "sub2",
        _prop(1, [[-0.08]], [[-0.03]], [[-0.02]], [[-0.01]], [[-0.02]], [[1]]),
    )
    _write_role(
        ROOT / "sub1",
        _prop(1, [[-0.15]], [[-0.06]], [[-0.04]], [[-0.02]], [[-0.03]], [[1]]),
    )

    def ha2kj(m):
        return (np.asarray(m, dtype=float) * KJ).tolist()

    reference = {
        "description": "Trusted kJ/mol reference derived from committed .property.json Hartree values",
        "energy_unit": "kJ/mol",
        "orca_pi_fixture": "synthetic; not a licensed production run",
        "roles": {
            "super": {
                "fragment_ids": [1, 2],
                "totint": ha2kj(super_tot),
                "refint": ha2kj(super_ref),
                "corrint": ha2kj(super_corr),
                "electrostref": ha2kj(super_elec),
                "exchangeref": ha2kj(super_exch),
            },
            "ligand": {
                "fragment_ids": [1],
                "totint": ha2kj([[-0.08]]),
                "refint": ha2kj([[-0.03]]),
                "corrint": ha2kj([[-0.02]]),
                "electrostref": ha2kj([[-0.01]]),
                "exchangeref": ha2kj([[-0.02]]),
            },
            "receptor": {
                "fragment_ids": [1],
                "totint": ha2kj([[-0.15]]),
                "refint": ha2kj([[-0.06]]),
                "corrint": ha2kj([[-0.04]]),
                "electrostref": ha2kj([[-0.02]]),
                "exchangeref": ha2kj([[-0.03]]),
            },
        },
        "covaled_partition": {"ligand_fragment_ids": [1], "receptor_fragment_ids": [2]},
    }
    (ROOT / "reference_led_matrices.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Wrote {ROOT}")


if __name__ == "__main__":
    main()
