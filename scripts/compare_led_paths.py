#!/usr/bin/env python3
"""
Compare legacy LEDAW vs OPI + chempyfi_covaled gas-phase CovaLED paths (developer oracle).

Usage:
  python scripts/compare_led_paths.py PATH_TO_FIXTURE_DIR
  CHEMPYFI_ORCA_LED_FIXTURE_DIR=... python scripts/compare_led_paths.py

Fixture directory must contain (default names):
  super.out, super.out.property.json
  ligand.out, ligand.out.property.json
  receptor.out, receptor.out.property.json

Does not commit proprietary ORCA outputs to the repository.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "module"))

from chempyfi_covaled.equivalence.harness import compare_gas_phase_led_paths, human_summary, write_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "fixture_dir",
        nargs="?",
        default=os.environ.get("CHEMPYFI_ORCA_LED_FIXTURE_DIR"),
        help="Directory with super/ligand/receptor ORCA outputs + .property.json sidecars",
    )
    parser.add_argument("--ligand-fragments", required=True, help="Comma-separated 1-based supersystem fragment IDs")
    parser.add_argument("--receptor-fragments", required=True, help="Comma-separated 1-based supersystem fragment IDs")
    parser.add_argument("--method", default="DLPNO-CCSD(T)")
    parser.add_argument("--super", default="super.out", help="Supersystem .out filename in fixture dir")
    parser.add_argument("--ligand", default="ligand.out")
    parser.add_argument("--receptor", default="receptor.out")
    parser.add_argument(
        "--auto-discover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use nested super/sub1/sub2 layout when flat .out files are absent",
    )
    parser.add_argument("--force-ledaw", action="store_true", help="Regenerate LEDAW Excel even if present")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--report-json",
        default="led_equivalence_report.json",
        help="Output JSON report path (default: cwd)",
    )
    parser.add_argument("--summary", action="store_true", help="Print human-readable summary to stdout")
    args = parser.parse_args()

    if not args.fixture_dir:
        parser.error("fixture_dir required or set CHEMPYFI_ORCA_LED_FIXTURE_DIR")

    lig = tuple(int(x.strip()) for x in args.ligand_fragments.split(",") if x.strip())
    rec = tuple(int(x.strip()) for x in args.receptor_fragments.split(",") if x.strip())

    report = compare_gas_phase_led_paths(
        Path(args.fixture_dir),
        lig,
        rec,
        method=args.method,
        force_ledaw=args.force_ledaw,
        verbose=args.verbose,
        super_name=args.super,
        ligand_name=args.ligand,
        receptor_name=args.receptor,
        auto_discover=args.auto_discover,
    )
    write_report(report, Path(args.report_json))
    print(f"Wrote {args.report_json}")
    if args.summary or report.get("overall_verdict") != "PASS":
        print(human_summary(report))
    verdict = report.get("overall_verdict")
    return 0 if verdict in ("PASS", "PASS_WITH_GAPS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
