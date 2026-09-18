#!/usr/bin/env bash
# Install ChemPyFi into the active env without replacing conda RDKit with rdkit-pypi.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
pip install --no-deps -e ".[dev,bondorder]"
echo "Installed chempyfi editable from $ROOT (RDKit should come from conda)."
