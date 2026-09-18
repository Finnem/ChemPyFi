# ChemPyFi

ChemPyFi is an **RDKit-based Python toolkit** for building local molecular models, detecting non-covalent interactions, fragmenting structures for quantum-chemistry workflows, and analyzing **CovaLED** / **fp-CovaLED** results from [ORCA](https://orcaforum.kofo.mpg.de/) (via [OPI](https://github.com/theochem/orca-pi) where applicable).

**ChemPyFi is a cleaned-up successor to the old [XBPy](https://github.com/Finnem/XBPy) repository.** The codebase was refactored into clearer packages, slimmer default dependencies, and explicit boundaries between core modeling, PyMOL visualization, and CovaLED/ORCA analysis. Legacy entry points under `chempyfi.orcautil` remain as compatibility shims where older XBPy scripts still import them.

## Packages

This monorepo ships **three installable distributions**; source lives under `module/`:

| Distribution | Import name | Purpose |
|--------------|-------------|---------|
| **chempyfi** | `chempyfi` | Core: RDKit I/O and geometry (`rdutil`), local modeling and fragmentation (`modeling`, `fragmentation`), interaction detection (`interactions`), Morgan fingerprints, optional bond-order inference |
| **chempyfi-pymol** | `chempyfi_pymol` | PyMOL / [pymolviz](https://github.com/) rendering of interaction specs (depends on core) |
| **chempyfi-covaled** | `chempyfi_covaled` | Gas-phase CovaLED Step 7 and fp-CovaLED Step 8, `LEDData`, OPI-backed ORCA I/O (depends on core) |

Dependency direction is one-way: optional packages depend on `chempyfi`; the core does not require PyMOL, OPI, or CovaLED at import time. `import chempyfi` only loads package metadata; submodules resolve lazily on first use.

More detail on the split and removed XBPy-era pieces is in [`docs/refactor/final_architecture.md`](docs/refactor/final_architecture.md).

## Requirements

- **Python ≥ 3.8** for `chempyfi` and `chempyfi-pymol`
- **Python ≥ 3.11** for `chempyfi-covaled`
- **RDKit** (conda-forge recommended for development)
- Optional: **z3-solver** (bond-order inference), **pymolviz** + PyMOL (visualization), **orca-pi ≥ 2** (ORCA input/output and LED extraction), **pandas** (CovaLED)

## Installation

### Conda (recommended for development)

```bash
conda env create -f environment.yml
conda activate chempyfi
./scripts/install_editable.sh
```

For optional LED/CovaLED Excel regression tooling used by developers:

```bash
conda env update -f environment-led.yml
```

### pip (core only)

From the repository root:

```bash
pip install -e ".[dev,bondorder]"
```

Optional extras on the root `pyproject.toml`:

- `viz` — pymolviz (or install `chempyfi-pymol` from `packages/chempyfi-pymol/`)
- `bondorder` — z3-solver
- `opi` / `covaled` — orca-pi and related analysis deps

CovaLED package (with OPI):

```bash
pip install -e "packages/chempyfi-covaled[opi]"
```

PyMOL adapters:

```bash
pip install -e packages/chempyfi-pymol/
```

## Quick start

**Local environment and interactions (core):**

```python
from chempyfi.modeling import extract_local_environment
from chempyfi.interactions import detect_interactions

# ... build or load RDKit mols, then detect interactions and extract a local region
```

**CovaLED analysis (optional package):**

```python
from chempyfi_covaled.led_data import LEDData
from chempyfi_covaled import compute_covaled, compute_fp_covaled
from chempyfi_covaled.pipeline import analyze_output

# Typical path: ORCA output + orca.out.property.json → LEDData → Step 7 / Step 8
```

**PyMOL visualization (optional package):**

```python
from chempyfi_pymol import show_interactions  # see chempyfi_pymol for full API
```

Legacy XBPy-style imports may still work via `chempyfi.orcautil` (delegating to `chempyfi_covaled` when CovaLED helpers are called).

## Repository layout

```text
module/
  chempyfi/           # core library
  chempyfi_covaled/   # CovaLED + OPI
  chempyfi_pymol/     # PyMOL visualization
packages/             # per-distribution pyproject.toml metadata
tests/                # pytest suite (unit, integration, architecture)
docs/refactor/        # architecture and dependency notes from the XBPy migration
scripts/              # developer utilities (e.g. LED path comparison)
```

## Testing

```bash
pytest
```

Common subsets:

```bash
# Core only (no OPI / PyMOL markers)
pytest tests/unit tests/characterization -m "not opi and not pymol"

# CovaLED unit tests
pytest -m "unit and chempyfi_covaled"

# OPI gate (requires orca-pi)
pytest -m opi
```

Markers are defined in `pytest.ini`. CI-oriented runs should not require a local ORCA binary (`orca` marker) or proprietary Excel regression fixtures (`chempyfi_covaled_oracle`).

Optional developer oracle data (e.g. the 5L4Q bundle) is not in the repository. Point tests and `scripts/compare_led_paths.py` at an external directory:

```bash
export CHEMPYFI_ORCA_LED_FIXTURE_DIR=/path/to/relax_5L4Q_full_pure_obj01_entry_00001_conf_01
```

## License

MIT — see package metadata in `pyproject.toml`. Author: Finn Mier.

## Related reading

- [`packages/chempyfi-covaled/README.md`](packages/chempyfi-covaled/README.md) — CovaLED install and API
- [`docs/refactor/final_architecture.md`](docs/refactor/final_architecture.md) — package boundaries and what changed vs. legacy XBPy
- [`docs/refactor/dependencies.md`](docs/refactor/dependencies.md) — import graph and lazy-loading design
