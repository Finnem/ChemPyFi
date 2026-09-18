# LED (Local Energy Decomposition) Extraction

This module provides tools for extracting and processing LED matrices from ORCA calculations using the CovaLED methodology.

## ✅ Zero Setup Required!

**LEDAW is bundled with chempyfi** - just import and use!

### Simple Usage

```python
from chempyfi.orcautil import compute_led_interaction_matrix

# That's it - no path configuration needed!
```

### About Bundled LEDAW

LEDAW is included as a vendored dependency in `chempyfi._vendor.ledaw_package`:
- **No external installation required**
- **No path configuration needed**
- **Full credit to LEDAW original developers**
- See `chempyfi/_vendor/VENDOR_README.md` for details

This is a "vendoring" approach where we bundle LEDAW directly to avoid installation/compatibility issues

## Quick Start

```python
from chempyfi.orcautil import (
    compute_led_interaction_matrix,
    extract_interaction_energy
)

# Compute LED interaction matrix with auto-detected fragments
led_int = compute_led_interaction_matrix(
    orca_super="complex.out",
    orca_ligand="ligand.out",
    orca_receptor="receptor.out",
    ligand_atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
)

# Get total interaction energy
e_int = extract_interaction_energy(led_int)
print(f"E_int = {e_int:.4f} kJ/mol")
```

That's it! No setup, no configuration - LEDAW is bundled.

## Functions

### `compute_led_interaction_matrix()`
Main function to compute LED interaction matrices using CovaLED methodology.
- Auto-detects fragments from ORCA output
- Runs LEDAW automatically if Excel files don't exist
- Applies CovaLED Step 7 (subsystem subtraction)
- Returns interaction LED matrix

### `extract_interaction_energy()`
Calculate total interaction energy from LED matrix.

### `parse_orca_fragments()`
Parse fragment assignments from ORCA output file.

### `determine_fragment_groups()`
Determine which fragments belong to ligand vs receptor.

### `clear_subsystem_cache()`
Clear cached subsystem LED matrices to free memory.

## Examples

See `LED_USAGE_EXAMPLE.py` for complete examples including:
- Automatic fragment detection
- Manual fragment specification
- Parsing fragments independently
- Extracting multiple energy components
- Troubleshooting

## CovaLED Methodology

This module implements CovaLED (Covalent LED) from ORCA documentation:
https://www.faccts.de/docs/opi/nightly/docs/contents/notebooks/covaled_ethane.html

### CovaLED Step 7: Subsystem Subtraction
- Intra-ligand: `LED_int[i,j] = LED_super[i,j] - LED_ligand[i',j']`
- Intra-receptor: `LED_int[i,j] = LED_super[i,j] - LED_receptor[i',j']`
- Inter-molecular: `LED_int[i,j] = LED_super[i,j]` (pure interaction)

After Step 7: Sum of ALL terms = Total interaction energy

### fp-CovaLED Step 8: Redistribution
Redistributes intra-molecular contributions to inter-molecular pairs so that:
Sum of inter-molecular pairs ONLY = Total interaction energy

## Troubleshooting

### LEDAW Import Error
```
ERROR: LEDAW not available
Vendored LEDAW import failed: ...
```

**Cause:** Bundled LEDAW has incompatible compiled extensions for your system.

**Possible Reasons:**
- Python version incompatibility
- Missing system libraries (BLAS, LAPACK, MKL)
- Platform not supported by bundled LEDAW

**Solution:** Contact chempyfi maintainers or rebuild LEDAW for your system and replace the vendored copy in `chempyfi/_vendor/ledaw_package/`.

### Fragment Assignment Errors
```
ValueError: Fragment X contains atoms from both ligand and receptor!
```

**Cause:** ORCA fragment definitions span both molecules.

**Solutions:**
1. Use finer fragments in ORCA input (recommended)
2. Use `assign_mixed_fragments='majority'` to force assignment

### LEDAW Not Found
```
ImportError: LEDAW package not found
```

**Solution:** Verify LEDAW installation and add to `sys.path`.

## Advanced Options

### Verbose Mode
Enable detailed debugging output:
```python
led_int = compute_led_interaction_matrix(
    ...,
    verbose=True  # Shows checkpoints during LEDAW execution
)
```

### Handle Mixed Fragments
```python
led_int = compute_led_interaction_matrix(
    ...,
    assign_mixed_fragments='majority'  # 'error', 'majority', 'ligand', 'receptor'
)
```

### Force Regenerate LED Files
```python
led_int = compute_led_interaction_matrix(
    ...,
    force_regenerate=True  # Regenerate even if Excel files exist
)
```

### Extract Specific Components
```python
# Get electrostatic component only
elec_int = compute_led_interaction_matrix(
    ...,
    component='Electrostat'  # 'TOTAL', 'Exchange', 'Correlation', etc.
)
```

