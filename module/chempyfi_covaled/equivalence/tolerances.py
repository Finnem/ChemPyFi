"""
Documented tolerances for legacy (LEDAW Excel) vs OPI gas-phase LED equivalence.

Excel matrices are typically written with finite precision after Hartree→kJ/mol
(``conversion_factor=2625.5``). OPI converts per-element from JSON floats.
"""

# Absolute tolerance on energy-like scalars and matrix elements (kJ/mol).
TOL_ENERGY_SCALAR_KJ_MOL = 0.05
TOL_MATRIX_ELEMENT_KJ_MOL = 0.05

# Step 8 conservation: sum(fp inter pairs) vs sum(step7 matrix) (kJ/mol).
TOL_CONSERVATION_RESIDUAL_KJ_MOL = 0.1

# Relative tolerance when |reference| > MIN_DENOM_KJ_MOL.
TOL_RELATIVE = 1e-4
MIN_DENOM_KJ_MOL = 1e-6

# fp-EL-PREP redistribution (same formula both paths; tighter).
TOL_FP_EL_PREP_KJ_MOL = 1e-6
