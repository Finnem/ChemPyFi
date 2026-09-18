"""Compatibility shim → ``chempyfi_covaled.analysis`` (lazy; keeps orcautil import light)."""

__all__ = [
    "compute_covaled_interaction_matrix",
    "compute_fp_covaled_interactions",
    "extract_interaction_energy",
]


def compute_covaled_interaction_matrix(*args, **kwargs):
    from chempyfi_covaled.analysis import compute_covaled_interaction_matrix as _fn

    return _fn(*args, **kwargs)


def compute_fp_covaled_interactions(*args, **kwargs):
    from chempyfi_covaled.analysis import compute_fp_covaled as _fn

    return _fn(*args, **kwargs)


def extract_interaction_energy(*args, **kwargs):
    from chempyfi_covaled.analysis import extract_interaction_energy as _fn

    return _fn(*args, **kwargs)
