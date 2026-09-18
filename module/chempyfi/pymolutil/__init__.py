"""Deprecated PyMOL integration path; prefer ``chempyfi_pymol``."""


def __getattr__(name):
    import chempyfi_pymol as pymol_pkg

    return getattr(pymol_pkg, name)


def __dir__():
    import chempyfi_pymol as pymol_pkg

    return sorted(dir(pymol_pkg))
