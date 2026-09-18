"""ChemPyFi: RDKit-based molecular toolkit (lightweight root import)."""

__version__ = "0.1"

# Subpackages loaded only on attribute access (``import chempyfi; chempyfi.rdutil``).
_LAZY_SUBMODULES = frozenset(
    {
        "rdutil",
        "morgan",
        "mathutils",
        "geometry",
        "fragmentation",
        "modeling",
        "interactions",
        "pymolutil",
        "orcautil",
    }
)

# Order for resolving ``from chempyfi import some_symbol`` without loading ORCA/PyMOL first.
_COMPAT_SEARCH_ORDER = (
    "mathutils",
    "geometry",
    "fragmentation",
    "modeling",
    "morgan",
    "rdutil",
    "interactions",
    "orcautil",
    "pymolutil",
)


def __getattr__(name):
    import importlib

    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f".{name}", __name__)

    for subname in _COMPAT_SEARCH_ORDER:
        submodule = importlib.import_module(f".{subname}", __name__)
        if hasattr(submodule, name):
            return getattr(submodule, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(_LAZY_SUBMODULES) + ["__version__"])
