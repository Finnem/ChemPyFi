"""PyMOL visualization for ChemPyFi (lazy imports; pymolviz/PyMOL load on use)."""

__all__ = [
    "PyMOLInterface",
    "create_interaction_display",
    "show_interactions",
    "vizualize_ligand_interactions",
    "get_first_parentheses_content",
    "interaction_point_specs",
]

_LAZY = {
    "PyMOLInterface": (".interface", "PyMOLInterface"),
    "create_interaction_display": (".display", "create_interaction_display"),
    "show_interactions": (".display", "show_interactions"),
    "interaction_point_specs": (".display", "interaction_point_specs"),
    "vizualize_ligand_interactions": (".visualize", "vizualize_ligand_interactions"),
    "get_first_parentheses_content": (".visualize", "get_first_parentheses_content"),
}


def __getattr__(name):
    import importlib

    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = _LAZY[name]
    mod = importlib.import_module(mod_name, __name__)
    return getattr(mod, attr)


def __dir__():
    return sorted(__all__)
