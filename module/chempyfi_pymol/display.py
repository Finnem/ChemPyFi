"""Render precomputed ChemPyFi interaction results with pymolviz (no redetection)."""

from pathlib import Path

HBOND_COLOR = "#4270D4"
BAD_HBOND_COLOR = "#4F87FF"
PI_PI_COLOR = "#000000"
CATION_DIPOLE_COLOR = "#3F48CC"
ANION_DIPOLE_COLOR = "#3F48CC"
SALT_COLOR = "#FF0000"


def _merge_self_lists(primary, additional, key):
    out = list(primary.get(key, []))
    if additional:
        out.extend(additional.get(key, []))
    return out


def interaction_point_specs(interactions, additional_interactions=None):
    """
    Renderer-neutral endpoint specs for optional point-cloud visualization.

    Each spec is ``(position, role_label)`` where ``position`` is a 3D coordinate from detection.
    """
    specs = []

    def _add(items, role):
        for row in items:
            if len(row) >= 4:
                specs.append((row[2], f"{role}_receptor"))
                specs.append((row[3], f"{role}_ligand"))

    _add(_merge_self_lists(interactions, additional_interactions, "hydrogen_bond_donor_interactions"), "hbond_donor")
    _add(_merge_self_lists(interactions, additional_interactions, "hydrogen_bond_acceptor_interactions"), "hbond_acceptor")
    _add(_merge_self_lists(interactions, additional_interactions, "bad_hydrogen_bond_donor_interactions"), "bad_hbond_donor")
    _add(_merge_self_lists(interactions, additional_interactions, "bad_hydrogen_bond_acceptor_interactions"), "bad_hbond_acceptor")
    _add(_merge_self_lists(interactions, additional_interactions, "cation_dipole_interactions"), "cation_dipole")
    _add(_merge_self_lists(interactions, additional_interactions, "dipole_cation_interactions"), "dipole_cation")
    _add(_merge_self_lists(interactions, additional_interactions, "anion_dipole_interactions"), "anion_dipole")
    _add(_merge_self_lists(interactions, additional_interactions, "dipole_anion_interactions"), "dipole_anion")
    _add(_merge_self_lists(interactions, additional_interactions, "anion_cation_interactions"), "salt_anion")
    _add(_merge_self_lists(interactions, additional_interactions, "cation_anion_interactions"), "salt_cation")
    _add(_merge_self_lists(interactions, additional_interactions, "pi_stacking_interactions"), "pi_stack")
    return specs


def _append_point_clouds(all_visuals, pmv, items, color, name_prefix, interaction_points):
    if not interaction_points or not items:
        return
    all_visuals.append(
        pmv.Points([i[2] for i in items], color=color, transparency=0.9, name=name_prefix + " Receptor")
    )
    all_visuals.append(
        pmv.Points([i[3] for i in items], color=color, transparency=0.9, name=name_prefix + " Ligand")
    )


def create_interaction_display(
    interactions,
    prefix="interactions",
    additional_interactions=None,
    interaction_points=False,
):
    """Build a pymolviz Group from a detection result dict."""
    import pymolviz as pmv

    all_visuals = []
    add = additional_interactions

    hbond_d = _merge_self_lists(interactions, add, "hydrogen_bond_donor_interactions")
    hbond_a = _merge_self_lists(interactions, add, "hydrogen_bond_acceptor_interactions")
    hbond_points = [[i[2], i[3]] for i in hbond_d] + [[i[2], i[3]] for i in hbond_a]
    if hbond_points:
        all_visuals.append(pmv.Lines(hbond_points, color=HBOND_COLOR, name=prefix + " Hydrogen Bonds").as_dotted())
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, hbond_d, HBOND_COLOR, prefix + " Hydrogen Bond Donors", True)
            _append_point_clouds(all_visuals, pmv, hbond_a, HBOND_COLOR, prefix + " Hydrogen Bond Acceptors", True)

    bad_d = _merge_self_lists(interactions, add, "bad_hydrogen_bond_donor_interactions")
    bad_a = _merge_self_lists(interactions, add, "bad_hydrogen_bond_acceptor_interactions")
    bad_hbond_points = [[i[2], i[3]] for i in bad_d] + [[i[2], i[3]] for i in bad_a]
    if bad_hbond_points:
        all_visuals.append(
            pmv.Lines(bad_hbond_points, color=BAD_HBOND_COLOR, name=prefix + " Bad Hydrogen Bonds", linewidth=0.035).as_dotted()
        )
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, bad_d, BAD_HBOND_COLOR, prefix + " Bad Hydrogen Bond Donors", True)
            _append_point_clouds(all_visuals, pmv, bad_a, BAD_HBOND_COLOR, prefix + " Bad Hydrogen Bond Acceptors", True)

    cation_d = _merge_self_lists(interactions, add, "cation_dipole_interactions")
    cation_c = _merge_self_lists(interactions, add, "dipole_cation_interactions")
    cation_dipole_points = [[i[2], i[3]] for i in cation_d] + [[i[2], i[3]] for i in cation_c]
    if cation_dipole_points:
        all_visuals.append(
            pmv.Lines(cation_dipole_points, color=CATION_DIPOLE_COLOR, name=prefix + " Cation Dipole Interactions", linewidth=0.065).as_dotted()
        )
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, cation_d, CATION_DIPOLE_COLOR, prefix + " Cation Dipole", True)
            _append_point_clouds(all_visuals, pmv, cation_c, CATION_DIPOLE_COLOR, prefix + " Dipole Cation", True)

    anion_d = _merge_self_lists(interactions, add, "anion_dipole_interactions")
    anion_c = _merge_self_lists(interactions, add, "dipole_anion_interactions")
    anion_dipole_points = [[i[2], i[3]] for i in anion_d] + [[i[2], i[3]] for i in anion_c]
    if anion_dipole_points:
        all_visuals.append(
            pmv.Lines(anion_dipole_points, color=ANION_DIPOLE_COLOR, name=prefix + " Anion Dipole Interactions", linewidth=0.065).as_dotted()
        )
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, anion_d, ANION_DIPOLE_COLOR, prefix + " Anion Dipole", True)
            _append_point_clouds(all_visuals, pmv, anion_c, ANION_DIPOLE_COLOR, prefix + " Dipole Anion", True)

    salt_a = _merge_self_lists(interactions, add, "anion_cation_interactions")
    salt_c = _merge_self_lists(interactions, add, "cation_anion_interactions")
    anion_cation_points = [[i[2], i[3]] for i in salt_a] + [[i[2], i[3]] for i in salt_c]
    if anion_cation_points:
        all_visuals.append(
            pmv.Lines(anion_cation_points, color=SALT_COLOR, name=prefix + " Salt Bridges", linewidth=0.08).as_dotted()
        )
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, salt_a, SALT_COLOR, prefix + " Salt Bridge Anion/Cation", True)
            _append_point_clouds(all_visuals, pmv, salt_c, SALT_COLOR, prefix + " Salt Bridge Cation/Anion", True)

    pi_items = _merge_self_lists(interactions, add, "pi_stacking_interactions")
    pi_stacking_points = [[i[2], i[3]] for i in pi_items]
    if pi_stacking_points:
        all_visuals.append(
            pmv.Lines(pi_stacking_points, color=PI_PI_COLOR, name=prefix + " Pi Stacking Interactions", linewidth=0.1).as_dotted()
        )
        if interaction_points:
            _append_point_clouds(all_visuals, pmv, pi_items, PI_PI_COLOR, prefix + " Pi Stacking", True)

    return pmv.Group(all_visuals, name=prefix + " Interactions_Group")


def show_interactions(interactions, prefix="interactions", **kwargs):
    visual = create_interaction_display(interactions, prefix=prefix, **kwargs)
    visual.load()
    return visual


def write_interaction_display(interactions, filename, prefix=None, **kwargs):
    if prefix is None:
        prefix = Path(filename).stem
    create_interaction_display(interactions, prefix=prefix, **kwargs).write(filename)
