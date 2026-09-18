import re

from chempyfi.interactions import Interactions
from chempyfi.rdutil import write_molecules

from .display import create_interaction_display
from .interface import PyMOLInterface


def get_first_parentheses_content(s):
    match = re.search(r"\(([^)]*)\)", s)
    return match.group(1) if match else None


def vizualize_ligand_interactions(selection, distance_threshold=8, only_enabled=True, update_graphics=True):
    """Visualize ligand–receptor interactions in the active PyMOL session."""
    from pymol import cmd

    ligand_interface = PyMOLInterface(selection, full_objects=False)
    receptor_selection = f"byres {selection} around {distance_threshold}"
    if only_enabled:
        receptor_selection += " and enabled"
    receptor_interface = PyMOLInterface(receptor_selection, full_objects=False)
    receptor_interactions = {
        object_name: Interactions(molecule) for object_name, molecule in receptor_interface.molecules.items()
    }

    for name, mol in ligand_interface.molecules.items():
        write_molecules(mol, name + "_ligand.sdf")

    interaction_visuals = {}
    found_interactions = {}
    for ligand_object_name, ligand_molecule in ligand_interface.molecules.items():
        for receptor_object_name, receptor_interaction in receptor_interactions.items():
            ligand_name = get_first_parentheses_content(ligand_object_name)
            receptor_name = get_first_parentheses_content(receptor_object_name)
            prefix = f"int_{ligand_name}_{receptor_name}"
            found = receptor_interaction.detect_interactions(ligand_molecule)
            found_interactions[(ligand_object_name, receptor_object_name)] = found
            interaction_visuals[(ligand_object_name, receptor_object_name)] = create_interaction_display(
                found, prefix=prefix
            )

    for interaction_visual in interaction_visuals.values():
        interaction_visual.load()
        interaction_visual.write(interaction_visual.name + ".py")

    if update_graphics:
        cmd.hide("everything", f"({selection}) or (bymol {receptor_selection})")
        cmd.show("sticks", selection)
        cmd.show("cartoon", f"bymol {receptor_selection}")
        for (ligand_object_name, receptor_object_name), interactions in found_interactions.items():
            for interaction_type, this_interactions in interactions.items():
                if not isinstance(this_interactions, list):
                    continue
                for interaction in this_interactions:
                    atom_indices = interaction[0]
                    try:
                        atom_indices[0]
                    except TypeError:
                        atom_indices = [atom_indices]
                    pymol_query = receptor_interface.get_pymol_query(receptor_object_name, atom_indices)
                    cmd.pseudoatom(
                        "residue_labels",
                        f"{pymol_query}",
                        label=f"{cmd.get_model(pymol_query).atom[0].resn} {cmd.get_model(pymol_query).atom[0].resi}",
                    )
                    if cmd.select(f"({pymol_query}) and backbone"):
                        cmd.show("sticks", f"(({pymol_query}) extend 2) and backbone")
                    else:
                        cmd.show("sticks", f"((byres ({pymol_query})) and sidechain) extend 1")
        cmd.color("wheat", "polymer and e. C")
        cmd.set("float_labels", 1)
        cmd.set("label_bg_color", "white")
        cmd.set("label_font_id", 5)
        cmd.hide("everything", "e. H and (neighbor e. C)")
        cmd.zoom(selection, animate=-1)
