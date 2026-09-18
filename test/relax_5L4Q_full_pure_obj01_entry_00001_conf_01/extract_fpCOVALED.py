# fp-LED / fp-CovaLED extraction using chempyfi
import sys
import numpy as np
import colorsys
import re
import os
import pandas as pd
from chempyfi.orcautil import (
    compute_fp_led_interactions_with_mapping,
    parse_orca_fragments,
    determine_fragment_groups,
)

# Add PyMolViz to path
sys.path.insert(0, '/home/highgarden/Workspace/Python_Libraries/PyMolViz/module')
import pymolviz as pmv
from pymolviz.Displayable import Displayable

SUPERSYSTEM_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/super/orca_2111303/orca.out"
SURR_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/sub1/orca_2111305/orca.out"
LIGAND_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/sub2/orca_2111304/orca.out"
XYZ_FILE = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/relax_5L4Q_full_pure_obj01_entry_00001_conf_01_LED_interaction_structure.xyz"
LIGAND_ATOM_INDICES = [277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294]


def build_fragment_mappings(orca_super, ligand_atom_indices):
    """Build LEDAW fragment mappings from supersystem atom assignments."""
    atom_to_fragment, _ = parse_orca_fragments(orca_super)
    ligand_frags, receptor_frags = determine_fragment_groups(
        atom_to_fragment, ligand_atom_indices
    )

    all_frags = sorted({frag_idx + 1 for frag_idx in atom_to_fragment})
    supersys = {frag_id: frag_id for frag_id in all_frags}
    subsys_ligand = {local_idx + 1: frag_id for local_idx, frag_id in enumerate(ligand_frags)}
    subsys_receptor = {local_idx + 1: frag_id for local_idx, frag_id in enumerate(receptor_frags)}

    return {
        "SUPERSYS": supersys,
        "SUBSYS1": subsys_ligand,
        "SUBSYS2": subsys_receptor,
    }, ligand_frags, receptor_frags


def fp_dict_to_dataframe(fp_interactions):
    """Convert fp-LED pair dict {'i_j': energy} to a symmetric DataFrame."""
    frag_ids = set()
    for key in fp_interactions:
        frag_i, frag_j = (int(part) for part in key.split("_"))
        frag_ids.add(frag_i)
        frag_ids.add(frag_j)

    frag_ids = sorted(frag_ids)
    led_int = pd.DataFrame(np.nan, index=frag_ids, columns=frag_ids, dtype=float)
    for key, value in fp_interactions.items():
        frag_i, frag_j = (int(part) for part in key.split("_"))
        led_int.loc[frag_i, frag_j] = value
        led_int.loc[frag_j, frag_i] = value

    return led_int


def upper_triangle_dataframe(df):
    """Mask lower triangle (keep upper triangle including diagonal)."""
    result = df.copy()
    frag_ids = list(result.index)
    for row_id in frag_ids:
        for col_id in frag_ids:
            if row_id > col_id:
                result.loc[row_id, col_id] = np.nan
    return result


def filter_insignificant_columns(df, threshold=1.0):
    """Drop columns where no individual cell has |energy| > threshold."""
    keep_cols = [col for col in df.columns if df[col].abs().gt(threshold).any()]
    return df.loc[:, keep_cols]


def zero_small_values(df, threshold=5.0):
    """Set values with |energy| < threshold to 0 (NaN entries unchanged)."""
    result = df.copy()
    small = result.notna() & result.abs().lt(threshold)
    return result.mask(small, 0.0)


def get_fp_interaction(fp_interactions, frag_a, frag_b):
    """Look up fp-LED interaction between two fragments."""
    frag_i, frag_j = sorted((int(frag_a), int(frag_b)))
    return fp_interactions.get(f"{frag_i}_{frag_j}")


fragment_mappings, ligand_frags, receptor_frags = build_fragment_mappings(
    SUPERSYSTEM_OUT, LIGAND_ATOM_INDICES
)
print("Fragment mappings:")
for label, mapping in fragment_mappings.items():
    print(f"  {label}: {mapping}")
print(f"  Ligand fragments: {ligand_frags}")
print(f"  Receptor fragments: {receptor_frags}")

print("\nComputing fp-LED interactions (N-body with subsystem mapping)...")
fp_interactions = compute_fp_led_interactions_with_mapping(
    main_filenames=[SUPERSYSTEM_OUT, LIGAND_OUT, SURR_OUT],
    alternative_filenames=["", "", ""],
    fragment_mappings=fragment_mappings,
    verbose=True,
    use_temp_dir=True,
    use_dataframes=True,
)

led_int = fp_dict_to_dataframe(fp_interactions)
led_int_upper = upper_triangle_dataframe(led_int)
led_int_filtered = filter_insignificant_columns(led_int_upper, threshold=1.0)
led_int_zeroed = zero_small_values(led_int_upper, threshold=5.0)
print(led_int_upper)

max_abs = float(led_int_upper.abs().max().max())
n_above_threshold = int((led_int_upper.abs() > 1.0).sum().sum())
n_above_zero_cutoff = int((led_int_upper.abs() >= 5.0).sum().sum())
print(f"\nMatrix stats: max |value| = {max_abs:.4f} kJ/mol, entries with |value| > 1: {n_above_threshold}")
print(f"  entries with |value| >= 5 (kept non-zero in zeroed sheet): {n_above_zero_cutoff}")

OUTPUT_EXCEL = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/relax_5L4Q_full_pure_obj01_entry_00001_conf_01_fpLED_interaction.xlsx"
with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
    led_int_upper.to_excel(writer, sheet_name="fpLED_Interaction")
    led_int_filtered.to_excel(writer, sheet_name="fpLED_Interaction_filtered")
    led_int_zeroed.to_excel(writer, sheet_name="fpLED_Interaction_zeroed")
print(f"\nfp-LED interaction matrix saved to: {OUTPUT_EXCEL}")
print(f"  Sheet 'fpLED_Interaction': upper triangle ({len(led_int_upper)} fragments)")
if len(led_int_filtered.columns) == 0:
    print("  Sheet 'fpLED_Interaction_filtered': empty (no column has |value| > 1)")
else:
    print(f"  Sheet 'fpLED_Interaction_filtered': {len(led_int_filtered.columns)} columns with |value| > 1")
print(f"  Sheet 'fpLED_Interaction_zeroed': |value| < 5 set to 0 ({n_above_zero_cutoff} non-zero entries)")
PMV_NAME = os.path.splitext(os.path.basename(OUTPUT_EXCEL))[0]

print("\nExtracting total interaction energy...")
e_int = float(sum(fp_interactions.values()))
print(f"\n{'='*80}")
print(f"TOTAL fp-LED INTERACTION ENERGY: {e_int:.4f} kJ/mol")
print(f"{'='*80}")

print("\n" + "="*80)
print("Creating PyMOL visualization with fragment isomeshes...")
print("="*80)


def parse_fragment_assignments(orca_output):
    """Parse fragment assignments from ORCA output file."""
    with open(orca_output, 'r') as f:
        lines = f.readlines()

    fragment_assignments = []
    in_coord_section = False

    for line in lines:
        if 'CARTESIAN COORDINATES (A.U.)' in line:
            in_coord_section = True
            continue

        if in_coord_section:
            match = re.match(r'\s+(\d+)\s+\w+\s+[\d.]+\s+(\d+)\s+[\d.]+', line)
            if match:
                atom_idx = int(match.group(1))
                frag_idx = int(match.group(2))
                fragment_assignments.append((atom_idx, frag_idx))
            elif line.strip().startswith('---') or line.strip() == '':
                if fragment_assignments:
                    break

    fragments = {}
    for atom_idx, frag_idx in fragment_assignments:
        fragments.setdefault(frag_idx, []).append(atom_idx)

    return fragments


def parse_xyz(filename):
    """Parse XYZ file and return atom positions."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    n_atoms = int(lines[0].strip())
    atoms = []
    positions = []

    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return np.array(positions), atoms


def create_gaussian_grid(positions, grid_spacing=0.4, padding=1.5, sigma=0.8):
    """Create a 3D grid with Gaussian density around atom positions."""
    positions = np.array(positions)

    min_coords = positions.min(axis=0) - padding
    max_coords = positions.max(axis=0) + padding

    x = np.arange(min_coords[0], max_coords[0], grid_spacing)
    y = np.arange(min_coords[1], max_coords[1], grid_spacing)
    z = np.arange(min_coords[2], max_coords[2], grid_spacing)

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    density = np.zeros(len(grid_points))
    for pos in positions:
        dist = np.linalg.norm(grid_points - pos, axis=1)
        density += np.exp(-dist**2 / (2 * sigma**2))

    origin = min_coords
    step_sizes = np.array([grid_spacing, grid_spacing, grid_spacing])
    step_counts = np.array([len(x) - 1, len(y) - 1, len(z) - 1], dtype=int)

    return density, origin, step_sizes, step_counts


positions, atoms = parse_xyz(XYZ_FILE)
print(f"Loaded {len(positions)} atoms from {XYZ_FILE}")

fragments = parse_fragment_assignments(SUPERSYSTEM_OUT)
print(f"\nDetected {len(fragments)} fragments:")
for frag_id in sorted(fragments.keys()):
    print(f"  Fragment {frag_id}: {len(fragments[frag_id])} atoms - {fragments[frag_id]}")

fragment_colors = {
    1:  ([0.8, 0.0, 0.0],   "Dark Red"),
    2:  ([0.0, 0.5, 0.8],   "Deep Blue"),
    3:  ([0.0, 0.6, 0.0],   "Dark Green"),
    4:  ([1.0, 0.4, 0.0],   "Dark Orange"),
    5:  ([0.6, 0.0, 0.6],   "Magenta"),
    6:  ([0.4, 0.4, 0.0],   "Olive"),
    7:  ([0.0, 0.7, 0.7],   "Teal"),
    8:  ([0.8, 0.4, 0.6],   "Rose"),
    9:  ([0.4, 0.2, 0.0],   "Brown"),
    10: ([0.5, 0.5, 0.8],   "Periwinkle"),
    11: ([0.9, 0.75, 0.0],  "Gold"),
    12: ([0.0, 0.4, 0.4],   "Dark Cyan"),
    13: ([0.7, 0.3, 0.3],   "Salmon"),
    14: ([0.3, 0.3, 0.5],   "Slate"),
    15: ([0.6, 0.8, 0.2],   "Lime"),
    16: ([0.5, 0.0, 0.3],   "Plum"),
}


def get_fragment_color(frag_id, frag_ids):
    if frag_id in fragment_colors:
        return fragment_colors[frag_id]
    idx = frag_ids.index(frag_id)
    hue = (idx / max(1, len(frag_ids))) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
    return ([r, g, b], f"Auto_{frag_id}")


class LabelSettings(Displayable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _script_string(self):
        return "\n".join([
            'cmd.set("label_bg_outline", 1)',
            'cmd.set("label_bg_color", "black")',
            'cmd.set("float_labels", 1)',
        ])


print("\nGenerating volumetric grids and isomeshes...")
isomeshes = []

frag_ids = sorted(fragments.keys())
for frag_id in frag_ids:
    atom_indices = fragments[frag_id]
    frag_positions = positions[atom_indices]

    density, origin, step_sizes, step_counts = create_gaussian_grid(frag_positions)

    grid_data = pmv.GridData(
        values=density,
        origin=origin,
        step_sizes=step_sizes,
        step_counts=step_counts,
        name="density_" + PMV_NAME + f"_fragment_{frag_id}"
    )

    color, color_name = get_fragment_color(frag_id, frag_ids)
    isomesh = pmv.IsoMesh(
        grid_data=grid_data,
        level=0.8,
        name="mesh_" + PMV_NAME + f"_fragment_{frag_id}__{color_name}",
        color=color,
        transparency=0.3
    )
    isomeshes.append(grid_data)
    isomeshes.append(isomesh)
    print(f"  Fragment {frag_id}: {len(atom_indices)} atoms")

# Label receptor fragments with their fp-LED interaction to the ligand fragment
ligand_frag = max(ligand_frags)

label_positions_pos = []
label_texts_pos = []
label_positions_neg = []
label_texts_neg = []
light_blue = [0.68, 0.85, 1.0]
light_red = [0.9, 0.7, 0.7]

for frag_id in frag_ids:
    if frag_id in ligand_frags:
        continue

    atom_indices = fragments[frag_id]
    frag_center = np.mean(positions[atom_indices], axis=0)
    value = get_fp_interaction(fp_interactions, frag_id, ligand_frag)
    if value is None:
        continue

    if value < 0:
        label_texts_neg.append(f"{value:.2f}")
        label_positions_neg.append(frag_center)
    else:
        label_texts_pos.append(f"{value:.2f}")
        label_positions_pos.append(frag_center)

labels_obj = pmv.Labels(
    label_positions_pos,
    label_texts_pos,
    name="labels_" + PMV_NAME + "_fragment_interactions_pos",
    size=24,
    color=light_red,
)
labels_obj_neg = pmv.Labels(
    label_positions_neg,
    label_texts_neg,
    name="labels_" + PMV_NAME + "_fragment_interactions_neg",
    size=24,
    color=light_blue,
)
group = pmv.Group(name=f"group_{PMV_NAME}_{e_int:.2f}")
group.extend(isomeshes)
group.append(labels_obj)
group.append(labels_obj_neg)

script = pmv.Script()
script.add([group, LabelSettings()])

output_script = "visualize_fragments_fpLED.py"
script.write(output_script)

print(f"\nPyMOL script written to: {output_script}")

print("\n" + "="*80)
print("FRAGMENT COLOR ASSOCIATIONS")
print("="*80)
print(f"{'Fragment':<12} {'Color':<12} {'Atoms':<8} {'Atom Indices'}")
print("-"*80)
for frag_id in sorted(fragments.keys()):
    color, color_name = fragment_colors.get(frag_id, ([0.5, 0.5, 0.5], "Gray"))
    atom_indices = fragments[frag_id]
    indices_str = str(atom_indices) if len(atom_indices) <= 10 else f"{atom_indices[:5]}...{atom_indices[-2:]}"
    print(f"Fragment {frag_id:<3} {color_name:<12} {len(atom_indices):<8} {indices_str}")
print("="*80)

print("\nTo visualize:")
print(f"1. Open PyMOL")
print(f"2. Load structure: load {XYZ_FILE}")
print(f"3. Run script: @{output_script}")
print(f"4. Each fragment will be displayed with its colored mesh (tighter fit)")
print("="*80)
