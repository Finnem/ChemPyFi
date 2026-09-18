# LEDAW is now bundled with chempyfi - no path setup needed!
import sys
import numpy as np
import colorsys
import re
import os
from chempyfi.orcautil import compute_led_interaction_matrix, extract_interaction_energy

# Add PyMolViz to path
sys.path.insert(0, '/home/highgarden/Workspace/Python_Libraries/PyMolViz/module')
import pymolviz as pmv
from pymolviz.Displayable import Displayable

SUPERSYSTEM_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/super/orca_2111303/orca.out"
SURR_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/sub1/orca_2111305/orca.out"
LIGAND_OUT = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/sub2/orca_2111304/orca.out"
XYZ_FILE = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/relax_5L4Q_full_pure_obj01_entry_00001_conf_01_LED_interaction_structure.xyz"
LIGAND_ATOM_INDICES = [277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294]

print("Computing LED interaction matrix...")
led_int = compute_led_interaction_matrix(
    SUPERSYSTEM_OUT, 
    LIGAND_OUT, 
    SURR_OUT, 
    ligand_atom_indices=LIGAND_ATOM_INDICES,
    verbose=True
)
print(led_int)

# Save DataFrame to Excel
OUTPUT_EXCEL = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/relax_5L4Q_full_pure_obj01_entry_00001_conf_01_LED_interaction.xlsx"
led_int.to_excel(OUTPUT_EXCEL, sheet_name="LED_Interaction")
print(f"\nLED interaction matrix saved to: {OUTPUT_EXCEL}")
PMV_NAME = os.path.splitext(os.path.basename(OUTPUT_EXCEL))[0]

print("\nExtracting total interaction energy...")
e_int = extract_interaction_energy(led_int)
print(f"\n{'='*80}")
print(f"TOTAL INTERACTION ENERGY: {e_int:.4f} kJ/mol")
print(f"{'='*80}")

# Parse fragment assignments from ORCA output
print("\n" + "="*80)
print("Creating PyMOL visualization with fragment isomeshes...")
print("="*80)

def parse_fragment_assignments(orca_output):
    """Parse fragment assignments from ORCA output file."""
    with open(orca_output, 'r') as f:
        lines = f.readlines()
    
    # Find the atomic coordinates section with fragment info
    fragment_assignments = []
    in_coord_section = False
    
    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (A.U.)' in line:
            in_coord_section = True
            continue
        
        if in_coord_section:
            # Look for lines with atom data: NO LB ZA FRAG MASS X Y Z
            match = re.match(r'\s+(\d+)\s+\w+\s+[\d.]+\s+(\d+)\s+[\d.]+', line)
            if match:
                atom_idx = int(match.group(1))
                frag_idx = int(match.group(2))
                fragment_assignments.append((atom_idx, frag_idx))
            elif line.strip().startswith('---') or line.strip() == '':
                if fragment_assignments:
                    break
    
    # Organize by fragment
    fragments = {}
    for atom_idx, frag_idx in fragment_assignments:
        if frag_idx not in fragments:
            fragments[frag_idx] = []
        fragments[frag_idx].append(atom_idx)
    
    return fragments

def parse_xyz(filename):
    """Parse XYZ file and return atom positions."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    atoms = []
    positions = []
    
    for line in lines[2:2+n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            atoms.append(parts[0])
            positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return np.array(positions), atoms

def create_gaussian_grid(positions, grid_spacing=0.4, padding=1.5, sigma=0.8):
    """Create a 3D grid with Gaussian density around atom positions.
    
    Tighter mesh settings:
    - padding: 1.5 Å (reduced from 3.0)
    - sigma: 0.8 Å (reduced from 1.5) - controls Gaussian spread
    - grid_spacing: 0.4 Å (slightly finer grid)
    """
    positions = np.array(positions)
    
    # Determine grid bounds
    min_coords = positions.min(axis=0) - padding
    max_coords = positions.max(axis=0) + padding
    
    # Create grid
    x = np.arange(min_coords[0], max_coords[0], grid_spacing)
    y = np.arange(min_coords[1], max_coords[1], grid_spacing)
    z = np.arange(min_coords[2], max_coords[2], grid_spacing)
    
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    
    # Calculate Gaussian density at each grid point
    density = np.zeros(len(grid_points))
    for pos in positions:
        dist = np.linalg.norm(grid_points - pos, axis=1)
        density += np.exp(-dist**2 / (2 * sigma**2))
    
    origin = min_coords
    step_sizes = np.array([grid_spacing, grid_spacing, grid_spacing])
    step_counts = np.array([len(x)-1, len(y)-1, len(z)-1], dtype=int)
    
    return density, origin, step_sizes, step_counts

# Parse structure
positions, atoms = parse_xyz(XYZ_FILE)
print(f"Loaded {len(positions)} atoms from {XYZ_FILE}")

# Parse fragment assignments from ORCA output
fragments = parse_fragment_assignments(SUPERSYSTEM_OUT)
print(f"\nDetected {len(fragments)} fragments:")
for frag_id in sorted(fragments.keys()):
    print(f"  Fragment {frag_id}: {len(fragments[frag_id])} atoms - {fragments[frag_id]}")

# Define distinct colors for each fragment with names
# Colors optimized for visibility on white background
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
    # Generate distinct colors if we run out of predefined ones
    idx = frag_ids.index(frag_id)
    hue = (idx / max(1, len(frag_ids))) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.85)
    return ([r, g, b], f"Auto_{frag_id}")

def find_key(axis, frag_id):
    candidates = [frag_id, str(frag_id), f"Fragment {frag_id}", f"frag_{frag_id}"]
    for cand in candidates:
        if cand in axis:
            return cand
    return None

def get_interaction_value(df, frag_a, frag_b):
    row = find_key(df.index, frag_a)
    col = find_key(df.columns, frag_b)
    if row is not None and col is not None:
        try:
            return float(df.loc[row, col])
        except Exception:
            return None
    # Try swapped axes
    row = find_key(df.index, frag_b)
    col = find_key(df.columns, frag_a)
    if row is not None and col is not None:
        try:
            return float(df.loc[row, col])
        except Exception:
            return None
    return None

class LabelSettings(Displayable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _script_string(self):
        lines = [
            'cmd.set("label_bg_outline", 1)',
            'cmd.set("label_bg_color", "black")',
            'cmd.set("float_labels", 1)',
        ]
        return "\n".join(lines)

# Create isomeshes for each fragment
print("\nGenerating volumetric grids and isomeshes...")
isomeshes = []

frag_ids = sorted(fragments.keys())
for frag_id in frag_ids:
    atom_indices = fragments[frag_id]
    frag_positions = positions[atom_indices]
    
    # Create grid for this fragment
    density, origin, step_sizes, step_counts = create_gaussian_grid(frag_positions)
    
    # Create GridData object
    grid_data = pmv.GridData(
        values=density,
        origin=origin,
        step_sizes=step_sizes,
        step_counts=step_counts,
        name="density_" + PMV_NAME + f"_fragment_{frag_id}"
    )
    
    # Create IsoMesh object with tighter threshold
    color, color_name = get_fragment_color(frag_id, frag_ids)
    isomesh = pmv.IsoMesh(
        grid_data=grid_data,
        level=0.8,  # Higher threshold = tighter mesh (increased from 0.5)
        name="mesh_" + PMV_NAME + f"_fragment_{frag_id}__{color_name}",
        color=color,
        transparency=0.3
    )
    isomeshes.append(grid_data)
    isomeshes.append(isomesh)
    print(f"  Fragment {frag_id}: {len(atom_indices)} atoms")

# Prepare labels at fragment centers with interaction values to ligand fragment
# The ligand is the last fragment, so read the last column in the LED matrix.
last_column = led_int.columns[-1]

label_positions_pos = []
label_texts_pos = []
label_positions_neg = []
label_texts_neg = []
light_blue = [0.68, 0.85, 1.0]
light_red = [0.9, 0.7, 0.7]
for frag_id in frag_ids:
    atom_indices = fragments[frag_id]
    frag_center = np.mean(positions[atom_indices], axis=0)
    value = None
    row_key = find_key(led_int.index, frag_id)
    if row_key is not None:
        try:
            value = float(led_int.loc[row_key, last_column])
        except Exception:
            value = None
    if value is None:
        continue
    if value < 0:
        label_texts_neg.append(f"{value:.2f}")
        label_positions_neg.append(frag_center)
    else:
        label_texts_pos.append(f"{value:.2f}")
        label_positions_pos.append(frag_center)

# Create PyMOL script
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

# Write PyMOL script
output_script = "visualize_fragments.py"
script.write(output_script)

print(f"\nPyMOL script written to: {output_script}")

# Print color association table
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