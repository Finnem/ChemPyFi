# LEDAW is now bundled with chempyfi - no path setup needed!
import sys
import numpy as np
import pandas as pd
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
# Ligand detection: automatically take the full covalently connected molecule
# containing bromine atom(s) in the XYZ structure. Atom indices are 0-based.
LIGAND_DETECTION_MODE = "bromine_connected_component"
BROMINE_SYMBOLS = {"Br", "BR", "br"}
# Optional manual override. Leave as None for automatic Br-based detection.
MANUAL_LIGAND_ATOM_INDICES = None

OUTPUT_EXCEL = "/home/highgarden/Workspace/Side_Projects/Janosch/Taut/LED/ORCA/LED_out/relax_5L4Q_full_pure_obj01_entry_00001_conf_01/relax_5L4Q_full_pure_obj01_entry_00001_conf_01_LED_interaction_fpCovaLED.xlsx"
PMV_NAME = os.path.splitext(os.path.basename(OUTPUT_EXCEL))[0]

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

COVALENT_RADII_ANGSTROM = {
    # Common elements in biomolecular/ligand structures. Values are approximate
    # single-bond covalent radii and are only used for connectivity detection.
    "H": 0.31, "B": 0.85, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "CL": 1.02, "Cl": 1.02, "BR": 1.20, "Br": 1.20,
    "I": 1.39, "NA": 1.66, "Na": 1.66, "MG": 1.41, "Mg": 1.41,
    "K": 2.03, "CA": 1.76, "Ca": 1.76, "FE": 1.32, "Fe": 1.32,
    "ZN": 1.22, "Zn": 1.22,
}


def covalent_radius(symbol):
    """Return an approximate covalent radius in Angstrom for bond detection."""
    clean = re.sub(r"[^A-Za-z]", "", str(symbol))
    if clean in COVALENT_RADII_ANGSTROM:
        return COVALENT_RADII_ANGSTROM[clean]
    titled = clean[:1].upper() + clean[1:].lower()
    if titled in COVALENT_RADII_ANGSTROM:
        return COVALENT_RADII_ANGSTROM[titled]
    upper = clean.upper()
    if upper in COVALENT_RADII_ANGSTROM:
        return COVALENT_RADII_ANGSTROM[upper]
    return 0.77  # Reasonable generic fallback for organic heavy atoms.


def build_covalent_bond_graph(atoms, positions, scale=1.25, tolerance=0.10):
    """Build a simple covalent bond graph from XYZ coordinates.

    Two atoms are connected when their distance is less than
    scale * (r_cov_i + r_cov_j) + tolerance. This is intentionally conservative
    enough to avoid connecting noncovalent contacts while still catching normal
    covalent bonds, including C-Br.
    """
    n_atoms = len(atoms)
    graph = {i: set() for i in range(n_atoms)}
    positions = np.asarray(positions, dtype=float)

    for i in range(n_atoms):
        radius_i = covalent_radius(atoms[i])
        for j in range(i + 1, n_atoms):
            radius_j = covalent_radius(atoms[j])
            cutoff = scale * (radius_i + radius_j) + tolerance
            distance = float(np.linalg.norm(positions[i] - positions[j]))
            if distance <= cutoff:
                graph[i].add(j)
                graph[j].add(i)

    return graph


def connected_component(graph, start):
    """Return sorted atom indices in the graph component containing start."""
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in graph[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return sorted(seen)


def detect_ligand_atoms_from_bromine(atoms, positions):
    """Detect ligand atoms as the connected molecule containing bromine atom(s)."""
    bromine_indices = [i for i, atom in enumerate(atoms) if atom in BROMINE_SYMBOLS]
    if not bromine_indices:
        raise ValueError(
            "Could not detect ligand automatically: no bromine atom was found in the XYZ file. "
            "Set MANUAL_LIGAND_ATOM_INDICES if this structure has no Br-labelled ligand."
        )

    graph = build_covalent_bond_graph(atoms, positions)

    ligand_atom_set = set()
    bromine_components = []
    for br_idx in bromine_indices:
        component = connected_component(graph, br_idx)
        bromine_components.append(component)
        ligand_atom_set.update(component)

    ligand_atom_indices = sorted(ligand_atom_set)
    if not ligand_atom_indices:
        raise ValueError("Bromine was found, but no connected ligand component could be built.")

    print("\nLigand detection from bromine connectivity:")
    print(f"  Bromine atom indices: {bromine_indices}")
    for br_idx, component in zip(bromine_indices, bromine_components):
        print(f"  Component attached to Br atom {br_idx}: {len(component)} atoms")
    print(f"  Union ligand atom count: {len(ligand_atom_indices)}")
    print(f"  Ligand atom indices: {ligand_atom_indices}")

    return ligand_atom_indices


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

if MANUAL_LIGAND_ATOM_INDICES is not None:
    LIGAND_ATOM_INDICES = sorted(MANUAL_LIGAND_ATOM_INDICES)
    print(f"Using manual ligand atom indices: {LIGAND_ATOM_INDICES}")
elif LIGAND_DETECTION_MODE == "bromine_connected_component":
    LIGAND_ATOM_INDICES = detect_ligand_atoms_from_bromine(atoms, positions)
else:
    raise ValueError(f"Unknown LIGAND_DETECTION_MODE: {LIGAND_DETECTION_MODE}")

print("Computing LED interaction matrix...")
led_int = compute_led_interaction_matrix(
    SUPERSYSTEM_OUT,
    LIGAND_OUT,
    SURR_OUT,
    ligand_atom_indices=LIGAND_ATOM_INDICES,
    verbose=True
)
print(led_int)

def zero_self_terms(df):
    """Return a copy of df with fragment self/diagonal terms set to zero.

    The CovaLED/fp-CovaLED visualization should show fragment-pair interactions.
    Diagonal entries are self terms, and in this workflow they can remain as copied
    supersystem values from the standard LED matrix. Keeping them makes both totals
    and redistribution look wrong, so they are excluded from pairwise analysis.
    """
    cleaned = df.copy()
    for idx in cleaned.index:
        col = find_key(cleaned.columns, idx)
        if col is not None:
            cleaned.loc[idx, col] = 0.0
    return cleaned


print("\nExtracting interaction energy...")
led_int_no_self = zero_self_terms(led_int)
e_int_raw_with_self = extract_interaction_energy(led_int)
e_int = extract_interaction_energy(led_int_no_self)
print(f"\n{'='*80}")
print(f"RAW MATRIX SUM INCLUDING DIAGONAL/SELF TERMS: {e_int_raw_with_self:.4f} kJ/mol")
print(f"PAIRWISE INTERACTION ENERGY, DIAGONAL ZEROED: {e_int:.4f} kJ/mol")
print(f"{'='*80}")

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


def get_matrix_value(df, frag_a, frag_b):
    """Return df[frag_a, frag_b] or df[frag_b, frag_a], handling int/string labels."""
    return get_interaction_value(df, frag_a, frag_b)


def determine_ligand_receptor_fragments(fragments, ligand_atom_indices):
    """Classify 1-based ORCA fragments as ligand or receptor from 0-based ligand atom indices."""
    ligand_atom_set = set(ligand_atom_indices)
    ligand_frags = []
    receptor_frags = []

    for frag_id, atom_indices in sorted(fragments.items()):
        atom_set = set(atom_indices)
        ligand_count = len(atom_set & ligand_atom_set)
        receptor_count = len(atom_set - ligand_atom_set)

        if ligand_count and receptor_count:
            raise ValueError(
                f"Fragment {frag_id} contains both ligand and receptor atoms. "
                "Use finer ORCA fragment definitions or adjust LIGAND_ATOM_INDICES."
            )
        if ligand_count:
            ligand_frags.append(frag_id)
        elif receptor_count:
            receptor_frags.append(frag_id)

    return sorted(ligand_frags), sorted(receptor_frags)


def compute_fp_covaled_redistributed_matrix(led_int, ligand_frags, receptor_frags):
    """Apply fp-CovaLED Step 8 redistribution to inter-molecular fragment pairs.

    Diagonal/self terms are excluded. They are not ligand-receptor fragment-pair
    interactions and can be copied standard-LED supersystem entries. Only
    off-diagonal intra-molecular terms are redistributed to inter-molecular
    ligand-receptor pairs.

    Returns:
        tuple[pd.DataFrame, dict, float, float]
        - fp_led_int: matrix with only redistributed inter-molecular values kept;
          all intra-molecular and diagonal entries are zeroed.
        - fp_pair_values: dict keyed as (receptor_frag, ligand_frag) -> value.
        - intra_total: off-diagonal intra-molecular contribution redistributed.
        - total_abs_inter: sum of absolute raw inter-molecular contributions.
    """
    led_int = zero_self_terms(led_int)
    fp_led_int = led_int.copy()

    # Start with a clean matrix; only inter-molecular fp-CovaLED terms remain.
    fp_led_int.loc[:, :] = 0.0

    intra_ligand = 0.0
    for i, frag_i in enumerate(ligand_frags):
        for frag_j in ligand_frags[i + 1:]:  # Off-diagonal only; exclude self terms.
            value = get_matrix_value(led_int, frag_i, frag_j)
            if value is not None and not np.isnan(value):
                intra_ligand += value

    intra_receptor = 0.0
    for i, frag_i in enumerate(receptor_frags):
        for frag_j in receptor_frags[i + 1:]:  # Off-diagonal only; exclude self terms.
            value = get_matrix_value(led_int, frag_i, frag_j)
            if value is not None and not np.isnan(value):
                intra_receptor += value

    intra_total = intra_ligand + intra_receptor

    raw_pair_values = {}
    for receptor_frag in receptor_frags:
        for ligand_frag in ligand_frags:
            value = get_matrix_value(led_int, receptor_frag, ligand_frag)
            if value is not None and not np.isnan(value):
                raw_pair_values[(receptor_frag, ligand_frag)] = value

    total_abs_inter = sum(abs(value) for value in raw_pair_values.values())

    fp_pair_values = {}
    for pair, raw_value in raw_pair_values.items():
        if total_abs_inter > 0 and abs(intra_total) > 1e-6:
            weight = abs(raw_value) / total_abs_inter
            fp_value = raw_value + intra_total * weight
        else:
            fp_value = raw_value

        receptor_frag, ligand_frag = pair
        fp_pair_values[pair] = fp_value

        row = find_key(fp_led_int.index, receptor_frag)
        col = find_key(fp_led_int.columns, ligand_frag)
        if row is None or col is None:
            # Fall back to the swapped orientation if needed.
            row = find_key(fp_led_int.index, ligand_frag)
            col = find_key(fp_led_int.columns, receptor_frag)
        if row is not None and col is not None:
            fp_led_int.loc[row, col] = fp_value

    return fp_led_int, fp_pair_values, intra_total, total_abs_inter

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

# Determine ligand/receptor fragment groups and apply fp-CovaLED Step 8 redistribution.
# This is the "complete Redistribution" step from the LED utility module:
# intra-molecular CovaLED terms are redistributed to ligand-receptor pairs.
ligand_frags, receptor_frags = determine_ligand_receptor_fragments(fragments, LIGAND_ATOM_INDICES)
print(f"\nLigand fragments: {ligand_frags}")
print(f"Receptor/surrounding fragments: {receptor_frags}")

fp_led_int, fp_pair_values, intra_total, total_abs_inter = compute_fp_covaled_redistributed_matrix(
    led_int_no_self,
    ligand_frags,
    receptor_frags,
)

fp_e_int = float(fp_led_int.sum().sum())
print(f"\n{'='*80}")
print("fp-CovaLED REDISTRIBUTION")
print(f"Redistributed intra-molecular contribution: {intra_total:.4f} kJ/mol")
print(f"Sum(|raw inter-molecular pairs|): {total_abs_inter:.4f} kJ/mol")
print(f"fp-CovaLED inter-molecular sum: {fp_e_int:.4f} kJ/mol")
print(f"Pairwise CovaLED interaction energy, diagonal zeroed: {e_int:.4f} kJ/mol")
print(f"Difference: {fp_e_int - e_int:.6f} kJ/mol")
print(f"{'='*80}")

# Save the raw matrix for debugging, plus pairwise matrices with diagonal/self terms zeroed.
with pd.ExcelWriter(OUTPUT_EXCEL) as writer:
    led_int.to_excel(writer, sheet_name="Raw_CovaLED_with_self")
    led_int_no_self.to_excel(writer, sheet_name="CovaLED_pairwise")
    fp_led_int.to_excel(writer, sheet_name="fpCovaLED_pairwise")
    pair_rows = [
        {"receptor_fragment": rec, "ligand_fragment": lig, "fpCovaLED_kJ_mol": val}
        for (rec, lig), val in sorted(fp_pair_values.items())
    ]
    pd.DataFrame(pair_rows).to_excel(writer, sheet_name="fp_pair_values", index=False)
print(f"\nLED matrices saved to: {OUTPUT_EXCEL}")

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

# Prepare labels at receptor/surrounding fragment centers with fp-CovaLED redistributed
# interaction values to the ligand. If the ligand contains multiple fragments, their
# redistributed contributions are summed per receptor fragment for a single label.
label_positions_pos = []
label_texts_pos = []
label_positions_neg = []
label_texts_neg = []
light_blue = [0.68, 0.85, 1.0]
light_red = [0.9, 0.7, 0.7]

for receptor_frag in receptor_frags:
    atom_indices = fragments[receptor_frag]
    frag_center = np.mean(positions[atom_indices], axis=0)

    values_to_ligand = [
        fp_pair_values[(receptor_frag, ligand_frag)]
        for ligand_frag in ligand_frags
        if (receptor_frag, ligand_frag) in fp_pair_values
    ]
    if not values_to_ligand:
        continue

    value = float(sum(values_to_ligand))
    if value < 0:
        label_texts_neg.append(f"{value:.2f}")
        label_positions_neg.append(frag_center)
    else:
        label_texts_pos.append(f"{value:.2f}")
        label_positions_pos.append(frag_center)

# Optionally label ligand fragments with zero/summary is skipped to keep the ligand uncluttered.

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
group = pmv.Group(name=f"group_{PMV_NAME}_fpCovaLED_{fp_e_int:.2f}")
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
print(f"5. Labels show fp-CovaLED Step 8 redistributed receptor-to-ligand contributions")
print("="*80)