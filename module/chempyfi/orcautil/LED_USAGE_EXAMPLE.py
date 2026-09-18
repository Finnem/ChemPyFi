"""
Example script showing proper usage of LED extraction functions.

LEDAW is bundled with chempyfi - no external setup required!
"""

# ============================================================================
# Simply import and use - LEDAW is vendored within chempyfi
# ============================================================================
from chempyfi.orcautil import (
    compute_led_interaction_matrix,
    extract_interaction_energy,
    parse_orca_fragments,
    determine_fragment_groups
)

# ============================================================================
# EXAMPLE 1: Automatic fragment detection
# ============================================================================
def example_auto_fragments():
    """Automatically detect fragments from ORCA output."""
    
    led_int = compute_led_interaction_matrix(
        orca_super="complex/complex.out",
        orca_ligand="ligand/ligand.out",
        orca_receptor="receptor/receptor.out",
        ligand_atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],  # Specify which atoms are ligand
        component='TOTAL',
        method="DLPNO-CCSD(T)",
        conversion_factor=2625.5,
        verbose=False  # Set to True for detailed debug output
    )
    
    # Get total interaction energy
    e_int = extract_interaction_energy(led_int)
    print(f"Total interaction energy: {e_int:.4f} kJ/mol")
    
    return led_int


# ============================================================================
# EXAMPLE 2: Manual fragment specification
# ============================================================================
def example_manual_fragments():
    """Manually specify which fragments belong to ligand vs receptor."""
    
    led_int = compute_led_interaction_matrix(
        orca_super="complex/complex.out",
        orca_ligand="ligand/ligand.out",
        orca_receptor="receptor/receptor.out",
        ligand_fpled_frags=[1, 2, 3],      # Fragments 1-3 are ligand (1-based indexing)
        receptor_fpled_frags=[4, 5, 6],    # Fragments 4-6 are receptor (1-based indexing)
        component='TOTAL'
    )
    
    e_int = extract_interaction_energy(led_int)
    print(f"Total interaction energy: {e_int:.4f} kJ/mol")
    
    return led_int


# ============================================================================
# EXAMPLE 3: Parse fragments independently
# ============================================================================
def example_parse_fragments():
    """Parse fragment assignments from ORCA output file."""
    
    # Parse fragment assignments
    atom_to_frag, positions = parse_orca_fragments("complex/complex.out")
    print(f"Fragment assignments: {atom_to_frag}")
    
    # Determine ligand vs receptor fragments
    ligand_atoms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ligand_frags, receptor_frags = determine_fragment_groups(
        atom_to_frag,
        ligand_atoms,
        assign_mixed='majority'  # Handle fragments spanning both molecules
    )
    
    print(f"Ligand fragments: {ligand_frags}")
    print(f"Receptor fragments: {receptor_frags}")
    
    # Use in LED calculation
    led_int = compute_led_interaction_matrix(
        orca_super="complex/complex.out",
        orca_ligand="ligand/ligand.out",
        orca_receptor="receptor/receptor.out",
        ligand_fpled_frags=ligand_frags,
        receptor_fpled_frags=receptor_frags
    )
    
    return led_int


# ============================================================================
# EXAMPLE 4: Extract multiple energy components
# ============================================================================
def example_components():
    """Extract different LED energy components."""
    
    components = ['TOTAL', 'Electrostat', 'Exchange', 'Correlation']
    
    results = {}
    for component in components:
        led_int = compute_led_interaction_matrix(
            orca_super="complex/complex.out",
            orca_ligand="ligand/ligand.out",
            orca_receptor="receptor/receptor.out",
            ligand_atom_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            component=component
        )
        
        e_int = extract_interaction_energy(led_int)
        results[component] = e_int
        print(f"{component:15s}: {e_int:>10.4f} kJ/mol")
    
    return results


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    print("LED Extraction Examples")
    print("="*80)
    print("Uncomment the example you want to run:\n")
    
    # Uncomment the example you want to run:
    # example_auto_fragments()
    # example_manual_fragments()
    # example_parse_fragments()
    # example_components()
    
    print("Note: LEDAW is bundled with chempyfi - no external setup needed!")

