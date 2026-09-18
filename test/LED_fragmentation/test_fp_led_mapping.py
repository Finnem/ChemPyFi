"""
Example of using compute_fp_led_interactions_with_mapping to compute fp-LED interactions
with custom fragment mappings (no position matching required).
"""

from chempyfi.orcautil import compute_fp_led_interactions_with_mapping
from pathlib import Path

def example_fp_led_with_custom_mappings():
    """
    Example: Compute fp-LED interactions for a complex system using custom fragment mappings.
    
    This example shows how to:
    1. Define custom fragment mappings (how subsystem fragments map to supersystem fragments)
    2. Run LEDAW with those mappings
    3. Get fp-LED pairwise interactions as a dictionary
    """
    
    # Define your ORCA output files
    main_filenames = [
        "complex/complex.out",      # Supersystem (complex)
        "ligand/ligand.out",        # Subsystem 1 (ligand)
        "receptor/receptor.out"     # Subsystem 2 (receptor)
    ]
    
    alternative_filenames = [
        "",  # No alternative for complex
        "",  # No alternative for ligand
        ""   # No alternative for receptor
    ]
    
    # Define fragment mappings
    # This tells LEDAW how fragments in subsystems map to fragments in the supersystem
    # Format: {system_label: {subsystem_frag_id: supersystem_frag_id, ...}}
    fragment_mappings = {
        "SUPERSYS": {
            # Supersystem fragments map to themselves (identity)
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7
        },
        "SUBSYS1": {
            # Ligand fragments: subsystem fragments 1,2,3 map to supersystem fragments 1,2,3
            1: 1,  # Ligand frag 1 -> Complex frag 1
            2: 2,  # Ligand frag 2 -> Complex frag 2
            3: 3   # Ligand frag 3 -> Complex frag 3
        },
        "SUBSYS2": {
            # Receptor fragments: subsystem fragment 1 maps to supersystem fragment 4
            1: 4,  # Receptor frag 1 -> Complex frag 4
            2: 5,  # Receptor frag 2 -> Complex frag 5
            3: 6,  # Receptor frag 3 -> Complex frag 6
            4: 7   # Receptor frag 4 -> Complex frag 7
        }
    }
    
    # Compute fp-LED interactions
    fp_interactions = compute_fp_led_interactions_with_mapping(
        main_filenames=main_filenames,
        alternative_filenames=alternative_filenames,
        fragment_mappings=fragment_mappings,
        method="DLPNO-CCSD(T)",
        conversion_factor=2625.5,  # Hartree to kJ/mol
        use_temp_dir=True,        # Use temporary directory (auto-cleaned)
        use_dataframes=True,      # Process fp-LED in memory (avoids fp-LED Excel file)
        verbose=True              # Print progress
    )
    
    # Access pairwise interactions
    print("\n=== fp-LED Pairwise Interactions ===")
    print(f"Total number of fragment pairs: {len(fp_interactions)}")
    print("\nSample interactions:")
    for key, energy in list(fp_interactions.items())[:10]:
        frag_i, frag_j = key.split('_')
        print(f"  Fragment {frag_i} - Fragment {frag_j}: {energy:.4f} kJ/mol")
    
    # Calculate total interaction energy
    total_energy = sum(fp_interactions.values())
    print(f"\nTotal fp-LED interaction energy: {total_energy:.4f} kJ/mol")
    
    # Access specific fragment pair
    interaction_1_4 = fp_interactions.get("1_4", None)
    if interaction_1_4 is not None:
        print(f"\nInteraction between fragment 1 and 4: {interaction_1_4:.4f} kJ/mol")
    
    return fp_interactions


def example_simple_system():
    """
    Example: Single system (no subsystems) - simpler case.
    """
    
    main_filenames = ["molecule.out"]
    alternative_filenames = [""]
    
    # For a single system, you only need to define SUPERSYS mapping
    fragment_mappings = {
        "SUPERSYS": {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5  # Identity mapping
        }
    }
    
    fp_interactions = compute_fp_led_interactions_with_mapping(
        main_filenames=main_filenames,
        alternative_filenames=alternative_filenames,
        fragment_mappings=fragment_mappings,
        use_temp_dir=True,
        use_dataframes=True,
        verbose=False
    )
    
    print(f"Found {len(fp_interactions)} fragment pair interactions")
    return fp_interactions


def example_from_orca_input():
    """
    Example: Extract fragment mappings from ORCA input file and use them.
    """
    from chempyfi.rdutil.io import read_orca_inp_file
    import json
    
    # Read ORCA input file to get fragment information
    mols = read_orca_inp_file(
        "complex/orca.inp",
        extract_energy=True,
        run_ledaw=False  # We'll run LEDAW separately with custom mappings
    )
    mol = mols[0]
    
    # Get fragments from molecule property
    if mol.HasProp("fragments"):
        fragments = json.loads(mol.GetProp("fragments"))
        print(f"Found {len(fragments)} fragments in input file")
        
        # Build fragment mappings
        # For a single system, fragments are just identity mapped
        fragment_mappings = {
            "SUPERSYS": {i+1: i+1 for i in range(len(fragments))}
        }
        
        # Now compute fp-LED interactions
        fp_interactions = compute_fp_led_interactions_with_mapping(
            main_filenames=["complex/complex.out"],
            alternative_filenames=[""],
            fragment_mappings=fragment_mappings,
            use_temp_dir=True,
            use_dataframes=True
        )
        
        return fp_interactions
    else:
        print("No fragments found in input file")
        return None


if __name__ == "__main__":
    # Run example
    print("Example 1: Complex system with ligand and receptor")
    print("=" * 60)
    # Uncomment to run:
    # fp_interactions = example_fp_led_with_custom_mappings()
    
    print("\n\nExample 2: Simple single system")
    print("=" * 60)
    # Uncomment to run:
    # fp_interactions = example_simple_system()
    
    print("\n\nExample 3: Extract mappings from ORCA input")
    print("=" * 60)
    # Uncomment to run:
    # fp_interactions = example_from_orca_input()


