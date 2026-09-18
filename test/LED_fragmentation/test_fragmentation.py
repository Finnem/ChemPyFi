from chempyfi import rdutil
from chempyfi.orcautil import fragment_molecule, write_orca_input
import rdkit.Chem as Chem
import numpy as np
import pymolviz as pmv
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
"""
def test_fragmentation():
    #mol = next(rdutil.read_molecules("larger_test_case.xyz"))
    #rdutil.write_molecules([mol], "larger_test_case_connected.sdf")
    mol = next(rdutil.read_molecules("larger_test_case_connected.sdf"))
    cutout_mol, frags, split_covalent_bonds = fragment_molecule(mol, 9329, verbose = True, additional_indices = [74198])
    rdutil.write_molecules([cutout_mol], "larger_test_case_cutout.sdf")
    i = 0
    for frag in frags:
        pmv.Points(rdutil.position(cutout_mol)[frag], name = f"frag_{i}").write(f"frag_{i}.py")
        i += 1
    print(split_covalent_bonds)
"""
def test_fragmentation():
    write_orca_input("larger_test_case.xyz", 9329, "ORCA_HEADER_PLACEHOLDER", "larger_test_case_cutout", additional_indices = [74198])
if __name__ == "__main__":
    test_fragmentation()