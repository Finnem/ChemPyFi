from pymol import cmd
import chempy
from chempy.sdf import SDFRec
from rdkit import Chem
from chempyfi import vizualize_ligand_interactions, PyMOLInterface, rdutil
vizualize_ligand_interactions("sele")
'''
pymol_interface = PyMOLInterface("sele")

print(pymol_interface.molecules)
for query, molecule in pymol_interface.molecules.items():
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge() != 0:
            print(atom.GetIdx(), atom.GetFormalCharge())
    rdutil.write_molecules(molecule, query.replace(" ", "_") + ".sdf")


print(pymol_interface.rdkit_atom_indices_to_pymol)
#Chem.MolToPDBFile(mol, "test.pdb")
model = cmd.get_model("sele")
#print(dir(model.list()))
print("boope")
for atom in model.atom:
    print(dir(atom))
for bond in model.bond:
    print(dir(bond))
    break
'''