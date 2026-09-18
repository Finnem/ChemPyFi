# %% [code]
from chempyfi import rdutil
import os
#os.environ["XB_DEBUG_SAT"] = "1"
# %% [code]
test_mol = next(rdutil.read_molecules("test_cases/faulty_xyz_test.xyz"))

# %% [code]
corrected_mol = rdutil.correct_bond_orders(test_mol)
# %%
print([(atom.GetFormalCharge(), atom.GetIdx()) for atom in corrected_mol.GetAtoms() if atom.GetFormalCharge() != 0])
# %%
print(type(corrected_mol))
# %%
rdutil.write_molecules([corrected_mol], "test_cases/faulty_xyz_test_corrected.sdf")
# %%
