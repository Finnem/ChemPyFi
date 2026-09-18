from chempyfi import rdutil
no_frags_mol = rdutil.read_molecules("bromobenzene_x_no_frags.inp")
rdutil.write_molecules(no_frags_mol, "bromobenzene_x_no_frags.sdf")
frags_mol = rdutil.read_molecules("bromobenzene_x.inp")
rdutil.write_molecules(frags_mol, "bromobenzene_x.sdf")
defer_mol = rdutil.read_molecules("chlorobenzene_x.inp")
rdutil.write_molecules(defer_mol, "chlorobenzene_x.sdf")