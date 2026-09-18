
from rdkit import Chem
from rdkit.Chem import rdchem, rdmolops
import rdkit
import numpy as np
from chempyfi.rdutil.util import bond_order_from_float
import logging
from rdkit.Chem import GetPeriodicTable

periodic_table = GetPeriodicTable()


class PyMOLInterface():
    """PyMOL selection ↔ RDKit molecule bridge."""

    def __init__(self, selection=None, full_objects=True, just_current_state=True, just_enabled=True):
        from pymol import cmd
        self.molecules = {}
        self.rdkit_atom_indices_to_pymol = {}
        self.pymol_to_rdkit_atom_indices = {}
        if selection is not None:
            self.update(selection, full_objects, just_current_state=just_current_state, just_enabled=just_enabled)

    def update(self, selection, full_objects=True, force_update=False, just_current_state=True, just_enabled=True):
        from pymol import cmd

        if just_enabled:
            selection = f"{selection} and enabled"

        for mol_object in cmd.get_object_list(selection):
            if full_objects:
                model_query = f"({mol_object})"
            else:
                model_query = f"({selection}) and {mol_object}"
            if (model_query in self.molecules) and not force_update:
                continue
            num_states = cmd.count_states(f"{model_query}")
            for state in range(1, num_states + 1):
                if just_current_state:
                    if state != cmd.get_state():
                        continue
                model = cmd.get_model(f"{model_query}", state=state)
                if model.nAtom == 0:
                    continue
                active_model_query = model_query + f" and state {state}"
                self.molecules[active_model_query], pymol_to_rdkit_atom_idx = self._construct_mol_from_chempy_model(
                    model, state=state
                )
                self.pymol_to_rdkit_atom_indices[active_model_query] = {
                    f"{key} and {mol_object}": value for key, value in pymol_to_rdkit_atom_idx.items()
                }
        self.rdkit_atom_indices_to_pymol.update(
            {
                mol_object: {value: key for key, value in mapping.items()}
                for mol_object, mapping in self.pymol_to_rdkit_atom_indices.items()
            }
        )

    def match_selection(self, selection):
        from pymol import cmd
        atom_indices = {}
        for atom in cmd.get_model(selection).atom:
            identifier = self.unique_pymol_atom_selection(atom)
            for model_query, mapping in self.pymol_to_rdkit_atom_indices.items():
                if identifier in mapping:
                    if model_query not in atom_indices:
                        atom_indices[model_query] = {}
                    atom_indices[model_query][identifier] = mapping[identifier]
        return atom_indices

    def get_pymol_query(self, model_query, atom_indices):
        query = []
        for idx in atom_indices:
            if model_query not in self.rdkit_atom_indices_to_pymol:
                raise ValueError(
                    f"Model {model_query} not found. Possible values are: "
                    f"{list(self.rdkit_atom_indices_to_pymol.keys())}"
                )
            query.append("(" + self.rdkit_atom_indices_to_pymol[model_query][int(idx)] + ")")
        return " or ".join(query)

    def _construct_mol_from_chempy_model(self, model, state=None):
        pymol_to_rdkit_atom_idx = {}
        symbols = np.full(model.nAtom, "")
        charges = np.zeros(model.nAtom)
        positions = np.zeros((model.nAtom, 3))
        bond_types = []
        bonds = np.zeros((model.nBond, 2), dtype=int)

        for i, atom in enumerate(model.atom):
            pymol_to_rdkit_atom_idx[self.unique_pymol_atom_selection(atom, state)] = i
            symbols[i] = atom.symbol
            positions[i] = atom.coord
            charges[i] = atom.formal_charge

        for i, bond in enumerate(model.bond):
            bond_types.append(bond_order_from_float(bond.order))
            bonds[i] = [bond.index[0], bond.index[1]]

        mol = Chem.RWMol()
        for symbol, charge in zip(symbols, charges):
            a = Chem.Atom(symbol)
            a.SetFormalCharge(int(charge))
            a.SetNoImplicit(True)
            mol.AddAtom(a)
        for i, bond in enumerate(bonds):
            mol.AddBond(int(bond[0]), int(bond[1]), bond_types[i])

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(i, pos)
        mol.AddConformer(conf, assignId=True)
        new_mol = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(new_mol)
            mol = new_mol
        except rdkit.Chem.rdchem.AtomValenceException:
            logging.exception("Warning: AtomValenceException encountered. Attempting to fix.")
            for atom in mol.GetAtoms():
                implicit_Hs = atom.GetNumImplicitHs()
                atom.SetFormalCharge(atom.GetFormalCharge() - implicit_Hs)
            mol = rdmolops.RemoveHs(mol, implicitOnly=True, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            try:
                Chem.SanitizeMol(mol)
            except rdkit.Chem.rdchem.AtomValenceException:
                logging.exception("Warning: AtomValenceException encountered. Could not fix.")
        return mol, pymol_to_rdkit_atom_idx

    def unique_pymol_atom_selection(self, atom, state=None):
        chain = f" and chain {atom.chain}" if atom.chain else ""
        state_str = f" and state {state}" if state else ""
        return f"(id {atom.id}{chain}{state_str})"
