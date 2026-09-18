from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

def draw_molecule(mol, highlights = None, highlight_color = (0, 0.8, 0, .8), height= 400, width = 400, labels = "indices", *args, **kwargs):
    """ Draw a molecule with highlights.

    Args:
        mol (RDKit.Mol): Molecule to draw.
        highlights (list): List of atoms or bonds to highlight.
        highlight_color (tuple): Defaults to red. Tuple of RGB values.
        height (int): Defaults to 400. Height of the SVG.
        width (int): Defaults to 400. Width of the SVG.
        labels (function): Defaults to atom index. Function to get the labels of the atoms. The function should take an atom as input and return a string. If None is passed, no labels will be shown.
        *args: Additional arguments to pass to the drawer.
        **kwargs: Additional keyword arguments to pass to the drawer.

    Returns:
        IPython.display.SVG: SVG of the molecule.
    """
    from IPython.display import SVG, display
    if highlights is None:
        highlights = []
    highlight_atoms = set()
    highlight_bonds = set()
    for highlight in highlights:
        if type(highlight) == Chem.Mol:
            highlight_atoms.update([atom.GetIdx() for atom in highlight.GetAtoms()])
            highlight_bonds.update([bond.GetIdx() for bond in highlight.GetBonds()])
        elif isinstance(highlight, int):
            highlight_atoms.add(highlight)
        elif isinstance(highlight, Chem.Atom):
            highlight_atoms.add(highlight.GetIdx())
        elif isinstance(highlight, Chem.Bond):
            highlight_bonds.add(highlight.GetIdx())
        else:
            raise ValueError("Highlight must be either an atom or a bond.")

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().highlight_color = highlight_color
    mol = Chem.Mol(mol)
    if labels is not None:
        if labels == "indices":
            for atom in mol.GetAtoms():
                atom.SetProp("atomLabel", f"{atom.GetSymbol()}:{str(atom.GetIdx())}")
        else:
            for atom in mol.GetAtoms():
                atom.SetProp("atomLabel", labels(atom))
    Chem.rdDepictor.Compute2DCoords(mol)
    drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms), highlightBonds=list(highlight_bonds),
                        highlightAtomColors={atom_idx : highlight_color for atom_idx in highlight_atoms}, highlightBondColors={bond_idx : highlight_color for bond_idx in highlight_bonds},
                        *args,
                        **kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    display(SVG(svg))
    return svg