"""Domain types for local-environment extraction (index-explicit)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem


@dataclass(frozen=True)
class Fragment:
    """Contiguous fragment membership in model atom index space."""

    model_atom_indices: Tuple[int, ...]


@dataclass(frozen=True)
class Boundary:
    """Artificial cut across a source bond, optionally capped with a model hydrogen."""

    retained_model_atom: int
    removed_source_atom: Optional[int]
    source_bond: Optional[Tuple[int, int]]
    cap_model_atom: Optional[int]


@dataclass
class MolecularModel:
    """
    Capped local cutout with explicit source↔model index semantics.

    ``model_to_source[i]`` is the source atom index for model atom ``i``, or ``None`` for cap hydrogens.
    """

    molecule: Chem.Mol
    fragments: Tuple[Fragment, ...]
    model_to_source: Tuple[Optional[int], ...]
    boundaries: Tuple[Boundary, ...]
    fragment_connection_bonds: Tuple[Tuple[int, int], ...]

    def to_legacy_tuple(self):
        """Historical ``fragment_molecule`` return shape."""
        frags = [list(f.model_atom_indices) for f in self.fragments]
        return self.molecule, frags, list(self.fragment_connection_bonds)

    @classmethod
    def from_pipeline(
        cls,
        reordered_mol: Chem.Mol,
        split_fragments: Sequence[Sequence[int]],
        split_covalent_bonds: Sequence[Tuple[int, int]],
        cutout_source_to_precut: dict,
        cap_source_heavy_to_precut_h: Sequence[Tuple[int, int]],
        cut_bond_pairs: Sequence[Tuple[int, int]],
        renumber_precut_to_final: dict,
    ) -> "MolecularModel":
        n_model = reordered_mol.GetNumAtoms()
        precut_to_final = {precut: final for precut, final in renumber_precut_to_final.items()}

        model_to_source: List[Optional[int]] = [None] * n_model
        for source_idx, precut_idx in cutout_source_to_precut.items():
            final_idx = precut_to_final.get(precut_idx)
            if final_idx is not None:
                model_to_source[final_idx] = int(source_idx)
        for source_heavy, precut_h in cap_source_heavy_to_precut_h:
            final_h = precut_to_final.get(precut_h)
            if final_h is not None:
                model_to_source[final_h] = None

        boundaries: List[Boundary] = []
        for atom_a, atom_b in cut_bond_pairs:
            retained_source = atom_a
            removed_source = atom_b
            precut_retained = cutout_source_to_precut.get(retained_source)
            cap_precut = None
            for heavy, h_idx in cap_source_heavy_to_precut_h:
                if heavy == retained_source:
                    cap_precut = h_idx
                    break
            if precut_retained is None:
                continue
            final_retained = precut_to_final.get(precut_retained)
            final_cap = precut_to_final.get(cap_precut) if cap_precut is not None else None
            boundaries.append(
                Boundary(
                    retained_model_atom=int(final_retained) if final_retained is not None else -1,
                    removed_source_atom=int(removed_source),
                    source_bond=tuple(sorted((int(atom_a), int(atom_b)))),
                    cap_model_atom=int(final_cap) if final_cap is not None else None,
                )
            )

        fragments = tuple(
            Fragment(model_atom_indices=tuple(int(i) for i in frag)) for frag in split_fragments
        )
        bonds = tuple(tuple(sorted((int(a), int(b)))) for a, b in split_covalent_bonds)
        return cls(
            molecule=reordered_mol,
            fragments=fragments,
            model_to_source=tuple(model_to_source),
            boundaries=tuple(b for b in boundaries if b.retained_model_atom >= 0),
            fragment_connection_bonds=bonds,
        )
