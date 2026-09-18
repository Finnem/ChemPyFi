"""Line-of-sight occlusion between atom pairs (geometry primitive, no ORCA/CovaLED)."""

import numpy as np
from rdkit import Chem


def check_occlusion(
    from_atoms,
    to_atoms,
    potential_occluders,
    return_occluders=False,
    ignore_outside=True,
    default_radius=1,
    occluders_vdw_factor=1,
):
    """
    Check if segments between paired ``from_atoms`` and ``to_atoms`` are occluded.

    See ``chempyfi.rdutil.geometry.check_occlusion`` for full parameter documentation.
    """
    from ..rdutil.geometry import position

    periodic_table = Chem.GetPeriodicTable()

    if len(from_atoms) != len(to_atoms):
        raise ValueError("from_atoms and to_atoms must have the same length")
    if len(from_atoms) == 0:
        raise ValueError("from_atoms must not be empty")
    if len(potential_occluders) == 0:
        raise ValueError("potential_occluders must not be empty")

    if isinstance(from_atoms, np.ndarray) and (len(from_atoms.shape) == 2) and (from_atoms.shape[1] == 3):
        from_positions = from_atoms
        from_vdw = np.full(len(from_atoms), default_radius)
    else:
        from_positions = position(from_atoms)
        from_vdw = np.array([periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in from_atoms])
    if isinstance(to_atoms, np.ndarray) and (len(to_atoms.shape) == 2) and (to_atoms.shape[1] == 3):
        to_positions = to_atoms
        to_vdw = np.full(len(to_atoms), default_radius)
    else:
        to_positions = position(to_atoms)
        to_vdw = np.array([periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in to_atoms])
    if (
        isinstance(potential_occluders, np.ndarray)
        and (len(potential_occluders.shape) == 2)
        and (potential_occluders.shape[1] == 3)
    ):
        occluders_positions = potential_occluders
        occluders_vdw = np.full(len(potential_occluders), default_radius)
    else:
        occluders_positions = position(potential_occluders)
        occluders_vdw = np.array(
            [periodic_table.GetRvdw(atom.GetAtomicNum()) for atom in potential_occluders]
        )

    occluders_vdw *= occluders_vdw_factor

    bond_directions = to_positions - from_positions
    bond_directions /= np.linalg.norm(bond_directions, axis=1)[:, None]

    from_projections = (bond_directions * from_positions).sum(axis=1)
    to_projections = (bond_directions * to_positions).sum(axis=1)
    occluders_projections = bond_directions @ occluders_positions.T

    from_projection_caps = from_projections + from_vdw
    to_projection_caps = to_projections - to_vdw
    if ignore_outside:
        occluders_filter = (occluders_projections > from_projection_caps[:, None]) & (
            occluders_projections < to_projection_caps[:, None]
        )
    else:
        occluders_projections = np.clip(
            occluders_projections, from_projection_caps[:, None], to_projection_caps[:, None]
        )
        occluders_filter = np.ones_like(occluders_projections, dtype=bool)

    translated_projection = occluders_projections - from_projections[:, None]
    translated_projection = (translated_projection[:, :, None] @ bond_directions[:, None, :]) + from_positions[
        :, None, :
    ]
    occluders_distance_to_projection = np.linalg.norm(
        translated_projection - occluders_positions[None, :, :], axis=-1
    )
    occluded = (occluders_distance_to_projection < occluders_vdw[None, :]) & occluders_filter
    if return_occluders:
        return occluded
    return occluded.any(axis=-1) & (from_projection_caps < to_projection_caps)
