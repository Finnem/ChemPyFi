"""Tests for the canonical atom ordering of :func:`chempyfi.morgan.unique_index`.

The molecules here are built with hand-picked coordinates rather than generated
conformers, because most of the properties under test are statements about exact
degeneracies that a force field would only reproduce approximately.
"""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.morgan import canonical_order, unique_index
from chempyfi.morgan.canonical import _placement_descriptor, _weisfeiler_lehman_labels

SINGLE = Chem.BondType.SINGLE
DOUBLE = Chem.BondType.DOUBLE
AROMATIC = Chem.BondType.AROMATIC


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def build_molecule(symbols, bonds, positions, aromatic_atoms=()):
    """Build a molecule with explicit hydrogens and one conformer.

    Every atom gets an atom map number that survives renumbering, which is what
    lets the tests talk about physical atoms instead of atom indices.
    """
    aromatic_atoms = set(aromatic_atoms)
    mol = Chem.RWMol()
    for i, symbol in enumerate(symbols):
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        atom.SetAtomMapNum(i + 1)
        atom.SetIsAromatic(i in aromatic_atoms)
        mol.AddAtom(atom)
    for begin, end, bond_type in bonds:
        mol.AddBond(begin, end, bond_type)

    mol = mol.GetMol()
    conformer = Chem.Conformer(len(symbols))
    for i, point in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*(float(value) for value in point)))
    mol.AddConformer(conformer)
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)
    return mol


def canonical_labels(mol):
    """The atom map numbers in canonical order."""
    index = unique_index(mol)
    map_numbers = np.array([atom.GetAtomMapNum() for atom in mol.GetAtoms()])
    return tuple(int(value) for value in map_numbers[np.argsort(index)])


def coordinates(mol):
    return np.array(mol.GetConformer().GetPositions(), dtype=float)


def moved(mol, transform, translation=(0.0, 0.0, 0.0)):
    """A copy of `mol` with `transform` applied to the coordinates."""
    moved_mol = Chem.Mol(mol)
    conformer = moved_mol.GetConformer()
    positions = coordinates(mol) @ np.asarray(transform, dtype=float).T
    positions = positions + np.asarray(translation, dtype=float)
    for i, point in enumerate(positions):
        conformer.SetAtomPosition(i, Point3D(*(float(value) for value in point)))
    return moved_mol


def rotation(seed):
    """A random proper rotation."""
    generator = np.random.RandomState(seed)
    matrix, upper = np.linalg.qr(generator.normal(size=(3, 3)))
    matrix = matrix * np.sign(np.diag(upper))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


def renumbered(mol, permutation):
    new_mol = Chem.RenumberAtoms(mol, [int(i) for i in permutation])
    new_mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(new_mol)
    return new_mol


def distance(mol, i, j):
    positions = coordinates(mol)
    return float(np.linalg.norm(positions[i] - positions[j]))


# --------------------------------------------------------------------------- #
# molecules
# --------------------------------------------------------------------------- #


def benzene_geometry(center=(0.0, 0.0, 0.0)):
    angles = np.deg2rad(np.arange(6) * 60.0)
    carbon = np.stack([1.397 * np.cos(angles), 1.397 * np.sin(angles), np.zeros(6)], axis=1)
    hydrogen = np.stack([2.481 * np.cos(angles), 2.481 * np.sin(angles), np.zeros(6)], axis=1)
    return np.concatenate([carbon, hydrogen]) + np.asarray(center, dtype=float)


def benzene_bonds():
    ring = [(i, (i + 1) % 6, DOUBLE if i % 2 == 0 else SINGLE) for i in range(6)]
    return ring + [(i, 6 + i, SINGLE) for i in range(6)]


def ion_cluster():
    """Five monoatomic fragments that need two refinement rounds.

    The bromine splits the two chlorines straight away.  The two fluorines are
    equidistant from the bromine and have the same distance multiset to the
    chlorine class as a whole, so they stay tied for as long as the chlorines
    are tied and can only be separated once the chlorines have been ordered.
    """
    root = np.sqrt(26.0)
    positions = [
        (0.0, 0.0, 0.0),  # 1 Br, the anchor
        (2.0, 0.0, 0.0),  # 2 Cl, closer to the anchor
        (0.0, 4.0, 0.0),  # 3 Cl
        (3.0, 1.0, root),  # 4 F, closer to chlorine 2
        (-1.0, 3.0, root),  # 5 F
    ]
    return build_molecule(["Br", "Cl", "Cl", "F", "F"], [], positions)


def chlorofluoromethane():
    """CH2ClF, whose two hydrogens are mirror images but not rotation images.

    The plane through carbon, chlorine and fluorine reflects one hydrogen onto
    the other, so every interatomic distance of the two agrees exactly.  No
    proper rotation exchanges them, because that would have to exchange the
    chlorine and the fluorine as well.
    """
    directions = np.array(
        [
            [0.0, 0.0, 0.0],  # 1 C
            [1.0, 1.0, 1.0],  # 2 Cl
            [1.0, -1.0, -1.0],  # 3 F
            [-1.0, 1.0, -1.0],  # 4 H
            [-1.0, -1.0, 1.0],  # 5 H
        ]
    )
    lengths = np.array([0.0, 1.77, 1.35, 1.09, 1.09])
    positions = directions / np.sqrt(3.0) * lengths[:, None]
    bonds = [(0, i, SINGLE) for i in (1, 2, 3, 4)]
    return build_molecule(["C", "Cl", "F", "H", "H"], bonds, positions)


def benzene():
    return build_molecule(["C"] * 6 + ["H"] * 6, benzene_bonds(), benzene_geometry())


def benzene_chloride():
    """Benzene next to a chloride placed off every symmetry element of the ring."""
    positions = np.concatenate([benzene_geometry(), [[2.0, 0.7, 1.3]]])
    return build_molecule(["C"] * 6 + ["H"] * 6 + ["Cl"], benzene_bonds(), positions)


def halobenzene(halogen):
    """Benzene with one hydrogen swapped for a halogen.

    Every halogen gets exactly the same geometry, so the only difference between
    these molecules is which member of the group carries the substituent.
    """
    positions = benzene_geometry()
    positions[6] = positions[6] / np.linalg.norm(positions[6]) * 3.10
    symbols = ["C"] * 6 + [halogen] + ["H"] * 5
    return build_molecule(symbols, benzene_bonds(), positions)


def aromatic_chlorobenzene(partner=None):
    """Chlorobenzene as a real aromatic ring, optionally facing a partner fragment.

    A Kekule ring would cheat: alternating bonds make the two ortho carbons
    differ topologically.  With aromatic bonds, as any sanitised molecule has
    them, the rotation by half a turn about the chlorine axis is a symmetry of
    the fragment and exchanges them, so the fragment alone cannot order them.

    The partner sits above the ring plane, off the mirror but at equal distance
    from both ortho carbons.
    """
    positions = benzene_geometry()
    positions[6] = positions[6] / np.linalg.norm(positions[6]) * 3.10
    symbols = ["C"] * 6 + ["Cl"] + ["H"] * 5
    if partner is not None:
        positions = np.concatenate([positions, [partner]])
        symbols = symbols + ["Ne"]
    ring = [(i, (i + 1) % 6, AROMATIC) for i in range(6)]
    bonds = ring + [(i, 6 + i, SINGLE) for i in range(6)]
    return build_molecule(symbols, bonds, positions, aromatic_atoms=range(6))


def stacked_chlorobenzene():
    return aromatic_chlorobenzene(partner=(0.4, 0.0, 3.3))


def kekulized_chlorobenzene(shift):
    """Chlorobenzene with the ring double bonds written from `shift` onwards.

    Both values of `shift` describe the same molecule.  A file only records which
    of the two resonance structures whoever wrote it happened to pick, so the two
    have to come out of the ordering the same way.
    """
    positions = benzene_geometry()
    positions[6] = positions[6] / np.linalg.norm(positions[6]) * 3.10
    ring = [(i, (i + 1) % 6, DOUBLE if (i + shift) % 2 == 0 else SINGLE) for i in range(6)]
    bonds = ring + [(i, 6 + i, SINGLE) for i in range(6)]
    return build_molecule(["C"] * 6 + ["Cl"] + ["H"] * 5, bonds, positions)


def chloride_pair():
    """Two chlorides that no invariant descriptor can tell apart."""
    return build_molecule(["Cl", "Cl"], [], [(1.0, 0.0, 0.0), (5.0, 0.0, 0.0)])


def anchored_chlorides():
    """An anchor plus two chlorides, ordered against the distance to the origin.

    Chloride 2 is the closer one to the bromine anchor but the farther one from
    the coordinate origin, so the two candidate rules disagree.
    """
    positions = [
        (0.0, 0.0, 10.0),  # 1 Br
        (0.0, 0.0, 6.0),  # 2 Cl, 4 from the anchor, 6 from the origin
        (0.0, 0.0, -1.0),  # 3 Cl, 11 from the anchor, 1 from the origin
    ]
    return build_molecule(["Br", "Cl", "Cl"], [], positions)


def benzene_and_water():
    """A large fragment far away and a small fragment on the origin."""
    water = np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.2400, 0.9266, 0.0]])
    positions = np.concatenate([benzene_geometry(center=(10.0, 0.0, 0.0)), water])
    bonds = benzene_bonds() + [(12, 13, SINGLE), (12, 14, SINGLE)]
    return build_molecule(["C"] * 6 + ["H"] * 6 + ["O", "H", "H"], bonds, positions)


MOLECULES = {
    "ion_cluster": ion_cluster,
    "chlorofluoromethane": chlorofluoromethane,
    "benzene": benzene,
    "benzene_chloride": benzene_chloride,
    "chloride_pair": chloride_pair,
    "anchored_chlorides": anchored_chlorides,
    "benzene_and_water": benzene_and_water,
    "aromatic_chlorobenzene": aromatic_chlorobenzene,
    "stacked_chlorobenzene": stacked_chlorobenzene,
}

# molecules that the rotation- and translation-invariant rungs resolve completely
INVARIANT_MOLECULES = [
    "ion_cluster",
    "chlorofluoromethane",
    "benzene_chloride",
    "anchored_chlorides",
    "benzene_and_water",
    "stacked_chlorobenzene",
]


# --------------------------------------------------------------------------- #
# basic guarantees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_index_is_a_permutation(name):
    mol = MOLECULES[name]()
    index = unique_index(mol)
    assert sorted(index) == list(range(mol.GetNumAtoms()))


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_renumbering_to_the_canonical_order_is_a_fixed_point(name):
    mol = MOLECULES[name]()
    canonical = Chem.RenumberAtoms(mol, [int(i) for i in np.argsort(unique_index(mol))])
    canonical.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(canonical)
    assert list(unique_index(canonical)) == list(range(mol.GetNumAtoms()))


@pytest.mark.parametrize("name", sorted(MOLECULES))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_independent_of_the_input_atom_order(name, seed):
    mol = MOLECULES[name]()
    permutation = np.random.RandomState(seed).permutation(mol.GetNumAtoms())
    assert canonical_labels(renumbered(mol, permutation)) == canonical_labels(mol)


@pytest.mark.parametrize("name", INVARIANT_MOLECULES)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_invariant_under_rotation_and_translation(name, seed):
    mol = MOLECULES[name]()
    translation = np.random.RandomState(seed + 100).uniform(-20.0, 20.0, size=3)
    assert canonical_labels(moved(mol, rotation(seed), translation)) == canonical_labels(mol)


@pytest.mark.parametrize("name", sorted(MOLECULES))
def test_fragments_partition_the_atoms_in_index_order(name):
    mol = MOLECULES[name]()
    order = canonical_order(mol)
    concatenated = np.concatenate(order.fragments)
    assert sorted(concatenated) == list(range(mol.GetNumAtoms()))
    assert list(order.index[concatenated]) == list(range(mol.GetNumAtoms()))


# --------------------------------------------------------------------------- #
# fragment ordering: chemistry first
# --------------------------------------------------------------------------- #


def test_fragment_chemistry_outranks_the_distance_to_the_origin():
    mol = benzene_and_water()
    order = canonical_order(mol)
    first_fragment = order.fragments[0]
    assert len(first_fragment) == 12
    assert {mol.GetAtomWithIdx(int(i)).GetSymbol() for i in first_fragment} == {"C", "H"}
    assert not order.fragment_placement_fallback


def test_fragment_chemistry_orders_monoatomic_fragments_by_element():
    labels = canonical_labels(ion_cluster())
    assert labels[0] == 1  # bromine before chlorine before fluorine
    assert set(labels[1:3]) == {2, 3}
    assert set(labels[3:]) == {4, 5}


# --------------------------------------------------------------------------- #
# fragment ordering: inter-fragment distances before the origin
# --------------------------------------------------------------------------- #


def test_inter_fragment_distance_outranks_the_distance_to_the_origin():
    mol = anchored_chlorides()
    # the two rules disagree for this geometry, which is the point of the test
    assert distance(mol, 0, 1) < distance(mol, 0, 2)
    assert np.linalg.norm(coordinates(mol)[1]) > np.linalg.norm(coordinates(mol)[2])

    order = canonical_order(mol)
    assert not order.fragment_placement_fallback
    assert canonical_labels(mol) == (1, 2, 3)


def test_tied_fragments_are_resolved_across_several_refinement_rounds():
    mol = ion_cluster()
    positions = coordinates(mol)
    bromine, first_chlorine, second_chlorine, first_fluorine, second_fluorine = range(5)

    # the bromine separates the chlorines
    assert distance(mol, bromine, first_chlorine) < distance(mol, bromine, second_chlorine)

    # but it says nothing about the fluorines, and neither does the chlorine
    # class as long as the chlorines themselves are still tied
    assert distance(mol, bromine, first_fluorine) == pytest.approx(
        distance(mol, bromine, second_fluorine)
    )
    to_chlorines = lambda fluorine: sorted(
        [distance(mol, first_chlorine, fluorine), distance(mol, second_chlorine, fluorine)]
    )
    assert to_chlorines(first_fluorine) == pytest.approx(to_chlorines(second_fluorine))

    # once the chlorines are ordered the fluorines differ in the first entry
    assert distance(mol, first_chlorine, first_fluorine) < distance(
        mol, first_chlorine, second_fluorine
    )

    order = canonical_order(mol)
    assert not order.fragment_placement_fallback
    assert canonical_labels(mol) == (1, 2, 3, 4, 5)

    # the same holds for a placement that reverses every distance to the origin
    far_away = moved(mol, rotation(7), (30.0, -12.0, 5.0))
    assert canonical_labels(far_away) == (1, 2, 3, 4, 5)


def test_origin_is_used_only_once_the_invariants_are_exhausted():
    mol = chloride_pair()
    order = canonical_order(mol)
    assert order.fragment_placement_fallback
    # nothing but the placement is left, so the closer fragment comes first
    assert canonical_labels(mol) == (1, 2)


# --------------------------------------------------------------------------- #
# atom ordering inside a fragment
# --------------------------------------------------------------------------- #


HALOGENS = ["F", "Cl", "Br", "I"]


@pytest.mark.parametrize("halogen", HALOGENS)
def test_halogen_of_a_halobenzene_lands_at_the_same_index(halogen):
    mol = halobenzene(halogen)
    # the halogen is the only atom of its group and the group leads the chemistry,
    # so it heads the ordering whichever halogen it happens to be
    assert unique_index(mol)[6] == 0


def test_whole_halobenzene_order_is_the_same_across_the_group():
    orders = {halogen: tuple(unique_index(halobenzene(halogen))) for halogen in HALOGENS}
    assert len(set(orders.values())) == 1, orders


def test_group_outranks_the_atomic_number():
    """A pnictogen sits behind a chalcogen even when it is the heavier atom.

    Ordering by atomic number would put phosphorus before oxygen but nitrogen
    behind it, so the group of an atom would depend on its partner.
    """
    positions = [(0.0, 0.0, 0.0), (1.4, 0.0, 0.0)]
    for pnictogen in ("N", "P"):
        mol = build_molecule([pnictogen, "O"], [(0, 1, SINGLE)], positions)
        index = unique_index(mol)
        assert index[1] == 0, pnictogen  # the oxygen leads
        assert index[0] == 1, pnictogen


def test_same_group_elements_stay_distinguishable():
    """Sharing a group must not make two elements interchangeable."""
    positions = [(0.0, 0.0, 0.0), (1.7, 0.0, 0.0), (-1.7, 0.2, 0.0)]
    mol = build_molecule(["C", "Cl", "Br"], [(0, 1, SINGLE), (0, 2, SINGLE)], positions)
    order = canonical_order(mol)
    assert not order.atom_placement_fallback
    # the heavier halogen first, then the lighter one, then the carbon
    assert list(order.index) == [2, 1, 0]


def test_bond_order_is_part_of_the_atom_chemistry():
    """The two outer carbons of a chain differ only in the order of their bond."""
    positions = [(0.0, 0.0, 0.0), (1.5, 0.0, 0.0), (2.8, 0.6, 0.0)]
    mixed = build_molecule(["C"] * 3, [(0, 1, SINGLE), (1, 2, DOUBLE)], positions)
    uniform = build_molecule(["C"] * 3, [(0, 1, SINGLE), (1, 2, SINGLE)], positions)

    assert _weisfeiler_lehman_labels(mixed)[0] != _weisfeiler_lehman_labels(mixed)[2]
    assert _weisfeiler_lehman_labels(uniform)[0] == _weisfeiler_lehman_labels(uniform)[2]


def test_atom_chemistry_outranks_geometry_inside_a_fragment():
    mol = benzene_chloride()
    order = canonical_order(mol)
    symbols = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in order.fragments[0]]
    assert symbols == ["C"] * 6 + ["H"] * 6


def test_symmetric_ring_is_resolved_by_a_neighbouring_fragment():
    mol = benzene_chloride()
    positions = coordinates(mol)
    chloride = positions[12]

    # a bare ring leaves every carbon equivalent, so this can only work through
    # the distances to the chloride, which have to be distinct for the test
    to_chloride = np.round(np.linalg.norm(positions[:6] - chloride, axis=1), 4)
    assert len(set(to_chloride)) == 6

    order = canonical_order(mol)
    assert not order.atom_placement_fallback

    carbons = order.fragments[0][:6]
    assert list(carbons) == list(np.argsort(to_chloride))

    hydrogens = order.fragments[0][6:]
    to_chloride_h = np.linalg.norm(positions[6:12] - chloride, axis=1)
    assert list(hydrogens - 6) == list(np.argsort(to_chloride_h))


def test_isolated_symmetric_ring_falls_back_to_the_placement():
    mol = benzene()
    order = canonical_order(mol)
    assert order.atom_placement_fallback
    assert sorted(order.index) == list(range(12))


def test_mirror_image_hydrogens_are_separated_without_using_the_placement():
    mol = chlorofluoromethane()
    positions = coordinates(mol)
    first_hydrogen, second_hydrogen = 3, 4

    # every interatomic distance of the two hydrogens agrees exactly
    for other in range(3):
        assert distance(mol, first_hydrogen, other) == pytest.approx(
            distance(mol, second_hydrogen, other)
        )

    order = canonical_order(mol)
    assert not order.atom_placement_fallback
    assert order.index[first_hydrogen] != order.index[second_hydrogen]


def test_reflection_exchanges_mirror_image_hydrogens():
    mol = chlorofluoromethane()
    reflected = moved(mol, np.diag([1.0, 1.0, -1.0]))

    labels = canonical_labels(mol)
    # a reflection maps the molecule onto a rotated copy of itself with the two
    # hydrogens exchanged, so the canonical order has to exchange them as well
    expected = tuple(5 if value == 4 else 4 if value == 5 else value for value in labels)
    assert canonical_labels(reflected) == expected
    assert canonical_labels(reflected) != labels


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_does_not_exchange_mirror_image_hydrogens(seed):
    mol = chlorofluoromethane()
    rotated = moved(mol, rotation(seed), np.random.RandomState(seed).uniform(-5.0, 5.0, size=3))
    assert canonical_labels(rotated) == canonical_labels(mol)


# --------------------------------------------------------------------------- #
# planar rings
# --------------------------------------------------------------------------- #


def test_the_written_kekule_structure_does_not_change_the_order():
    """The bond orders of a ring record a resonance structure, not a difference.

    Two files of one docked ligand can disagree on where the double bonds sit,
    and reading them literally hands the same ring two different orders.
    """
    first = kekulized_chlorobenzene(0)
    second = kekulized_chlorobenzene(1)

    ring_bonds = lambda mol: sorted(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() < 6 and b.GetEndAtomIdx() < 6
    )
    assert ring_bonds(first) != ring_bonds(second)
    assert canonical_labels(first) == canonical_labels(second)


def test_a_kekulized_ring_and_an_aromatic_ring_agree():
    assert canonical_labels(kekulized_chlorobenzene(0)) == canonical_labels(
        aromatic_chlorobenzene()
    )


def test_a_planar_ring_cannot_break_its_own_mirror():
    """The premise of the two rungs below: the fragment alone runs out of descriptors.

    Half a turn about the chlorine axis is a proper symmetry of the flat ring and
    exchanges the two ortho carbons, so they share every distance, and the class
    centroids all sit on that axis, so no frame can be built from the ring.
    """
    mol = aromatic_chlorobenzene()
    positions = coordinates(mol)
    chemistry = _weisfeiler_lehman_labels(mol)

    assert chemistry[1] == chemistry[5]
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    assert np.allclose(np.sort(distances[1]), np.sort(distances[5]))
    assert canonical_order(mol).atom_placement_fallback


def test_a_partner_fragment_breaks_the_ring_mirror():
    mol = stacked_chlorobenzene()
    positions = coordinates(mol)

    # the partner is off the mirror plane, yet exactly equidistant from the two
    # ortho carbons, so no distance to it can order them either
    assert np.linalg.norm(positions[1] - positions[12]) == pytest.approx(
        np.linalg.norm(positions[5] - positions[12])
    )
    # anchoring the frame on the partner turns that into a difference in sign
    order = canonical_order(mol)
    assert not order.atom_placement_fallback
    assert order.index[1] != order.index[5]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_stacked_ring_order_survives_rigid_motion(seed):
    mol = stacked_chlorobenzene()
    translation = np.random.RandomState(seed + 11).uniform(-15.0, 15.0, size=3)
    assert canonical_labels(moved(mol, rotation(seed), translation)) == canonical_labels(mol)


def test_moving_the_partner_through_the_ring_plane_exchanges_the_ortho_carbons():
    """The rung has to be sensitive to the side the partner sits on, and only that."""
    above = canonical_order(aromatic_chlorobenzene(partner=(0.4, 0.0, 3.3))).index
    below = canonical_order(aromatic_chlorobenzene(partner=(0.4, 0.0, -3.3))).index

    assert above[1] == below[5] and above[5] == below[1]
    unaffected = [0, 3, 6, 9, 12]  # the atoms on the mirror plane keep their place
    assert list(above[unaffected]) == list(below[unaffected])


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_isolated_ring_keeps_its_symmetry_orbits_in_place(seed):
    """What an exchanged pair can still promise once every descriptor is spent.

    No ordering of the two ortho carbons is invariant under rigid motion, since
    a symmetry of the ring swaps them and therefore forces every invariant
    descriptor to agree on them.  The indices the pair occupies stay fixed, only
    the choice of which carbon takes which does not.
    """
    mol = aromatic_chlorobenzene()
    chemistry = _weisfeiler_lehman_labels(mol)

    def orbits(molecule):
        index = unique_index(molecule)
        return {c: tuple(sorted(index[chemistry == c])) for c in np.unique(chemistry)}

    translation = np.random.RandomState(seed + 5).uniform(-15.0, 15.0, size=3)
    assert orbits(moved(mol, rotation(seed), translation)) == orbits(mol)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_placement_in_an_internal_frame_survives_rigid_motion(seed):
    """The last rung stops depending on the laboratory frame once it has anchors."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [0.3, 2.0, 0.4], [-1.0, 0.5, 1.7], [2.2, -0.8, 0.9]]
    )
    matrix = rotation(seed)
    moved_points = points @ matrix.T + np.array([9.0, -4.0, 2.5])

    anchored = _placement_descriptor(points, 4, points[:3])
    moved_anchored = _placement_descriptor(moved_points, 4, moved_points[:3])
    assert np.allclose(anchored, moved_anchored, atol=1e-3)

    # without anchors the very same rung moves with the molecule
    assert not np.allclose(
        _placement_descriptor(points, 4), _placement_descriptor(moved_points, 4), atol=1e-3
    )


def test_placement_falls_back_to_the_laboratory_frame_for_degenerate_anchors():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(
        _placement_descriptor(points, 4, points), _placement_descriptor(points, 4)
    )


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #


def test_single_atom():
    mol = build_molecule(["C"], [], [(1.0, 2.0, 3.0)])
    assert list(unique_index(mol)) == [0]


def test_empty_molecule():
    order = canonical_order(Chem.Mol())
    assert list(order.index) == []
    assert order.fragments == []


def test_coincident_atoms_still_yield_a_permutation():
    mol = build_molecule(["Cl", "Cl"], [], [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    assert sorted(unique_index(mol)) == [0, 1]
