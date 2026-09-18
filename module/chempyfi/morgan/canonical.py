"""Canonical atom ordering built as a ladder of tie-breakers.

Every rung of the ladder is consulted only for the atoms or fragments that the
rungs above it left tied, and all rungs that are invariant under rotation and
translation come before the ones that are not.  Note that interatomic distances
are invariant under reflection as well, so a distance-only ladder is E(3)- and
not SE(3)-complete; the signed volume rung is what removes the reflection.

Fragments (connected components) are ordered by

1. fragment chemistry -- size plus the Weisfeiler-Lehman refinement of the atom
   chemistry, so graph topology and element/bond identity.  Elements are led by
   their periodic group rather than their atomic number, which keeps a
   substituent at the same index across the group,
2. the distances to the *other* fragments, visited in the order of the fragment
   classes that are already distinguishable from the tied group and iterated to
   a fixed point, so that a group whose members only become separable after some
   other group has been split is still resolved,
3. signed volumes of the fragment centroids against a canonical frame,
4. the distance of the fragment to the coordinate origin -- SE(3)-dependent.

Atoms inside a fragment follow the same principle

1. atom chemistry refined by Weisfeiler-Lehman relaxation,
2. intra-fragment distance geometry, again with the tied class visited last,
3. inter-fragment distance geometry,
4. signed volumes against a frame built from the fragment itself, which is what
   separates atoms related by a mirror image but not by any rotation,
5. signed volumes against a frame anchored on the partners of the complex, which
   is what separates the two sides of a planar ring,
6. the position in an internal frame, and only in the laboratory frame when no
   internal frame exists -- the SE(3)-dependent rung.

The last rung is reached only for a group whose members some proper symmetry of
the structure exchanges.  For such a group no ordering exists that is invariant
under rigid motion, because an isometry that maps the structure onto itself while
swapping two atoms forces every invariant descriptor to agree on them.  What does
stay fixed is which indices the group occupies, not which of its members takes
which index.

``decimals`` controls the tolerance at which two distances count as equal.  Ties
that survive even the coordinates are broken by the input atom index, which
guarantees that the result is always a permutation; this can only happen for
atoms that share a position.

Distances are only ever computed from the atoms or fragments that are still tied,
so a molecule whose chemistry already separates everything costs nothing beyond
the refinement, and a large structure only pays for its symmetric groups.  A
tied group big enough that its distances would exceed ``max_distance_cells``
skips the invariant geometry and drops to the placement rung instead, which
keeps the ordering usable on systems with hundreds of thousands of atoms at the
price of a coarser tie-break.
"""

import logging
from collections import namedtuple

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

DEFAULT_DECIMALS = 4

# roughly 270 MB worth of float64 distances for a single descriptor
DEFAULT_MAX_DISTANCE_CELLS = 2 ** 25

# below this a difference of lengths or cross products counts as degenerate
_DEGENERACY_TOLERANCE = 1e-6

CanonicalOrder = namedtuple(
    "CanonicalOrder",
    ["index", "fragments", "fragment_placement_fallback", "atom_placement_fallback"],
)


def canonical_order(mol, decimals=DEFAULT_DECIMALS, max_distance_cells=DEFAULT_MAX_DISTANCE_CELLS):
    """Order the atoms of `mol` canonically.

    Args:
        mol (rdkit.Chem.Mol): Molecule with at least one conformer.
        decimals (int): Tolerance for comparing distances, as a number of decimals.
        max_distance_cells (int): Largest number of pairwise distances a single
            descriptor may materialize before the invariant geometry is skipped.

    Returns:
        CanonicalOrder: ``index`` holds the canonical position of every atom,
        ``fragments`` the atom indices per fragment in canonical order, and the
        two flags report whether the SE(3)-dependent rungs had to be consulted.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return CanonicalOrder(np.zeros(0, dtype=int), [], False, False)

    _ensure_cached_properties(mol)
    positions = np.array(mol.GetConformer().GetPositions(), dtype=float)
    chemistry = _weisfeiler_lehman_labels(_with_perceived_aromaticity(mol), decimals)
    fragments = _connected_components(mol)
    budget = _Budget(max_distance_cells)

    fragment_labels, fragment_fallback = _fragment_order(
        fragments, positions, chemistry, decimals, budget
    )
    ordered_fragments = [fragments[i] for i in np.argsort(fragment_labels)]

    # the inter-fragment rung between atoms costs a pass over the whole molecule for
    # every fragment whose chemistry is not already conclusive, so it is priced once
    # for all of them rather than per fragment
    unresolved = sum(
        1 for fragment in fragments if len(np.unique(chemistry[fragment])) < len(fragment)
    )
    allow_inter = bool(unresolved) and budget.allows(
        unresolved * n_atoms, "inter-fragment distance geometry between atoms"
    )

    index = np.zeros(n_atoms, dtype=int)
    atom_fallback = False
    offset = 0
    canonical_fragments = []
    for rank, fragment in enumerate(ordered_fragments):
        labels, fallback = _atom_order(
            rank, ordered_fragments, positions, chemistry, decimals, budget, allow_inter
        )
        atom_fallback = atom_fallback or fallback
        index[fragment] = labels + offset
        offset += len(fragment)
        canonical_fragments.append(fragment[np.argsort(labels)])

    return CanonicalOrder(index, canonical_fragments, fragment_fallback, atom_fallback)


class _Budget:
    """Caps how many pairwise distances a single descriptor may materialize."""

    def __init__(self, max_cells):
        self.max_cells = max_cells
        self._reported = set()

    def allows(self, cells, description):
        if cells <= self.max_cells:
            return True
        if description not in self._reported:
            self._reported.add(description)
            logging.warning(
                "Skipping the %s of the canonical atom order: it would need %d pairwise "
                "distances and max_distance_cells is %d. The atoms and fragments that this "
                "descriptor would have separated fall back to their position in space.",
                description,
                cells,
                self.max_cells,
            )
        return False


# --------------------------------------------------------------------------- #
# generic class refinement
# --------------------------------------------------------------------------- #


def _dense_rank(keys):
    """Rank the rows of `keys` lexicographically, smallest row first, ties shared.

    This is what ``numpy.unique(axis=0)`` would return as its inverse, but a
    plain lexsort avoids the structured-array detour and the refinement loops
    below call this on every pass.
    """
    keys = np.asarray(keys, dtype=float)
    if keys.ndim == 1:
        keys = keys.reshape(-1, 1)
    n_items = keys.shape[0]
    if n_items == 0:
        return np.zeros(0, dtype=int)

    # lexsort treats the last key as the primary one
    order = np.lexsort(keys.T[::-1])
    in_order = keys[order]
    boundary = np.empty(n_items, dtype=bool)
    boundary[0] = True
    if n_items > 1:
        boundary[1:] = (in_order[1:] != in_order[:-1]).any(axis=1)

    ranks = np.empty(n_items, dtype=int)
    ranks[order] = np.cumsum(boundary) - 1
    return ranks


def _n_classes(labels):
    """How many classes a dense label array holds."""
    return int(labels.max()) + 1 if len(labels) else 0


def _class_groups(labels):
    """The members of every class, in ascending class order.

    Grouping through a single sort rather than one mask per class keeps this
    linear-ish even when nearly every item sits in a class of its own.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    order = np.argsort(labels, kind="stable")
    in_order = labels[order]
    starts = np.flatnonzero(np.concatenate([[True], in_order[1:] != in_order[:-1]]))
    ends = np.concatenate([starts[1:], [len(order)]])
    return [(in_order[start], order[start:end]) for start, end in zip(starts, ends)]


def _tied_rows(labels):
    """The items that still share their class with somebody else."""
    labels = np.asarray(labels)
    counts = np.bincount(labels)
    return np.flatnonzero(counts[labels] > 1)


def _scatter_rows(descriptor, rows, n_items):
    """Put a descriptor computed for the tied items back into a full array.

    An item that is alone in its class cannot be split any further, so a
    constant stands in for it.
    """
    full = np.zeros((n_items, descriptor.shape[1]))
    full[rows] = descriptor
    return full


def _refine(labels, descriptor):
    """Split the classes of `labels` by `descriptor` without reordering the classes."""
    labels = np.asarray(labels, dtype=float).reshape(-1, 1)
    descriptor = np.asarray(descriptor, dtype=float)
    if descriptor.ndim == 1:
        descriptor = descriptor.reshape(-1, 1)
    return _dense_rank(np.column_stack([labels, descriptor]))


def _refine_to_fixed_point(labels, descriptor_fn):
    """Apply `descriptor_fn` until it stops splitting classes.

    The descriptor is rebuilt from the current labels on every pass, which is
    what lets a class become separable only after another class has been split.
    ``descriptor_fn`` may return ``None`` to signal that it cannot contribute.
    """
    labels = np.asarray(labels, dtype=int)
    n_items = len(labels)
    while True:
        n_classes = _n_classes(labels)
        if n_classes == n_items:
            return labels
        descriptor = descriptor_fn(labels)
        if descriptor is None or np.asarray(descriptor).size == 0:
            return labels
        refined = _refine(labels, descriptor)
        if _n_classes(refined) == n_classes:
            return labels
        labels = refined


def _order(initial_keys, invariant_stages, placement_stages, tie_break):
    """Run the ladder and report whether the SE(3)-dependent stages were needed."""
    labels = _dense_rank(initial_keys)
    n_items = len(labels)
    for stage in invariant_stages:
        if _n_classes(labels) == n_items:
            break
        labels = _refine_to_fixed_point(labels, stage)

    used_placement = _n_classes(labels) != n_items
    if used_placement:
        for stage in placement_stages:
            if _n_classes(labels) == n_items:
                break
            labels = _refine_to_fixed_point(labels, stage)

    tie_break = np.asarray(tie_break, dtype=float)
    return _dense_rank(np.column_stack([labels, tie_break])), used_placement


# --------------------------------------------------------------------------- #
# SE(3)-invariant descriptors
# --------------------------------------------------------------------------- #


def _grouped_distance_descriptor(distances, column_labels, row_class=None):
    """Sorted distances from every row to the members of each column class.

    Classes are visited in ascending label order so that the comparison follows
    the chemistry-derived class order.  The class a row belongs to itself is
    moved past all others, because a tie has to be attacked with the partners
    that are already distinguishable from the tied group before the members of
    the group are allowed to speak.
    """
    distances = np.asarray(distances, dtype=float)
    n_rows, n_columns = distances.shape
    if n_columns == 0:
        return None

    groups = _class_groups(column_labels)
    largest_class = max(len(members) for _, members in groups)

    row_class = np.full(n_rows, -1) if row_class is None else np.asarray(row_class)
    other = np.full((n_rows, n_columns), np.inf)
    own = np.full((n_rows, largest_class), np.inf)

    offset = 0
    for label, members in groups:
        block = np.sort(distances[:, members], axis=1)
        in_class = row_class == label
        other[~in_class, offset:offset + len(members)] = block[~in_class]
        own[in_class, :len(members)] = block[in_class]
        offset += len(members)

    return np.column_stack([other, own])


def _class_centroids(points, labels):
    """One point per class, in ascending class order."""
    return np.array([points[members].mean(axis=0) for _, members in _class_groups(labels)])


def _first_independent_triple(candidates):
    """The first three candidate points that span a plane, in the order given."""
    candidates = np.asarray(candidates, dtype=float)
    if len(candidates) < 3:
        return None

    origin = candidates[0]
    first = None
    for candidate in candidates[1:]:
        offset = candidate - origin
        if np.linalg.norm(offset) < _DEGENERACY_TOLERANCE:
            continue
        if first is None:
            first = candidate
            continue
        if np.linalg.norm(np.cross(first - origin, offset)) > _DEGENERACY_TOLERANCE:
            return origin, first, candidate
    return None


def _chirality_descriptor(points, candidates, decimals):
    """Signed volumes against a frame taken from `candidates`, or None if degenerate.

    The value is invariant under rotation and translation but changes sign under
    a reflection, which is what distinguishes points that distance geometry
    alone cannot tell apart.  Feeding in anchors from outside the fragment lets
    the partners of a complex break a mirror that the fragment cannot break on
    its own, and it does so without leaving SE(3).
    """
    triple = _first_independent_triple(candidates)
    if triple is None:
        return None
    origin, first, second = triple
    normal = np.cross(first - origin, second - origin)
    return np.round((points - origin) @ normal, decimals).reshape(-1, 1)


def _internal_frame(candidates):
    """An orthonormal frame anchored on `candidates`, or None if they are degenerate."""
    triple = _first_independent_triple(candidates)
    if triple is None:
        return None
    origin, first, second = triple
    forward = first - origin
    forward = forward / np.linalg.norm(forward)
    normal = np.cross(forward, second - origin)
    normal = normal / np.linalg.norm(normal)
    return origin, np.stack([forward, np.cross(normal, forward), normal])


def _frame_origin(candidates):
    frame = None if candidates is None else _internal_frame(candidates)
    return np.zeros(3) if frame is None else frame[0]


def _placement_descriptor(points, decimals, candidates=None):
    """Where a point sits, measured in an internal frame whenever one exists.

    This is the only rung that depends on where the structure happens to lie in
    space, so it is worth spending a frame built from the structure itself to
    keep it from mattering.  Falling back to the laboratory frame is a real loss
    of invariance, but it only happens when the candidates are degenerate, and a
    group whose members a proper symmetry exchanges cannot be ordered by any
    frame anyway: no function of the geometry tells its members apart.
    """
    points = np.asarray(points, dtype=float)
    frame = None if candidates is None else _internal_frame(candidates)
    if frame is not None:
        origin, axes = frame
        points = (points - origin) @ axes.T
    return np.column_stack(
        [np.round(np.linalg.norm(points, axis=1), decimals), np.round(points, decimals)]
    )


# --------------------------------------------------------------------------- #
# chemistry
# --------------------------------------------------------------------------- #


def _ensure_cached_properties(mol):
    """Make the chemistry queries used below safe on unsanitized molecules."""
    if mol.NeedsUpdatePropertyCache():
        mol.UpdatePropertyCache(strict=False)
    try:
        initialized = mol.GetRingInfo().IsInitialized()
    except (AttributeError, RuntimeError):
        initialized = False
    if not initialized:
        Chem.FastFindRings(mol)


_AROMATICITY_OPS = Chem.SanitizeFlags.SANITIZE_SYMMRINGS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY


def _with_perceived_aromaticity(mol):
    """A copy of `mol` whose aromatic rings are flagged, or `mol` if that fails.

    The two Kekule structures of one ring are the same molecule, but they write
    the double bonds in different places, so chemistry read straight off the bond
    orders makes the same ring look different depending on which structure a file
    happens to carry.  Perceiving aromaticity collapses both onto one answer.
    Anything that will not sanitize keeps its literal bond orders, which is no
    worse than reading them directly.
    """
    working = Chem.Mol(mol)
    try:
        failed = Chem.SanitizeMol(working, sanitizeOps=_AROMATICITY_OPS, catchErrors=True)
    except Exception:
        return mol
    if failed != Chem.SanitizeFlags.SANITIZE_NONE:
        return mol
    _ensure_cached_properties(working)
    return working


_PERIODIC_TABLE = Chem.GetPeriodicTable()
_OUTER_ELECTRONS = {}


def _periodic_group(atomic_number):
    """The valence electron count, which is the periodic group of a main-group element.

    Leading with this rather than with the atomic number is what keeps a
    substituent from a given group at the same canonical index no matter which
    member of the group it is, so the halogen of a halobenzene always lands in
    the same place.  For the d-block it is only an approximation of the group,
    but the atomic number still follows as a tie-break, so elements are never
    conflated -- only their position in the order is shared.
    """
    if atomic_number not in _OUTER_ELECTRONS:
        _OUTER_ELECTRONS[atomic_number] = _PERIODIC_TABLE.GetNOuterElecs(atomic_number)
    return _OUTER_ELECTRONS[atomic_number]


def _atom_invariants(mol):
    """Per-atom chemistry, signed so that a smaller row sorts first."""
    rows = []
    for atom in mol.GetAtoms():
        rows.append(
            [
                -_periodic_group(atom.GetAtomicNum()),
                -atom.GetAtomicNum(),
                -atom.GetTotalDegree(),
                -atom.GetTotalNumHs(includeNeighbors=True),
                atom.GetFormalCharge(),
                0 if atom.GetIsAromatic() else 1,
                0 if atom.IsInRing() else 1,
                atom.GetIsotope(),
            ]
        )
    return np.array(rows, dtype=float)


def _weisfeiler_lehman_labels(mol, decimals=DEFAULT_DECIMALS):
    """Refine the atom chemistry by the multiset of bonded neighbour classes.

    The refinement runs over the whole molecule at once.  There are no bonds
    between fragments, so each fragment is refined on its own while all
    fragments end up sharing one comparable label space.
    """
    n_atoms = mol.GetNumAtoms()
    initial = _dense_rank(_atom_invariants(mol))

    # Collect the bonds one atom at a time.  Reaching them through the molecule,
    # by index or by iterating mol.GetBonds(), scans the whole bond graph on every
    # lookup and turns this into a quadratic cost on large structures.  Walking the
    # atoms instead visits every bond once from either end, which is exactly the
    # sparse list of directed edges the refinement below wants.
    sources = []
    targets = []
    keys = []
    for atom in mol.GetAtoms():
        index = atom.GetIdx()
        for bond in atom.GetBonds():
            sources.append(index)
            targets.append(bond.GetOtherAtomIdx(index))
            keys.append(bond.GetBondTypeAsDouble())
    if not sources:
        return initial

    sources = np.array(sources, dtype=int)
    targets = np.array(targets, dtype=int)
    # only a handful of bond types occur, so a neighbour class and a bond type pack
    # into one number whose order is still the order of the pair
    bond_types, bond_index = np.unique(-np.round(keys, decimals), return_inverse=True)
    n_bond_types = len(bond_types)
    bond_index = np.asarray(bond_index).reshape(-1)

    width = int(np.bincount(sources, minlength=n_atoms).max())

    def descriptor(labels):
        # an atom that is already alone in its class can never split again, so only
        # the atoms that are still tied need a neighbour descriptor.  On a large
        # structure that set collapses after a few rounds
        tied = _tied_rows(labels)
        if len(tied) == 0:
            return None
        row_of_atom = np.full(n_atoms, -1)
        row_of_atom[tied] = np.arange(len(tied))

        edges = np.flatnonzero(row_of_atom[sources] >= 0)
        if len(edges) == 0:
            return None
        packed = labels[targets[edges]] * n_bond_types + bond_index[edges]
        order = np.lexsort((packed, sources[edges]))

        # group the edges by atom, each group sorted by neighbour class and bond
        rows = row_of_atom[sources[edges][order]]
        starts = np.flatnonzero(np.concatenate([[True], rows[1:] != rows[:-1]]))
        lengths = np.diff(np.append(starts, len(rows)))
        neighbours = np.full((len(tied), width), np.inf)
        neighbours[rows, np.arange(len(rows)) - np.repeat(starts, lengths)] = packed[order]

        # collapse the neighbour multiset to a single rank, which keeps the pass that
        # follows over all atoms down to one sort key
        column = np.zeros((n_atoms, 1))
        column[tied, 0] = _dense_rank(neighbours)
        return column

    return _refine_to_fixed_point(initial, descriptor)


def _connected_components(mol):
    """Fragments as sorted arrays of atom indices."""
    n_atoms = mol.GetNumAtoms()
    seen = np.zeros(n_atoms, dtype=bool)
    components = []
    for start in range(n_atoms):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbour in mol.GetAtomWithIdx(index).GetNeighbors():
                other = neighbour.GetIdx()
                if not seen[other]:
                    seen[other] = True
                    stack.append(other)
        components.append(np.sort(np.array(component, dtype=int)))
    return components


# --------------------------------------------------------------------------- #
# distances, computed only where a tie has to be broken
# --------------------------------------------------------------------------- #


def _distance_block(positions, rows, columns, decimals):
    return np.round(cdist(positions[rows], positions[columns]), decimals)


def _contact_distances(rows, fragments, positions, decimals):
    """Closest approach from the fragments in `rows` to every fragment.

    Reducing one fragment against all atoms at a time keeps the peak allocation
    at the size of a single fragment block rather than the whole molecule.
    """
    sizes = [len(fragment) for fragment in fragments]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
    all_atoms = np.arange(len(positions))

    by_fragment = np.empty(len(positions), dtype=int)
    for start, fragment in zip(starts, fragments):
        by_fragment[start : start + len(fragment)] = fragment

    contact = np.empty((len(rows), len(fragments)))
    for row, i in enumerate(rows):
        closest = _distance_block(positions, fragments[i], all_atoms, decimals).min(axis=0)
        contact[row] = np.minimum.reduceat(closest[by_fragment], starts)
        contact[row, i] = np.inf
    return contact


def _mask_self(distances, rows):
    """Blank out the entry that pairs a row with itself."""
    distances[np.arange(len(rows)), rows] = np.inf
    return distances


# --------------------------------------------------------------------------- #
# the two ladders
# --------------------------------------------------------------------------- #


def _fragment_signatures(fragments, chemistry):
    """Size and refined atom chemistry, the larger fragment first.

    Size is the primary key, so the chemistry only ever has to be compared
    between fragments that hold the same number of atoms.  Ranking the sorted
    chemistry per size group keeps this linear in the number of atoms, whereas
    padding every signature to the largest fragment would cost the number of
    fragments times the size of the largest one.
    """
    sizes = np.array([len(fragment) for fragment in fragments])
    signatures = np.empty((len(fragments), 2))
    signatures[:, 0] = -sizes
    for size in np.unique(sizes):
        members = np.where(sizes == size)[0]
        block = np.array([np.sort(chemistry[fragments[i]]) for i in members])
        signatures[members, 1] = _dense_rank(block)
    return signatures


def _fragment_order(fragments, positions, chemistry, decimals, budget):
    centroids = np.array([positions[fragment].mean(axis=0) for fragment in fragments])

    def geometry(labels):
        rows = _tied_rows(labels)
        if len(rows) == 0:
            return None
        if not budget.allows(len(rows) * len(positions), "inter-fragment distance geometry"):
            return None
        contact = _contact_distances(rows, fragments, positions, decimals)
        centroid = _mask_self(np.round(cdist(centroids[rows], centroids), decimals), rows)
        blocks = [
            _grouped_distance_descriptor(contact, labels, labels[rows]),
            _grouped_distance_descriptor(centroid, labels, labels[rows]),
        ]
        if any(block is None for block in blocks):
            return None
        return _scatter_rows(np.column_stack(blocks), rows, len(fragments))

    def chirality(labels):
        return _chirality_descriptor(centroids, _class_centroids(centroids, labels), decimals)

    def placement(labels):
        candidates = _class_centroids(centroids, labels)
        origin = _frame_origin(candidates)
        closest = np.round(
            [np.linalg.norm(positions[f] - origin, axis=1).min() for f in fragments], decimals
        )
        return np.column_stack(
            [closest, _placement_descriptor(centroids, decimals, candidates)]
        )

    return _order(
        _fragment_signatures(fragments, chemistry),
        [geometry, chirality],
        [placement],
        np.array([fragment.min() for fragment in fragments], dtype=float),
    )


def _atom_order(rank, ordered_fragments, positions, chemistry, decimals, budget, allow_inter):
    fragment = ordered_fragments[rank]
    fragment_positions = positions[fragment]
    n_own = len(fragment)
    n_outside = len(positions) - n_own

    def intra(labels):
        rows = _tied_rows(labels)
        if len(rows) == 0:
            return None
        if not budget.allows(len(rows) * n_own, "intra-fragment distance geometry"):
            return None
        distances = _mask_self(
            _distance_block(positions, fragment[rows], fragment, decimals), rows
        )
        descriptor = _grouped_distance_descriptor(distances, labels, labels[rows])
        return None if descriptor is None else _scatter_rows(descriptor, rows, n_own)

    # group the outside atoms by canonical fragment rank and then by chemistry, both
    # of which are fixed before any atom inside this fragment is ordered.  Several
    # rungs want this, and building it costs a pass over the rest of the molecule
    outside_cache = []

    def outside_groups():
        if not outside_cache:
            outside = ordered_fragments[:rank] + ordered_fragments[rank + 1 :]
            columns = np.concatenate(outside)
            ranks = np.concatenate([np.full(len(f), i) for i, f in enumerate(outside)])
            groups = _dense_rank(np.column_stack([ranks, chemistry[columns]]))
            outside_cache.append((columns, groups, _class_centroids(positions[columns], groups)))
        return outside_cache[0]

    def inter(labels):
        rows = _tied_rows(labels)
        if len(rows) == 0:
            return None
        columns, groups, _ = outside_groups()
        distances = _distance_block(positions, fragment[rows], columns, decimals)
        descriptor = _grouped_distance_descriptor(distances, groups)
        return None if descriptor is None else _scatter_rows(descriptor, rows, n_own)

    def chirality(labels):
        candidates = _class_centroids(fragment_positions, labels)
        return _chirality_descriptor(fragment_positions, candidates, decimals)

    def complex_chirality(labels):
        # a planar ring cannot break its own mirror, because the rotation that
        # exchanges two of its atoms is a symmetry of the fragment.  Anchoring the
        # frame on the partners of the complex breaks it, and every partner that
        # sits off the mirror plane does so even when its distances to the two
        # atoms are exactly equal
        own = _class_centroids(fragment_positions, labels)
        candidates = np.concatenate([own, outside_groups()[2]])
        return _chirality_descriptor(fragment_positions, candidates, decimals)

    def placement(labels):
        candidates = _class_centroids(fragment_positions, labels)
        if allow_inter and n_outside:
            candidates = np.concatenate([candidates, outside_groups()[2]])
        return _placement_descriptor(fragment_positions, decimals, candidates)

    stages = [intra]
    if allow_inter and n_outside:
        stages.extend([inter, chirality, complex_chirality])
    else:
        stages.append(chirality)

    return _order(
        chemistry[fragment], stages, [placement], np.asarray(fragment, dtype=float)
    )
