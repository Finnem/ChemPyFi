import math

from rdkit import Chem
from rdkit.Geometry import Point3D

from chempyfi.interactions.ApolarSurface import ApolarSurfaceReceptor


def _build_molecule(atom_symbols, coordinates, residue_specs=None):
    rw = Chem.RWMol()
    for symbol in atom_symbols:
        rw.AddAtom(Chem.Atom(symbol))
    mol = rw.GetMol()

    conf = Chem.Conformer(len(atom_symbols))
    for idx, xyz in enumerate(coordinates):
        conf.SetAtomPosition(idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol.AddConformer(conf, assignId=True)

    if residue_specs is not None:
        for atom_idx, (resn, resi, chain) in residue_specs.items():
            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName(resn)
            info.SetResidueNumber(int(resi))
            info.SetChainId(chain)
            mol.GetAtomWithIdx(atom_idx).SetMonomerInfo(info)
    return mol


def _detector(receptor):
    return ApolarSurfaceReceptor(
        receptor,
        probe_radius=1.4,
        n_sphere_points=240,
        pocket_cutoff=6.0,
        epsilon=1e-8,
        zero_denominator="zero",
    )


def test_isolated_ligand_has_no_burial():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    empty_receptor = Chem.RWMol().GetMol()

    result = _detector(empty_receptor).detect_interactions(ligand)
    assert result["lig_apolar_sasa_free"] > 0.0
    assert result["lig_apolar_buried_by_apolar_protein"] == 0.0
    assert result["lig_apolar_buried_total"] == 0.0
    assert result["lig_apolar_sasa_bound"] == result["lig_apolar_sasa_free"]


def test_far_protein_atom_behaves_like_isolated():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    receptor = _build_molecule(["C"], [(20.0, 0.0, 0.0)], residue_specs={0: ("LEU", 10, "A")})

    result = _detector(receptor).detect_interactions(ligand)
    assert result["lig_apolar_buried_by_apolar_protein"] == 0.0
    assert result["lig_apolar_buried_total"] == 0.0
    assert result["lig_apolar_sasa_bound"] == result["lig_apolar_sasa_free"]


def test_near_apolar_blocker_increases_apolar_burial():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    receptor = _build_molecule(["C"], [(3.0, 0.0, 0.0)], residue_specs={0: ("LEU", 10, "A")})

    result = _detector(receptor).detect_interactions(ligand)
    assert result["lig_apolar_buried_total"] > 0.0
    assert result["lig_apolar_buried_by_apolar_protein"] > 0.0
    assert result["lig_apolar_buried_by_apolar_protein"] <= result["lig_apolar_buried_total"]


def test_near_polar_blocker_does_not_count_as_apolar_burial():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    receptor = _build_molecule(["O"], [(3.0, 0.0, 0.0)], residue_specs={0: ("SER", 20, "A")})

    result = _detector(receptor).detect_interactions(ligand)
    assert result["lig_apolar_buried_total"] > 0.0
    assert result["lig_apolar_buried_by_apolar_protein"] == 0.0


def test_mixed_blockers_count_toward_apolar_burial():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    receptor = _build_molecule(
        ["C", "O"],
        [(3.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
        residue_specs={0: ("LEU", 10, "A"), 1: ("SER", 20, "A")},
    )

    result = _detector(receptor).detect_interactions(ligand)
    assert result["lig_apolar_buried_total"] > 0.0
    assert result["lig_apolar_buried_by_apolar_protein"] > 0.0


def test_reproducibility_is_deterministic():
    ligand = _build_molecule(["C"], [(0.0, 0.0, 0.0)])
    receptor = _build_molecule(["C"], [(3.0, 0.0, 0.0)], residue_specs={0: ("LEU", 10, "A")})

    detector = _detector(receptor)
    result_1 = detector.detect_interactions(ligand)
    result_2 = detector.detect_interactions(ligand)

    assert result_1["lig_apolar_sasa_free"] == result_2["lig_apolar_sasa_free"]
    assert result_1["lig_apolar_sasa_bound"] == result_2["lig_apolar_sasa_bound"]
    assert (
        result_1["lig_apolar_buried_by_apolar_protein"]
        == result_2["lig_apolar_buried_by_apolar_protein"]
    )
    assert (
        result_1["fraction_buried_by_apolar_protein"]
        == result_2["fraction_buried_by_apolar_protein"]
    )

    # Per-residue contributions should sum to global apolar burial.
    residue_sum_1 = sum(result_1["per_protein_residue"].values())
    assert math.isclose(
        residue_sum_1,
        result_1["lig_apolar_buried_by_apolar_protein"],
        rel_tol=1e-12,
        abs_tol=1e-12,
    )
