
from . import AnionReceptor, AnionLigand, \
        CationReceptor, CationLigand, \
        HydrogenBondAcceptorReceptor, HydrogenBondDonorLigand, \
        HydrogenBondDonorReceptor, HydrogenBondAcceptorLigand, \
        AromaticProximityLigand, AromaticProximityReceptor, \
        ApolarSurfaceReceptor
class Interactions():

    def __init__(self, receptor):
        self.receptor = receptor
        self.cation_receptor = CationReceptor(receptor)
        self.anion_receptor = AnionReceptor(receptor)
        self.hydrogen_bond_acceptor_receptor = HydrogenBondAcceptorReceptor(receptor)
        self.bad_hydrogen_bond_acceptor_receptor = HydrogenBondAcceptorReceptor(receptor, angle_threshold=60, distance_threshold=3.5)
        self.hydrogen_bond_donor_receptor = HydrogenBondDonorReceptor(receptor)
        self.bad_hydrogen_bond_donor_receptor = HydrogenBondDonorReceptor(receptor, angle_threshold=60, distance_threshold=3.5)
        self.pi_stacking_receptor = AromaticProximityReceptor(receptor)
        self.apolar_surface_receptor = ApolarSurfaceReceptor(receptor)

    def detect_interactions(self, ligand):
        cation_ligand = CationLigand(ligand)
        anion_ligand = AnionLigand(ligand)
        hydrogen_bond_acceptor_ligand = HydrogenBondAcceptorLigand(ligand)
        hydrogen_bond_donor_ligand = HydrogenBondDonorLigand(ligand)
        pi_stacking_ligand = AromaticProximityLigand(ligand)

        # hydrogen_bond_donor_interactions
        hydrogen_bond_donor_interactions = self.hydrogen_bond_donor_receptor.detect_interactions(hydrogen_bond_acceptor_ligand)
        bad_hydrogen_bond_donor_interactions = self.bad_hydrogen_bond_donor_receptor.detect_interactions(hydrogen_bond_acceptor_ligand)
        # remove good interactions from bad interactions
        good_indices = set([(*interaction[:2],) for interaction in hydrogen_bond_donor_interactions])
        bad_hydrogen_bond_donor_interactions = [interaction for interaction in bad_hydrogen_bond_donor_interactions if (*interaction[:2],) not in good_indices]
        # hydrogen_bond_acceptor_interactions
        hydrogen_bond_acceptor_interactions = self.hydrogen_bond_acceptor_receptor.detect_interactions(hydrogen_bond_donor_ligand)
        bad_hydrogen_bond_acceptor_interactions = self.bad_hydrogen_bond_acceptor_receptor.detect_interactions(hydrogen_bond_donor_ligand)
        # remove good interactions from bad interactions
        good_indices = set([(*interaction[:2],) for interaction in hydrogen_bond_acceptor_interactions])
        bad_hydrogen_bond_acceptor_interactions = [interaction for interaction in bad_hydrogen_bond_acceptor_interactions if (*interaction[:2],) not in good_indices]
        # cation-dipole interactions
        cation_dipole_interactions = self.cation_receptor.detect_interactions(hydrogen_bond_acceptor_ligand, distance_threshold=4)
        dipole_cation_interactions = self.hydrogen_bond_acceptor_receptor.detect_interactions(cation_ligand, distance_threshold=4)

        # anion-dipole interactions
        anion_dipole_interactions = self.anion_receptor.detect_interactions(hydrogen_bond_donor_ligand, distance_threshold=4)
        dipole_anion_interactions = self.hydrogen_bond_donor_receptor.detect_interactions(anion_ligand, distance_threshold=4)

        # anion-cation interactions
        anion_cation_interactions = self.anion_receptor.detect_interactions(cation_ligand)
        cation_anion_interactions = self.cation_receptor.detect_interactions(anion_ligand)

        # pi stacking interactions
        pi_stacking_interactions = self.pi_stacking_receptor.detect_interactions(pi_stacking_ligand)
        apolar_surface_interactions = self.apolar_surface_receptor.detect_interactions(ligand)

        return {
            "hydrogen_bond_donor_interactions": hydrogen_bond_donor_interactions,
            "hydrogen_bond_acceptor_interactions": hydrogen_bond_acceptor_interactions,
            "bad_hydrogen_bond_donor_interactions": bad_hydrogen_bond_donor_interactions,
            "bad_hydrogen_bond_acceptor_interactions": bad_hydrogen_bond_acceptor_interactions,
            "cation_dipole_interactions": cation_dipole_interactions,
            "dipole_cation_interactions": dipole_cation_interactions,
            "anion_dipole_interactions": anion_dipole_interactions,
            "dipole_anion_interactions": dipole_anion_interactions,
            "anion_cation_interactions": anion_cation_interactions,
            "cation_anion_interactions": cation_anion_interactions,
            "pi_stacking_interactions": pi_stacking_interactions,
            "apolar_surface_interactions": apolar_surface_interactions,
        }

    def _interaction_dicts(self, ligand, with_self=False):
        interactions = self.detect_interactions(ligand)
        additional = None
        if with_self:
            additional = Interactions(ligand).detect_interactions(ligand)
        return interactions, additional

    def write_interaction_display(self, ligand, filename, prefix=None, with_self=False):
        interactions, additional = self._interaction_dicts(ligand, with_self=with_self)
        from chempyfi_pymol.display import write_interaction_display as _write

        _write(interactions, filename, prefix=prefix, additional_interactions=additional)

    def create_interaction_display(
        self,
        ligand,
        prefix=None,
        with_self=False,
        interaction_points=False,
        return_interactions=False,
    ):
        interactions, additional = self._interaction_dicts(ligand, with_self=with_self)
        from chempyfi_pymol.display import create_interaction_display as _create

        if prefix is None:
            prefix = "interactions"
        visual = _create(
            interactions,
            prefix=prefix,
            additional_interactions=additional,
            interaction_points=interaction_points,
        )
        if return_interactions:
            return visual, interactions
        return visual
