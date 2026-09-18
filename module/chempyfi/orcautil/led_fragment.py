"""Compatibility re-exports for local-model construction and CovaLED input (delegates to ``chempyfi_covaled.opi``)."""

import logging

from ..fragmentation.capping import create_molecule_cutout
from ..fragmentation.peptide import split_fragments_by_peptide_bonds
from ..modeling.local_environment import extract_local_environment, fragment_molecule

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def write_fragment_input(*args, **kwargs):
    from chempyfi_covaled.opi.write import write_fragment_input as _fn

    return _fn(*args, **kwargs)


def write_orca_input(*args, **kwargs):
    from chempyfi_covaled.opi.write import write_orca_input as _fn

    return _fn(*args, **kwargs)
