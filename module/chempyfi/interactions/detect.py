"""Headless interaction detection entry point."""

from .Detection import Interactions


def detect_interactions(receptor, ligand):
    """Detect receptor–ligand interactions; returns the same dict as ``Interactions.detect_interactions``."""
    return Interactions(receptor).detect_interactions(ligand)
