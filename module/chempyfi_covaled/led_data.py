"""Normalized LED representation (no OPI, no LEDAW, no Excel)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

KJMOL_PER_HARTREE = 2625.5


@dataclass(frozen=True)
class LEDComponents:
    """Optional decomposed LED matrices (same fragment axis as ``total``)."""

    reference: Optional[np.ndarray] = None
    correlation: Optional[np.ndarray] = None
    electrostatics: Optional[np.ndarray] = None
    exchange: Optional[np.ndarray] = None
    dispersion_strong: Optional[np.ndarray] = None
    dispersion_weak: Optional[np.ndarray] = None


@dataclass(frozen=True)
class LEDData:
    """
    Fragment-resolved LED for one ORCA calculation (one geometry index).

    ``fragment_ids[i]`` labels row/column ``i`` (1-based ORCA IDs for supersystem jobs;
    local 1…N for subsystem jobs).
    """

    fragment_ids: Tuple[int, ...]
    total: np.ndarray
    energy_unit: str = "kJ/mol"
    components: LEDComponents = field(default_factory=LEDComponents)
    source_path: Optional[str] = None

    def __post_init__(self):
        n = len(self.fragment_ids)
        if self.total.shape != (n, n):
            raise ValueError(
                f"LED total shape {self.total.shape} != ({n}, {n}) for {len(self.fragment_ids)} fragments"
            )
        if self.energy_unit not in ("kJ/mol", "hartree"):
            raise ValueError(f"Unsupported energy_unit: {self.energy_unit!r}")

    @property
    def n_fragments(self) -> int:
        return len(self.fragment_ids)

    def to_pandas(self):
        """View for legacy ``compute_covaled_interaction_matrix`` (1-based labels)."""
        import pandas as pd

        labels = list(self.fragment_ids)
        return pd.DataFrame(self.total, index=labels, columns=labels)

    def upper_triangle_pairs(self) -> dict[str, float]:
        """Keys ``\"i_j\"`` with i <= j, finite values only."""
        out: dict[str, float] = {}
        ids = self.fragment_ids
        for i in range(len(ids)):
            for j in range(i, len(ids)):
                val = float(self.total[i, j])
                if np.isfinite(val):
                    out[f"{ids[i]}_{ids[j]}"] = val
        return out


@dataclass(frozen=True)
class LEDSystemData:
    """Supersystem + subsystems for CovaLED Step 7."""

    supersystem: LEDData
    ligand: LEDData
    receptor: LEDData
    ligand_fragment_ids: Tuple[int, ...]
    receptor_fragment_ids: Tuple[int, ...]

    def validate_partition(self) -> None:
        all_ids = set(self.ligand_fragment_ids) | set(self.receptor_fragment_ids)
        super_ids = set(self.supersystem.fragment_ids)
        if not all_ids <= super_ids:
            missing = sorted(all_ids - super_ids)
            raise ValueError(f"Partition fragment IDs not in supersystem: {missing}")


def hartree_to_kjmol(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) * KJMOL_PER_HARTREE


def as_led_matrix(raw, fragment_ids: Tuple[int, ...], *, energy_unit: str = "hartree") -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (len(fragment_ids), len(fragment_ids)):
        raise ValueError(f"Matrix shape {arr.shape} != ({len(fragment_ids)}, {len(fragment_ids)})")
    if energy_unit == "hartree":
        return hartree_to_kjmol(arr)
    if energy_unit == "kJ/mol":
        return arr
    raise ValueError(energy_unit)
