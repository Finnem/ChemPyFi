"""CovaLED result containers (framework-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class LEDMatrix:
    """Fragment×fragment LED values (1-based index labels as in ORCA)."""

    values: pd.DataFrame

    def total_interaction_energy(self) -> float:
        from .analysis import extract_interaction_energy

        return extract_interaction_energy(self.values)


@dataclass(frozen=True)
class CovaLEDResult:
    """Output of CovaLED Step 7."""

    interaction_matrix: LEDMatrix
    ligand_fragment_ids: Tuple[int, ...]
    receptor_fragment_ids: Tuple[int, ...]


@dataclass(frozen=True)
class FPCovaLEDResult:
    """Output of fp-CovaLED Step 8 redistribution."""

    pairwise: Dict[str, float]
    redistributed_intra_total: float
    source_matrix: Optional[LEDMatrix] = None
