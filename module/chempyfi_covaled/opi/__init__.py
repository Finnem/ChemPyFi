"""ORCA I/O for CovaLED (orca-pi boundary)."""

from .adapter import (
    configure_led_calculation,
    extract_fragment_mapping_from_output,
    intent_from_model,
    is_opi_available,
    model_from_fragment_write_args,
    parse_fragment_output,
    require_opi,
    to_opi_structure,
    write_orca_input_opi,
)
from .errors import OpiDependencyError
from .intent import (
    OrcaAtomSpec,
    OrcaCalculationIntent,
    build_intent_from_inp_file,
    build_intent_from_legacy_args,
    build_intent_from_model,
    intents_semantically_equal,
)
from .write import write_fragment_input, write_orca_input

__all__ = [
    "OpiDependencyError",
    "OrcaAtomSpec",
    "OrcaCalculationIntent",
    "build_intent_from_inp_file",
    "build_intent_from_legacy_args",
    "build_intent_from_model",
    "configure_led_calculation",
    "extract_fragment_mapping_from_output",
    "intent_from_model",
    "intents_semantically_equal",
    "is_opi_available",
    "model_from_fragment_write_args",
    "parse_fragment_output",
    "require_opi",
    "to_opi_structure",
    "write_fragment_input",
    "write_orca_input",
    "write_orca_input_opi",
]
