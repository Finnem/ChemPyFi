"""Compatibility shim — use ``chempyfi_covaled.opi`` instead."""

from chempyfi_covaled.opi.adapter import (  # noqa: F401
    configure_led_calculation,
    extract_fragment_mapping_from_output,
    intent_from_model,
    intent_from_opi_structure,
    is_opi_available,
    model_from_fragment_write_args,
    parse_fragment_output,
    require_opi,
    to_opi_structure,
    write_orca_input_opi,
)
from chempyfi_covaled.opi.errors import OpiDependencyError as OpiNotAvailableError  # noqa: F401
