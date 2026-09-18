"""Covaled / OPI LED extraction errors."""


class MissingLEDDataError(ValueError):
    """Required LED quantity missing from OPI output (no silent NaN/None fallback)."""

    def __init__(self, quantity: str, *, source: str | None = None):
        self.quantity = quantity
        self.source = source
        if quantity == "orca.out.property.json":
            msg = (
                "CovaLED analysis requires ORCA output with a JSON property sidecar "
                f"({quantity} next to the .out file). Re-run ORCA with JSON property output enabled."
            )
        else:
            msg = f"Missing required LED data: {quantity}"
        if source:
            msg += f" Source: {source}."
        super().__init__(msg)
