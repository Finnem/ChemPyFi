"""Optional dependency errors for CovaLED ORCA workflows."""


class OpiDependencyError(ImportError):
    """Raised when orca-pi is required but not installed."""

    def __init__(self, detail: str = ""):
        msg = (
            "ORCA input generation and CovaLED analysis require the OPI optional dependency. "
            "Install CovaLED with: pip install 'chempyfi-covaled[opi]' (orca-pi; Python >=3.11). "
            "For execution you also need a licensed ORCA installation with JSON property output enabled."
        )
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)
