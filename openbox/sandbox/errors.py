"""Constant public errors for private governed-command deployment bootstrap."""


class GovernedCommandDeploymentError(ValueError):
    """Raised when deployment material cannot be accepted."""

    def __init__(self) -> None:
        super().__init__("governed-command deployment rejected")
