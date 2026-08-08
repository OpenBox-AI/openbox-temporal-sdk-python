"""Temporal error compatibility exports.

All framework-neutral OpenBox errors live in :mod:`openbox_core.errors`.  These
names are direct aliases (not subclasses) so callers can catch an error through
either SDK and retain exact class/function identity.  Only Temporal
``ApplicationError.type`` constants remain local.
"""

from typing import Final

from openbox_core.errors import (
    ApprovalExpiredError,
    ApprovalRejectedError,
    ApprovalTimeoutError,
    ContractError,
    GovernanceAPIError,
    GovernanceBlockedError,
    GovernanceHaltError,
    GuardrailsValidationError,
    OpenBoxAuthError,
    OpenBoxConfigError,
    OpenBoxError,
    OpenBoxInsecureURLError,
    OpenBoxNetworkError,
    OpenBoxSigningError,
    extract_governance_error,
    map_signing_error,
)

# ApplicationError.type values raised by Temporal governance activities.
GOVERNANCE_HALT_ERROR_TYPE: Final[str] = "GovernanceHalt"
GOVERNANCE_BLOCK_ERROR_TYPE: Final[str] = "GovernanceBlock"
GOVERNANCE_API_ERROR_TYPE: Final[str] = "GovernanceAPIError"
# Legacy alias retained for histories predating the rename.
GOVERNANCE_STOP_ERROR_TYPE: Final[str] = "GovernanceStop"
GOVERNANCE_PATCH_ERROR_TYPE: Final[str] = "GovernancePatch"
GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE: Final[str] = "GovernancePatchLimitExceeded"
GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE: Final[str] = "GovernanceRetryableBlock"

__all__ = [
    "GOVERNANCE_HALT_ERROR_TYPE",
    "GOVERNANCE_BLOCK_ERROR_TYPE",
    "GOVERNANCE_API_ERROR_TYPE",
    "GOVERNANCE_STOP_ERROR_TYPE",
    "GOVERNANCE_PATCH_ERROR_TYPE",
    "GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE",
    "GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE",
    "OpenBoxError",
    "ContractError",
    "OpenBoxConfigError",
    "OpenBoxAuthError",
    "OpenBoxNetworkError",
    "OpenBoxInsecureURLError",
    "OpenBoxSigningError",
    "GovernanceBlockedError",
    "GovernanceHaltError",
    "GovernanceAPIError",
    "GuardrailsValidationError",
    "ApprovalExpiredError",
    "ApprovalRejectedError",
    "ApprovalTimeoutError",
    "extract_governance_error",
    "map_signing_error",
]
