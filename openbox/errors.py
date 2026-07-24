"""OpenBox Temporal SDK — Unified exception hierarchy.

All SDK errors inherit from OpenBoxError.

Hierarchy:
    OpenBoxError (base)
    ├── OpenBoxConfigError          # backward-compat bridge
    │   ├── OpenBoxAuthError
    │   ├── OpenBoxNetworkError
    │   └── OpenBoxInsecureURLError
    ├── GovernanceBlockedError      # hook/activity verdict BLOCK
    ├── GovernanceHaltError         # verdict HALT (workflow termination)
    ├── GovernanceAPIError          # governance API failure (fail_closed)
    ├── GuardrailsValidationError   # guardrails validation_passed=False
    ├── ApprovalExpiredError        # HITL approval window expired
    ├── ApprovalRejectedError       # HITL approval explicitly rejected
    └── ApprovalTimeoutError        # HITL polling exceeded max wait
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Union

if TYPE_CHECKING:
    from .types import Verdict


# ApplicationError.type values raised by governance activities.
# Use these constants — never string-match on error messages or
# exception class names (brittle against locale / reformatting /
# Temporal's ActivityError wrapping).

GOVERNANCE_HALT_ERROR_TYPE: Final[str] = "GovernanceHalt"
GOVERNANCE_BLOCK_ERROR_TYPE: Final[str] = "GovernanceBlock"
GOVERNANCE_API_ERROR_TYPE: Final[str] = "GovernanceAPIError"
# Legacy alias; kept for histories predating the rename.
GOVERNANCE_STOP_ERROR_TYPE: Final[str] = "GovernanceStop"

# BLOCK-with-patch restart transport. A governance BLOCK carrying a valid patch
# crosses the activity/workflow boundary as a non-retryable ApplicationError of
# this stable type; the workflow interceptor catches it and Continue-As-News.
GOVERNANCE_PATCH_ERROR_TYPE: Final[str] = "GovernancePatch"
# Legacy alias — the pre-rename emit value. Already-recorded Temporal histories
# (open restart chains started before this rename shipped) may still carry this
# type on a replayed/pending ActivityError, so every extractor must keep
# accepting BOTH this and GOVERNANCE_PATCH_ERROR_TYPE above (mirrors the
# GovernanceHalt/GovernanceStop precedent). New executions only ever emit
# GOVERNANCE_PATCH_ERROR_TYPE; this alias is never raised going forward.
GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE: Final[str] = "GovernanceRetryableBlock"
# Raised when the bounded restart chain would exceed max_patch_restarts.
GOVERNANCE_PATCH_LIMIT_EXCEEDED_ERROR_TYPE: Final[str] = "GovernancePatchLimitExceeded"
# Raised when replacement input cannot be converted for Continue-As-New.
GOVERNANCE_PATCH_INPUT_INVALID_ERROR_TYPE: Final[str] = "GovernancePatchInputInvalid"


class OpenBoxError(Exception):
    """Base class for all OpenBox SDK errors."""


class OpenBoxConfigError(OpenBoxError):
    """Raised when OpenBox configuration fails."""


class OpenBoxAuthError(OpenBoxConfigError):
    """Raised when API key validation fails."""


class OpenBoxNetworkError(OpenBoxConfigError):
    """Raised when network connectivity fails."""


class OpenBoxInsecureURLError(OpenBoxConfigError):
    """Raised when HTTP is used for non-localhost URLs."""


class OpenBoxSigningError(OpenBoxAuthError):
    """Raised when Core rejects a signed (AIP DID) request.

    Attributes:
        reason_code: Core's machine reason code (e.g. ``signature_invalid``).
    """

    def __init__(self, message: str, reason_code: str | None = None):
        self.reason_code = reason_code
        super().__init__(message)


# Core signed-request rejection reason codes → actionable SDK guidance.
# NOTE: forward-compatible. Core today often collapses identity failures into a
# generic "invalid token" body with no machine code; when that happens
# map_signing_error() is not reached and callers get a generic OpenBoxAuthError.
# These richer messages activate once Core emits a machine reason code in the
# body ("reason_code"/"code"/"reason"). Keys mirror Core's verification handler.
_SIGNING_REASON_MESSAGES: dict[str, str] = {
    "signature_invalid": (
        "Request signature rejected (signature_invalid). The signed bytes did not "
        "match — usually a body-hash mismatch (send content= bytes, never json=) or "
        "a wrong/rotated private key."
    ),
    "nonce_replayed": (
        "Request nonce was already used (nonce_replayed). Each request must carry a "
        "fresh nonce; do not retry a fully-prepared request verbatim."
    ),
    "did_agent_mismatch": (
        "DID does not match the authenticated agent (did_agent_mismatch). Check that "
        "agent_did matches the agent the API key/private key were provisioned for."
    ),
    "verifier_not_configured": (
        "Core has no verifier for this agent (verifier_not_configured). The agent's "
        "public key may not be imported to KMS yet — re-provision the agent."
    ),
    # Core's code is "timestamp_outside_window"; "timestamp_skew" kept as an alias.
    "timestamp_outside_window": (
        "Request timestamp outside the allowed window (timestamp_outside_window). Sync "
        "the host clock (NTP); signatures are valid only within ±300s."
    ),
    "timestamp_skew": (
        "Request timestamp outside the allowed window (timestamp_skew). Sync the host "
        "clock (NTP); signatures are valid only within ±300s."
    ),
}


def map_signing_error(
    reason_code: str | None, fallback: str = ""
) -> OpenBoxSigningError:
    """Map a Core signing reason code to an actionable OpenBoxSigningError.

    Unknown/empty codes fall back to a generic message (optionally augmented with
    ``fallback`` context). Never raises — always returns an exception to raise.
    """
    if reason_code and reason_code in _SIGNING_REASON_MESSAGES:
        return OpenBoxSigningError(_SIGNING_REASON_MESSAGES[reason_code], reason_code)
    msg = fallback or (
        f"Signed request rejected by OpenBox Core"
        + (f" ({reason_code})" if reason_code else "")
        + "."
    )
    return OpenBoxSigningError(msg, reason_code)


class GovernanceBlockedError(OpenBoxError):
    """Raised by OTel hooks when governance blocks an operation.

    Attributes:
        verdict: The Verdict enum value (normalized from string if needed).
        reason: Human-readable explanation from the policy engine.
        url: The URL or resource identifier that was blocked (optional).
    """

    def __init__(self, verdict: Union[str, "Verdict"], reason: str, url: str = ""):
        # Lazy import to avoid circular dependency with types.py
        if isinstance(verdict, str):
            from .types import Verdict

            self.verdict = Verdict.from_string(verdict)
        else:
            self.verdict = verdict
        self.reason = reason
        self.url = url
        super().__init__(f"Governance {self.verdict.value}: {reason}")


class GovernanceHaltError(OpenBoxError):
    """Raised when governance halts workflow execution (HALT verdict).

    HALT is the nuclear option — triggers workflow termination.
    """

    def __init__(self, message: str):
        super().__init__(message)


class GovernanceAPIError(OpenBoxError):
    """Raised when governance API fails and policy is fail_closed."""


class GuardrailsValidationError(OpenBoxError):
    """Raised when guardrails validation_passed is False.

    Attributes:
        reasons: List of reason strings from the guardrails evaluation.
    """

    def __init__(self, reasons: list[str] | None = None):
        self.reasons = reasons or []
        reason_str = (
            "; ".join(self.reasons) if self.reasons else "Guardrails validation failed"
        )
        super().__init__(reason_str)


class ApprovalExpiredError(OpenBoxError):
    """Raised when HITL approval window expires (server-side deadline)."""


class ApprovalRejectedError(OpenBoxError):
    """Raised when HITL approval is explicitly rejected by a human."""


class ApprovalTimeoutError(OpenBoxError):
    """Raised when HITL polling exceeds the configured max wait time."""

    def __init__(self, max_wait_ms: int | None = None):
        self.max_wait_ms = max_wait_ms
        msg = (
            f"Approval polling timed out after {max_wait_ms}ms"
            if max_wait_ms
            else "Approval polling timed out"
        )
        super().__init__(msg)


def extract_governance_error(exc: BaseException) -> GovernanceBlockedError | None:
    """Walk exception chain to find a wrapped GovernanceBlockedError.

    Temporal wraps activity exceptions: ActivityError → ApplicationError → original.
    External SDKs (OpenAI, Anthropic) wrap httpx errors similarly. This utility
    recovers the original GovernanceBlockedError for verdict inspection.

    Args:
        exc: Any exception, potentially wrapping a GovernanceBlockedError.

    Returns:
        The GovernanceBlockedError if found in the chain, None otherwise.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GovernanceBlockedError):
            return current
        # Walk both explicit (__cause__) and implicit (__context__) chains
        next_exc = getattr(current, "__cause__", None) or getattr(
            current, "__context__", None
        )
        # Also check Temporal's .cause property (ActivityError.cause → ApplicationError)
        if next_exc is None:
            next_exc = getattr(current, "cause", None)
        current = next_exc
    return None
