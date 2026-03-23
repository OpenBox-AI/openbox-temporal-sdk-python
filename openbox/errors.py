# openbox/errors.py
"""OpenBox Temporal SDK — Unified exception hierarchy.

All SDK errors inherit from OpenBoxError. Mirrors the LangGraph SDK error
hierarchy for consistency across the OpenBox SDK family.

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

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .types import Verdict


# ═══════════════════════════════════════════════════════════════════
# Base
# ═══════════════════════════════════════════════════════════════════

class OpenBoxError(Exception):
    """Base class for all OpenBox SDK errors."""


# ═══════════════════════════════════════════════════════════════════
# Configuration errors (backward-compat: OpenBoxConfigError stays)
# ═══════════════════════════════════════════════════════════════════

class OpenBoxConfigError(OpenBoxError):
    """Raised when OpenBox configuration fails."""


class OpenBoxAuthError(OpenBoxConfigError):
    """Raised when API key validation fails."""


class OpenBoxNetworkError(OpenBoxConfigError):
    """Raised when network connectivity fails."""


class OpenBoxInsecureURLError(OpenBoxConfigError):
    """Raised when HTTP is used for non-localhost URLs."""


# ═══════════════════════════════════════════════════════════════════
# Governance verdict errors
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Guardrails errors
# ═══════════════════════════════════════════════════════════════════

class GuardrailsValidationError(OpenBoxError):
    """Raised when guardrails validation_passed is False.

    Attributes:
        reasons: List of reason strings from the guardrails evaluation.
    """

    def __init__(self, reasons: list[str] | None = None):
        self.reasons = reasons or []
        reason_str = "; ".join(self.reasons) if self.reasons else "Guardrails validation failed"
        super().__init__(reason_str)


# ═══════════════════════════════════════════════════════════════════
# HITL approval errors
# ═══════════════════════════════════════════════════════════════════

class ApprovalExpiredError(OpenBoxError):
    """Raised when HITL approval window expires (server-side deadline)."""


class ApprovalRejectedError(OpenBoxError):
    """Raised when HITL approval is explicitly rejected by a human."""


class ApprovalTimeoutError(OpenBoxError):
    """Raised when HITL polling exceeds the configured max wait time."""

    def __init__(self, max_wait_ms: int | None = None):
        self.max_wait_ms = max_wait_ms
        msg = f"Approval polling timed out after {max_wait_ms}ms" if max_wait_ms else "Approval polling timed out"
        super().__init__(msg)


# ═══════════════════════════════════════════════════════════════════
# Utility: exception chain walker
# ═══════════════════════════════════════════════════════════════════

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
        next_exc = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        # Also check Temporal's .cause property (ActivityError.cause → ApplicationError)
        if next_exc is None:
            next_exc = getattr(current, "cause", None)
        current = next_exc
    return None
