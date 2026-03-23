# openbox/verdict_handler.py
"""OpenBox Temporal SDK — Centralized verdict enforcement.

NOT sandbox-safe — uses logging at module level. Do NOT import from
workflow_interceptor.py or other workflow-context code.

Single enforce_verdict() entry point for activity interceptor verdict handling.
Hook governance keeps its own _handle_verdict() (different contract — GovernanceBlockedError
+ abort flags, not ApplicationError). Workflow interceptor verdict handling runs in sandbox
(limited imports) so stays inlined.
"""
from __future__ import annotations

import logging
from typing import Literal

from .errors import GovernanceBlockedError, GovernanceHaltError, GuardrailsValidationError
from .types import GovernanceVerdictResponse, Verdict

logger = logging.getLogger(__name__)

# Context where verdict is being enforced — for logging / future branching
VerdictContext = Literal["activity_start", "activity_end", "workflow_event"]


class VerdictEnforcementResult:
    """Result of enforce_verdict — tells caller what to do next.

    When requires_hitl is True the caller must set buffer.pending_approval and raise
    a retryable ApplicationError. When both flags are False the caller may proceed.
    """

    __slots__ = ("requires_hitl", "blocked")

    def __init__(self, *, requires_hitl: bool = False, blocked: bool = False):
        self.requires_hitl = requires_hitl
        self.blocked = blocked


def enforce_verdict(
    response: GovernanceVerdictResponse,
    context: VerdictContext,
) -> VerdictEnforcementResult:
    """Enforce governance verdict. Raises on HALT / BLOCK / guardrails failure.

    Verdict priority order: HALT > BLOCK > guardrails > REQUIRE_APPROVAL > CONSTRAIN > ALLOW

    Returns VerdictEnforcementResult with requires_hitl=True for REQUIRE_APPROVAL.
    Caller is responsible for the HITL flow and translating typed exceptions to
    Temporal ApplicationError.

    Args:
        response: Parsed governance verdict from OpenBox Core.
        context: Logical context for logging/diagnostics.

    Raises:
        GovernanceHaltError: Verdict is HALT — workflow must be terminated.
        GovernanceBlockedError: Verdict is BLOCK — activity must be failed non-retryably.
        GuardrailsValidationError: guardrails_result.validation_passed is False.
    """
    verdict = response.verdict
    reason = response.reason

    # 1. HALT — highest priority, triggers workflow termination
    if verdict == Verdict.HALT:
        raise GovernanceHaltError(reason or "Workflow halted by governance policy")

    # 2. BLOCK — non-retryable activity failure
    if verdict == Verdict.BLOCK:
        raise GovernanceBlockedError(verdict, reason or "Action blocked by governance policy")

    # 3. Guardrails validation failure — checked before REQUIRE_APPROVAL so a
    #    guardrails failure is never silently swallowed by an approval flow
    if response.guardrails_result and not response.guardrails_result.validation_passed:
        reasons = response.guardrails_result.get_reason_strings()
        if not reasons:
            gr_type = response.guardrails_result.input_type
            label = "output " if gr_type == "activity_output" else ""
            reasons = [f"Guardrails {label}validation failed"]
        raise GuardrailsValidationError(reasons)

    # 4. REQUIRE_APPROVAL — caller sets pending_approval flag and raises retryable error
    if verdict.requires_approval():
        return VerdictEnforcementResult(requires_hitl=True)

    # 5. CONSTRAIN — allow execution but log the constraint for observability
    if verdict == Verdict.CONSTRAIN and reason:
        policy_id = response.policy_id
        suffix = f" (policy: {policy_id})" if policy_id else ""
        logger.info(f"Governance constraint applied [{context}]: {reason}{suffix}")

    # 6. ALLOW (or CONSTRAIN without reason) — no action required
    return VerdictEnforcementResult()
