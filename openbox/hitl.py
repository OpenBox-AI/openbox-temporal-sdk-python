# openbox/hitl.py
"""OpenBox Temporal SDK — Human-in-the-Loop (HITL) approval handling.

NOT sandbox-safe — uses temporalio.exceptions. Do NOT import from
workflow_interceptor.py or other workflow-context code.

Extracted from activity_interceptor.py. Uses flat GovernanceConfig fields
(hitl_enabled, skip_hitl_activity_types) — no nested HITLConfig.

Temporal HITL uses the retry-based pattern:
  1. First attempt: set buffer.pending_approval=True, raise retryable ApprovalPending
  2. On retry: poll approval status, handle terminal verdict (expired/approved/rejected)
     or raise retryable ApprovalPending again to keep waiting.

handle_approval_response() takes the raw dict from GovernanceClient.poll_approval()
and raises the same ApplicationError types as the original inline code.
"""
from __future__ import annotations

import logging
from typing import NoReturn, Optional

logger = logging.getLogger(__name__)


def should_skip_hitl(
    activity_type: str,
    *,
    hitl_enabled: bool,
    skip_types: set,
) -> bool:
    """Return True if HITL should not apply to this activity.

    Args:
        activity_type: The Temporal activity type name.
        hitl_enabled: Whether HITL is globally enabled in GovernanceConfig.
        skip_types: Activity types exempt from HITL (e.g. to avoid loops).

    Returns:
        True when HITL should be skipped (not enabled or explicitly excluded).
    """
    return not hitl_enabled or activity_type in skip_types


def handle_approval_response(
    response: Optional[dict],
    activity_type: str,
    workflow_id: str,
    run_id: str,
    activity_id: str,
) -> bool:
    """Process an approval poll response dict from GovernanceClient.poll_approval().

    Raises the same ApplicationError types as the original inline activity_interceptor
    code so that existing Temporal retry policies and test assertions work unchanged.

    Args:
        response: Dict from poll_approval() or None if the API call failed.
        activity_type: Activity type name (for error messages).
        workflow_id: Workflow ID (for error messages).
        run_id: Workflow run ID (for error messages).
        activity_id: Activity ID (for error messages).

    Returns:
        True if the approval was granted (Verdict.ALLOW).

    Raises:
        ApplicationError(type="ApprovalPending", non_retryable=False):
            Response is None (poll failed) or verdict is still pending.
        ApplicationError(type="ApprovalExpired", non_retryable=True):
            Approval window has expired.
        ApplicationError(type="ApprovalRejected", non_retryable=True):
            A human explicitly rejected the request (BLOCK/HALT verdict).
    """
    if response is None:
        # API call failed — retry activity to poll again
        raise_approval_pending("Failed to check approval status, retrying...")

    # Check expiration before verdict (expired flag takes precedence)
    if response.get("expired"):
        from temporalio.exceptions import ApplicationError
        raise ApplicationError(
            f"Approval expired for activity {activity_type} "
            f"(workflow_id={workflow_id}, run_id={run_id}, activity_id={activity_id})",
            type="ApprovalExpired",
            non_retryable=True,
        )

    from .types import Verdict
    verdict = Verdict.from_string(response.get("verdict") or response.get("action"))

    if verdict == Verdict.ALLOW:
        # Approval granted — caller clears buffer.pending_approval and proceeds
        return True

    if verdict.should_stop():
        # Human explicitly rejected (BLOCK or HALT)
        reason = response.get("reason", "Activity rejected")
        from temporalio.exceptions import ApplicationError
        raise ApplicationError(
            f"Activity rejected: {reason}",
            type="ApprovalRejected",
            non_retryable=True,
        )

    # Still pending (REQUIRE_APPROVAL / CONSTRAIN) — keep retrying
    raise_approval_pending(f"Awaiting approval for activity {activity_type}")


def raise_approval_pending(reason: str) -> NoReturn:
    """Raise a retryable Temporal ApplicationError for the HITL retry loop.

    Temporal will retry the activity according to its retry policy, and the
    interceptor will poll again on the next attempt.

    Args:
        reason: Human-readable reason shown in Temporal UI / logs.
    """
    from temporalio.exceptions import ApplicationError
    raise ApplicationError(reason, type="ApprovalPending", non_retryable=False)
