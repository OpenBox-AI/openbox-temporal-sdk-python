# openbox/activities.py
#
# IMPORTANT: This module imports httpx which uses os.stat internally.
# Do NOT import this module from workflow code (workflow_interceptor.py)!
# The workflow interceptor references this activity by string name "send_governance_event".
"""
Governance event activity for workflow-level HTTP calls.

CRITICAL: Temporal workflows must be deterministic. HTTP calls are NOT allowed directly
in workflow code (including interceptors). WorkflowInboundInterceptor sends events via
workflow.execute_activity() using this activity.

Events sent via this activity:
- WorkflowStarted
- WorkflowCompleted
- SignalReceived

Note: ActivityStarted/Completed events are sent directly from ActivityInboundInterceptor
since activities are allowed to make HTTP calls.

TIMESTAMP HANDLING: This activity adds the "timestamp" field to the payload when it
executes. This ensures timestamps are generated in activity context (non-deterministic
code allowed) rather than workflow context (must be deterministic).
"""

import httpx
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional


def _rfc3339_now() -> str:
    """Return current UTC time in RFC3339 format."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

from temporalio import activity
from temporalio.exceptions import ApplicationError

logger = logging.getLogger(__name__)


class GovernanceAPIError(Exception):
    """Raised when governance API fails and policy is fail_closed."""
    pass


def raise_governance_stop(reason: str, policy_id: str = None, risk_score: float = None):
    """
    Raise a non-retryable ApplicationError when governance blocks an operation.

    Using ApplicationError with non_retryable=True ensures:
    1. The activity fails immediately (no retries)
    2. The workflow fails with a clear error message
    """
    details = {"policy_id": policy_id, "risk_score": risk_score}
    raise ApplicationError(
        f"Governance blocked: {reason}",
        details,
        type="GovernanceStop",
        non_retryable=True,  # Don't retry - terminate the workflow
    )


@activity.defn(name="send_governance_event")
async def send_governance_event(input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Activity that sends governance events to OpenBox Core.

    This activity is called from WorkflowInboundInterceptor via workflow.execute_activity()
    to maintain workflow determinism. HTTP calls cannot be made directly in workflow context.

    Args (in input dict):
        api_url: OpenBox Core API URL
        api_key: API key for authentication
        payload: Event payload (without timestamp)
        timeout: Request timeout in seconds
        on_api_error: "fail_open" (default) or "fail_closed"

    When on_api_error == "fail_closed" and API fails, raises GovernanceAPIError.
    This is caught by the workflow interceptor and re-raised as GovernanceHaltError.

    Logging is safe here because activities run outside the workflow sandbox.
    """
    # Extract input fields
    api_url = input.get("api_url", "")
    api_key = input.get("api_key", "")
    event_payload = input.get("payload", {})
    timeout = input.get("timeout", 30.0)
    on_api_error = input.get("on_api_error", "fail_open")

    # Add timestamp here in activity context (non-deterministic code allowed)
    # Use RFC3339 format: 2024-01-15T10:30:45.123Z
    payload = {**event_payload, "timestamp": _rfc3339_now()}
    event_type = event_payload.get("event_type", "unknown")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{api_url}/api/v1/governance/evaluate",
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code == 200:
                data = response.json()
                action = data.get("action", "continue")
                reason = data.get("reason")
                policy_id = data.get("policy_id")
                risk_score = data.get("risk_score", 0.0)

                # Check if governance wants to stop the workflow
                if action == "stop":
                    logger.info(f"Governance blocked {event_type}: {reason} (policy: {policy_id})")

                    # For SignalReceived events, return result instead of raising
                    # The workflow interceptor will store verdict for activity interceptor to check
                    if event_type == "SignalReceived":
                        return {
                            "success": True,
                            "action": action,
                            "reason": reason,
                            "policy_id": policy_id,
                            "risk_score": risk_score,
                        }

                    # For other events, raise non-retryable error to terminate workflow immediately
                    raise_governance_stop(
                        reason=reason or "No reason provided",
                        policy_id=policy_id,
                        risk_score=risk_score,
                    )

                return {
                    "success": True,
                    "action": action,
                    "reason": reason,
                    "policy_id": policy_id,
                    "risk_score": risk_score,
                }
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.warning(f"Governance API error for {event_type}: {error_msg}")
                if on_api_error == "fail_closed":
                    raise GovernanceAPIError(error_msg)
                return {"success": False, "error": error_msg}

    except (GovernanceAPIError, ApplicationError):
        raise  # Re-raise to workflow (ApplicationError is non-retryable)
    except Exception as e:
        logger.warning(f"Failed to send {event_type} event: {e}")
        if on_api_error == "fail_closed":
            raise GovernanceAPIError(str(e))
        return {"success": False, "error": str(e)}


