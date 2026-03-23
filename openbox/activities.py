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


from .types import rfc3339_now as _rfc3339_now  # shared utility

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .types import Verdict
from .hook_governance import build_auth_headers

logger = logging.getLogger(__name__)

# Module-level Temporal client reference, set by worker.py during initialization.
# Used by send_governance_event to call client.terminate() for HALT verdicts.
_temporal_client = None


def set_temporal_client(client) -> None:
    """Store Temporal client reference for HALT terminate calls."""
    global _temporal_client
    _temporal_client = client


# Re-export from errors.py for backward compatibility
from .errors import GovernanceAPIError  # noqa: F401


async def _terminate_workflow_for_halt(workflow_id: str, reason: str) -> None:
    """Force-terminate workflow via Temporal client for HALT verdict.

    HALT is the nuclear option — no cleanup, no finally blocks, immediate kill.
    Always raises ApplicationError after terminate to also stop the current activity.
    client.terminate() signals the server, but doesn't stop the running activity code.
    """
    if _temporal_client:
        try:
            logger.info(f"HALT: calling client.terminate() for workflow {workflow_id}")
            handle = _temporal_client.get_workflow_handle(workflow_id)
            await handle.terminate(f"Governance HALT: {reason}")
            logger.info(f"HALT: workflow {workflow_id} terminated successfully")
        except Exception as e:
            logger.warning(f"HALT: failed to terminate workflow {workflow_id}: {e}")
    else:
        logger.warning(f"HALT: _temporal_client is None, cannot terminate workflow {workflow_id}")

    # Always raise to stop the current activity execution.
    # Even after successful terminate(), the activity code keeps running
    # until an exception stops it.
    raise ApplicationError(
        f"Governance HALT: {reason}",
        type="GovernanceHalt",
        non_retryable=True,
    )


def raise_governance_block(reason: str, policy_id: str = None, risk_score: float = None):
    """Raise non-retryable ApplicationError for BLOCK verdict — blocks activity only."""
    details = {"policy_id": policy_id, "risk_score": risk_score}
    raise ApplicationError(
        f"Governance blocked: {reason}",
        details,
        type="GovernanceBlock",
        non_retryable=True,
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
                headers=build_auth_headers(api_key),
            )

            if response.status_code == 200:
                data = response.json()
                # Parse verdict (v1.1) or action (v1.0)
                verdict = Verdict.from_string(data.get("verdict") or data.get("action", "continue"))
                reason = data.get("reason")
                policy_id = data.get("policy_id")
                risk_score = data.get("risk_score", 0.0)

                # Check if governance wants to stop (BLOCK or HALT)
                if verdict.should_stop():
                    logger.info(f"Governance {verdict.value} {event_type}: {reason} (policy: {policy_id})")

                    # For SignalReceived events, return result instead of raising/terminating
                    # The workflow interceptor will store verdict for activity interceptor to check
                    if event_type == "SignalReceived":
                        return {
                            "success": True,
                            "verdict": verdict.value,
                            "action": verdict.value,  # backward compat
                            "reason": reason,
                            "policy_id": policy_id,
                            "risk_score": risk_score,
                        }

                    # HALT → terminate workflow + raise to stop activity
                    if verdict == Verdict.HALT:
                        workflow_id = event_payload.get("workflow_id", "")
                        # Always raises ApplicationError(type="GovernanceHalt")
                        await _terminate_workflow_for_halt(
                            workflow_id, reason or "No reason provided"
                        )
                    else:
                        # BLOCK → fail this activity only, workflow can continue
                        raise_governance_block(
                            reason=reason or "No reason provided",
                            policy_id=policy_id,
                            risk_score=risk_score,
                        )

                return {
                    "success": True,
                    "verdict": verdict.value,
                    "action": verdict.value,  # backward compat
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


