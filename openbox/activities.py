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
        logger.warning(
            f"HALT: _temporal_client is None, cannot terminate workflow {workflow_id}"
        )

    # Always raise to stop the current activity execution.
    # Even after successful terminate(), the activity code keeps running
    # until an exception stops it.
    raise ApplicationError(
        f"Governance HALT: {reason}",
        type="GovernanceHalt",
        non_retryable=True,
    )


def raise_governance_block(
    reason: str, policy_id: str = None, risk_score: float = None
):
    """Raise non-retryable ApplicationError for BLOCK verdict — blocks activity only."""
    details = {"policy_id": policy_id, "risk_score": risk_score}
    raise ApplicationError(
        f"Governance blocked: {reason}",
        details,
        type="GovernanceBlock",
        non_retryable=True,
    )


def _build_verdict_result(verdict: Verdict, reason, policy_id, risk_score) -> dict:
    """Build a success result dict from a governance verdict."""
    return {
        "success": True,
        "verdict": verdict.value,
        "action": verdict.value,  # backward compat
        "reason": reason,
        "policy_id": policy_id,
        "risk_score": risk_score,
    }


async def _handle_stop_verdict(
    verdict: Verdict, reason, policy_id, risk_score, event_type, event_payload
) -> Optional[dict]:
    """Handle BLOCK/HALT verdicts. Returns result for signals, raises for others."""
    logger.info(
        f"Governance {verdict.value} {event_type}: {reason} (policy: {policy_id})"
    )

    # SignalReceived: return result instead of raising
    if event_type == "SignalReceived":
        return _build_verdict_result(verdict, reason, policy_id, risk_score)

    # HALT: terminate workflow + raise
    if verdict == Verdict.HALT:
        workflow_id = event_payload.get("workflow_id", "")
        await _terminate_workflow_for_halt(workflow_id, reason or "No reason provided")

    # BLOCK: fail this activity only
    raise_governance_block(
        reason=reason or "No reason provided",
        policy_id=policy_id,
        risk_score=risk_score,
    )


def _handle_api_error(event_type: str, error_msg: str, on_api_error: str) -> dict:
    """Handle non-200 responses or exceptions based on error policy."""
    logger.warning(f"Governance API error for {event_type}: {error_msg}")
    if on_api_error == "fail_closed":
        raise GovernanceAPIError(error_msg)
    return {"success": False, "error": error_msg}


class GovernanceActivities:
    """Container for Temporal activities that need OpenBox credentials.

    Credentials live on the instance (never in activity inputs → never in
    workflow history → not visible to anyone with namespace read access).
    The worker registers bound methods of a single instance, created at
    worker-init time by the plugin / create_openbox_worker factory.
    """

    def __init__(self, api_url: str, api_key: str):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key

    @activity.defn(name="send_governance_event")
    async def send_governance_event(
        self, input: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send a governance event to OpenBox Core.

        Called from WorkflowInboundInterceptor via workflow.execute_activity()
        to maintain workflow determinism. The `input` dict carries only the
        event payload and per-call policy — never credentials.
        """
        event_payload = input.get("payload", {})
        timeout = input.get("timeout", 30.0)
        on_api_error = input.get("on_api_error", "fail_open")

        # Add timestamp in activity context (non-deterministic code allowed)
        payload = {**event_payload, "timestamp": _rfc3339_now()}
        event_type = event_payload.get("event_type", "unknown")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{self._api_url}/api/v1/governance/evaluate",
                    json=payload,
                    headers=build_auth_headers(self._api_key),
                )

                if response.status_code != 200:
                    return _handle_api_error(
                        event_type,
                        f"HTTP {response.status_code}: {response.text}",
                        on_api_error,
                    )

                data = response.json()
                verdict = Verdict.from_string(
                    data.get("verdict") or data.get("action", "continue")
                )
                reason = data.get("reason")
                policy_id = data.get("policy_id")
                risk_score = data.get("risk_score", 0.0)

                if verdict.should_stop():
                    result = await _handle_stop_verdict(
                        verdict,
                        reason,
                        policy_id,
                        risk_score,
                        event_type,
                        event_payload,
                    )
                    if result:
                        return result

                return _build_verdict_result(verdict, reason, policy_id, risk_score)

        except (GovernanceAPIError, ApplicationError):
            raise
        except Exception as e:
            logger.warning(f"Failed to send {event_type} event: {e}")
            if on_api_error == "fail_closed":
                raise GovernanceAPIError(str(e))
            return {"success": False, "error": str(e)}


def build_governance_activities(
    api_url: str, api_key: str
) -> GovernanceActivities:
    """Factory used by plugin.py and worker.py to build the activities instance."""
    return GovernanceActivities(api_url=api_url, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compat module-level helper.
#
# Not decorated with @activity.defn — worker/plugin register the class-based
# version above so credentials never flow through activity inputs. This shim
# exists for direct callers (tests, scripts) who already hold credentials and
# want to invoke the HTTP logic without constructing the class themselves.
# ─────────────────────────────────────────────────────────────────────────────
async def send_governance_event(input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Backward-compat wrapper — delegates to GovernanceActivities.

    Tests and direct callers can keep passing api_url/api_key in the input
    dict. The class-based activity registered with the worker does NOT see
    these fields (it reads credentials from self), so nothing written to
    workflow history ever carries the API key.
    """
    instance = GovernanceActivities(
        api_url=input.get("api_url", ""),
        api_key=input.get("api_key", ""),
    )
    forwarded = {k: v for k, v in input.items() if k not in ("api_url", "api_key")}
    return await instance.send_governance_event(forwarded)
