"""
Governance event activity for workflow-level HTTP calls.

NOT sandbox-safe: this module imports httpx. Workflow code references the
activity by string name instead of importing it.

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

import logging
from typing import Any, Dict, NoReturn, Optional

from temporalio import activity
from temporalio.exceptions import ApplicationError

from .errors import GovernanceAPIError
from .types import Verdict
from .types import rfc3339_now as _rfc3339_now

logger = logging.getLogger(__name__)

# Module-level Temporal client reference, set by worker.py during initialization.
# Used by send_governance_event to call client.terminate() for HALT verdicts.
_temporal_client = None


def set_temporal_client(client) -> None:
    """Store Temporal client reference for HALT terminate calls."""
    global _temporal_client
    _temporal_client = client


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
    reason: str,
    policy_id: Optional[str] = None,
    risk_score: Optional[float] = None,
) -> NoReturn:
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

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        agent_did=None,
        signer=None,
        timeout: float = 30.0,
        on_api_error: str = "fail_open",
        governance_client=None,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._agent_did = agent_did
        self._signer = signer
        self._on_api_error = on_api_error
        self._owns_governance_client = governance_client is None
        if governance_client is None:
            from .client import GovernanceClient

            governance_client = GovernanceClient(
                api_url=self._api_url,
                api_key=self._api_key,
                timeout=timeout,
                on_api_error=on_api_error,
                agent_did=agent_did,
                signer=signer,
            )
        self._governance_client = governance_client

    @activity.defn(name="send_governance_event")
    async def send_governance_event(
        self, input: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Send a governance event to OpenBox Core.

        Called from WorkflowInboundInterceptor via workflow.execute_activity()
        to maintain workflow determinism. The `input` dict carries the event
        payload and legacy policy marker, never credentials; the Worker-owned
        policy remains authoritative.
        """
        event_payload = input.get("payload", {})
        on_api_error = self._on_api_error

        # Add timestamp in activity context (non-deterministic code allowed)
        payload = {**event_payload, "timestamp": _rfc3339_now()}
        event_type = event_payload.get("event_type", "unknown")

        from openbox_core.errors import ContractError as CoreContractError

        try:
            parsed = await self._governance_client.evaluate_event(payload)
            if parsed is None:
                return {"success": False, "error": "Governance API unavailable"}

            verdict = parsed.verdict
            reason = parsed.reason
            policy_id = parsed.policy_id
            risk_score = parsed.risk_score
            if parsed.fallback_used and verdict is Verdict.HALT:
                return _handle_api_error(
                    event_type,
                    reason or "Governance API unavailable",
                    on_api_error,
                )

            from .errors import GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE
            from .retryable_block import retryable_block_request

            retry_req = retryable_block_request(parsed, event_type=event_type)
            if retry_req is not None:
                raise ApplicationError(
                    "Governance requested workflow restart",
                    retry_req.to_dict(),
                    type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
                    non_retryable=True,
                )

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
        except CoreContractError as error:
            raise GovernanceAPIError(str(error)) from error
        except Exception as error:
            logger.warning(f"Failed to send {event_type} event: {error}")
            if on_api_error == "fail_closed":
                raise GovernanceAPIError(str(error))
            return {"success": False, "error": str(error)}

    async def close(self) -> None:
        """Close the client only when this instance constructed it."""
        if self._owns_governance_client:
            await self._governance_client.close()


def build_governance_activities(
    api_url: str,
    api_key: str,
    *,
    agent_did=None,
    signer=None,
    timeout: float = 30.0,
    on_api_error: str = "fail_open",
    governance_client=None,
) -> GovernanceActivities:
    """Factory used by plugin.py and worker.py to build the activities instance.

    agent_did + signer enable AIP signed requests; both stay on the instance
    (never in inputs).
    """
    # Fall back to the globally-configured signer/DID when omitted (manual setups),
    # so workflow/signal events routed through this activity are signed too.
    from .config import resolve_signing_defaults

    agent_did, signer = resolve_signing_defaults(agent_did, signer)
    return GovernanceActivities(
        api_url=api_url,
        api_key=api_key,
        agent_did=agent_did,
        signer=signer,
        timeout=timeout,
        on_api_error=on_api_error,
        governance_client=governance_client,
    )


# Backward-compat module-level helper.
#
# Not decorated with @activity.defn — worker/plugin register the class-based
# version above so credentials never flow through activity inputs. This shim
# exists for direct callers (tests, scripts) who already hold credentials and
# want to invoke the HTTP logic without constructing the class themselves.
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
        timeout=input.get("timeout", 30.0),
        on_api_error=input.get("on_api_error", "fail_open"),
    )
    forwarded = {k: v for k, v in input.items() if k not in ("api_url", "api_key")}
    try:
        return await instance.send_governance_event(forwarded)
    finally:
        await instance.close()
