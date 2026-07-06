"""OpenBox Temporal SDK — Governance HTTP Client.

Centralizes governance API HTTP calls for activity-level events.
Used by ActivityGovernanceInterceptor for ActivityStarted/ActivityCompleted.

NOT sandbox-safe — uses logging at module level. Do NOT import from
workflow_interceptor.py or other workflow-context code.

Note: httpx is imported lazily inside methods to avoid loading it at module
level. Module-level httpx import triggers Temporal sandbox restrictions
(os.stat). This mirrors the existing pattern in activity_interceptor.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .types import GovernanceVerdictResponse, Verdict

logger = logging.getLogger(__name__)


def _check_expiration(data: dict) -> dict:
    """Check approval_expiration_time and set expired=True if past.

    Modifies data in-place. Returns data dict.
    Handles formats: ISO Z, ISO offset, space-separated from DB.
    """
    expiration_time_str = data.get("approval_expiration_time")
    if not expiration_time_str:
        return data

    try:
        normalized = expiration_time_str.replace("Z", "+00:00").replace(" ", "T")
        expiration_time = datetime.fromisoformat(normalized)
        if expiration_time.tzinfo is None:
            expiration_time = expiration_time.replace(tzinfo=timezone.utc)
        current_time = datetime.now(timezone.utc)
        if current_time > expiration_time:
            data["expired"] = True
    except (ValueError, TypeError) as e:
        logger.warning(
            f"Failed to parse approval_expiration_time '{expiration_time_str}': {e}"
        )

    return data


class GovernanceClient:
    """HTTP client for OpenBox Core governance API.

    Centralizes evaluate_event and poll_approval HTTP calls with
    consistent auth headers and error policy handling.

    Note: Uses per-call httpx.AsyncClient (async with) for test mock
    compatibility. The client object itself provides persistent auth
    header caching and error policy configuration.
    """

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout: float = 30.0,
        on_api_error: str = "fail_open",
        agent_did: Optional[str] = None,
        signer=None,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._on_api_error = on_api_error
        # DID + Ed25519 signer for AIP signed requests (None = unsigned mode).
        # Fall back to the globally-configured signer when omitted so manual
        # setups that called initialize() with signing don't send unsigned calls.
        from .config import resolve_signing_defaults

        self._agent_did, self._signer = resolve_signing_defaults(agent_did, signer)

    async def evaluate_event(
        self, payload: dict
    ) -> Optional[GovernanceVerdictResponse]:
        """Send governance event to /api/v1/governance/evaluate.

        Args:
            payload: Pre-built governance event payload dict.

        Returns:
            GovernanceVerdictResponse on success.
            None on API error with fail_open policy.
            GovernanceVerdictResponse(HALT) on API error with fail_closed policy.
        """
        # Lazy import — avoids Temporal sandbox restrictions at module level
        import httpx
        from .request_signing import prepare_signed_request

        headers, body = prepare_signed_request(
            "POST",
            "/api/v1/governance/evaluate",
            payload,
            api_key=self._api_key,
            agent_did=self._agent_did,
            signer=self._signer,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._api_url}/api/v1/governance/evaluate",
                    content=body,
                    headers=headers,
                )

                if response.status_code >= 400:
                    error_msg = f"HTTP {response.status_code}"
                    logger.warning(f"Governance API error: {error_msg}")
                    return self._handle_api_error(f"Governance API error: {error_msg}")

                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.info(
                            f"Governance response: verdict={data.get('verdict') or data.get('action', 'unknown')}, "
                            f"reason={data.get('reason')}"
                        )
                        verdict = GovernanceVerdictResponse.from_dict(data)
                        if verdict.verdict.should_stop():
                            logger.info(
                                f"Governance blocked: {verdict.reason} (policy: {verdict.policy_id})"
                            )
                        if verdict.guardrails_result:
                            logger.info(
                                f"Guardrails redaction: input_type={verdict.guardrails_result.input_type}"
                            )
                        return verdict
                    except Exception as e:
                        logger.warning(f"Failed to parse governance response: {e}")

                return None

        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            logger.warning(f"Governance API error ({type(e).__name__}): {error_msg}")
            return self._handle_api_error(f"Governance API error: {error_msg}")

    async def poll_approval(
        self, workflow_id: str, run_id: str, activity_id: str
    ) -> Optional[dict]:
        """Poll /api/v1/governance/approval for HITL status.

        Returns dict with verdict/action and optional fields, or None on failure.
        Sets expired=True in the dict if approval_expiration_time has passed.
        """
        # Lazy import — avoids Temporal sandbox restrictions at module level
        import httpx
        from .request_signing import prepare_signed_request

        payload = {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "activity_id": activity_id,
        }

        headers, body = prepare_signed_request(
            "POST",
            "/api/v1/governance/approval",
            payload,
            api_key=self._api_key,
            agent_did=self._agent_did,
            signer=self._signer,
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._api_url}/api/v1/governance/approval",
                    content=body,
                    headers=headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Approval status response: {data}")
                    _check_expiration(data)
                    return data

                logger.warning(
                    f"Failed to get approval status: HTTP {response.status_code}"
                )
                return None

        except Exception as e:
            logger.warning(f"Failed to poll approval status: {e}")
            return None

    async def close(self) -> None:
        """No-op: per-call clients close automatically via async with."""
        pass

    def _handle_api_error(self, error_msg: str) -> Optional[GovernanceVerdictResponse]:
        """Apply on_api_error policy. Returns None (fail_open) or HALT (fail_closed)."""
        if self._on_api_error == "fail_closed":
            return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=error_msg)
        return None

    @staticmethod
    def halt_response(reason: str) -> GovernanceVerdictResponse:
        """Build a HALT verdict response."""
        return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=reason)
