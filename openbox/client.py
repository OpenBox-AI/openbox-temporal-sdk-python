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
from typing import Any, Optional

from .types import GovernanceVerdictResponse, Verdict

logger = logging.getLogger(__name__)

# v1 (OpenBox DID / unsigned) routes — byte-compatible, unchanged.
_EVALUATE_PATH = "/api/v1/governance/evaluate"
_APPROVAL_PATH = "/api/v1/governance/approval"
# v2 (Okta AI Agent) routes (proposal §13.7; contract §2.2).
_EVALUATE_PATH_V2 = "/api/v2/governance/evaluate"
_APPROVAL_PATH_V2 = "/api/v2/governance/approval"


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


def _extract_reason_code(body: Optional[bytes]) -> Optional[str]:
    """Machine reason code from Core's JSON error body, if present."""
    import json

    if not body:
        return None
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    code = data.get("reason_code") or data.get("code") or data.get("reason")
    return code if isinstance(code, str) else None


def _raise_okta_auth_failure(response: Any) -> None:
    """Raise an actionable auth error for a 401/403 v2 (Okta) response.

    Delegates reason-code CLASSIFICATION to the base SDK's own
    ``map_signing_error`` (contract §7's full v2 code set) but always raises
    THIS package's own ``OpenBoxSigningError``/``OpenBoxAuthError`` type, so
    callers catching ``openbox.errors.*`` work identically for v1 and v2
    (proposal §13.6 — 401/403 are hard authentication failures, never
    laundered into a fallback ALLOW or "still pending").
    """
    from .errors import OpenBoxAuthError, OpenBoxSigningError

    reason_code = _extract_reason_code(getattr(response, "content", None))
    if reason_code:
        from openbox_core.errors import map_signing_error as core_map_signing_error

        core_exc = core_map_signing_error(reason_code)
        raise OpenBoxSigningError(str(core_exc), reason_code) from core_exc
    raise OpenBoxAuthError(
        f"Authentication rejected (HTTP {response.status_code}). "
        "Check your API key at dashboard.openbox.ai"
    )


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
        okta_identity=None,
        workload_private_key: Optional[str] = None,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._on_api_error = on_api_error
        # DID + Ed25519 signer for AIP signed requests (None = unsigned mode).
        # okta_identity (v2, proposal §13.7): a loaded OktaAgentIdentity —
        # mutually exclusive with agent_did/signer. Fall back to the
        # globally-configured identity when ALL THREE are omitted, so manual
        # setups that called initialize() with signing don't send unsigned
        # (or wrong-version) calls.
        from .config import resolve_signing_defaults

        self._agent_did, self._signer, self._okta_identity = resolve_signing_defaults(
            agent_did, signer, okta_identity
        )
        self._workload_client = None
        if workload_private_key:
            from openbox_core.client import EvaluationClient

            from .request_signing import _sdk_identifier

            self._workload_client = EvaluationClient(
                self._api_url,
                self._api_key,
                timeout_seconds=self._timeout,
                on_api_error=self._on_api_error,
                workload_private_key=workload_private_key,
                sdk_version=_sdk_identifier(),
            )

    async def evaluate_event(
        self, payload: dict
    ) -> Optional[GovernanceVerdictResponse]:
        """Send governance event to the version-appropriate evaluate route.

        Args:
            payload: Pre-built governance event payload dict.

        Returns:
            GovernanceVerdictResponse on success.
            None on API error with fail_open policy.
            GovernanceVerdictResponse(HALT) on API error with fail_closed policy.

        Raises:
            OpenBoxAuthError / OpenBoxSigningError: Okta (v2) mode ONLY — a
                401/403 fails closed unconditionally (proposal §13.6), never
                laundered into a fallback ALLOW. v1/unsigned mode preserves
                its existing (`_handle_api_error`) behavior unchanged.
        """
        # Lazy import — avoids Temporal sandbox restrictions at module level
        import httpx

        from .errors import OpenBoxAuthError

        if self._workload_client is not None:
            try:
                result = await self._workload_client.aevaluate(payload)
            except Exception as exc:
                return self._handle_workload_error(exc)
            data = dict(result.raw)
            data.setdefault("verdict", result.verdict.value)
            data.setdefault("reason", result.reason)
            data.setdefault("policy_id", result.policy_id)
            data.setdefault("risk_score", result.risk_score)
            return GovernanceVerdictResponse.from_dict(data)

        if self._okta_identity is not None:
            from .request_signing import prepare_okta_signed_request

            headers, body = prepare_okta_signed_request(
                "POST",
                _EVALUATE_PATH_V2,
                payload,
                api_key=self._api_key,
                okta_identity=self._okta_identity,
            )
            evaluate_path = _EVALUATE_PATH_V2
        else:
            from .request_signing import prepare_signed_request

            headers, body = prepare_signed_request(
                "POST",
                _EVALUATE_PATH,
                payload,
                api_key=self._api_key,
                agent_did=self._agent_did,
                signer=self._signer,
            )
            evaluate_path = _EVALUATE_PATH

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._api_url}{evaluate_path}",
                    content=body,
                    headers=headers,
                )

                if self._okta_identity is not None and response.status_code in (
                    401,
                    403,
                ):
                    _raise_okta_auth_failure(response)

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

        except OpenBoxAuthError:
            # Raised by `_raise_okta_auth_failure` above (a 401/403 in Okta
            # mode) — an authentication failure, never a network error
            # (proposal §13.6). Must propagate, never be laundered into a
            # fallback ALLOW/HALT via `_handle_api_error` below.
            raise
        except Exception as e:
            error_msg = str(e) if str(e) else repr(e)
            logger.warning(f"Governance API error ({type(e).__name__}): {error_msg}")
            return self._handle_api_error(f"Governance API error: {error_msg}")

    async def poll_approval(
        self, workflow_id: str, run_id: str, activity_id: str
    ) -> Optional[dict]:
        """Poll the version-appropriate approval route for HITL status.

        Returns dict with verdict/action and optional fields, or None on failure.
        Sets expired=True in the dict if approval_expiration_time has passed.

        Raises:
            OpenBoxAuthError / OpenBoxSigningError: Okta (v2) mode ONLY — a
                401/403 fails closed unconditionally (proposal §13.6) rather
                than being read as "still pending" and retried forever.
        """
        # Lazy import — avoids Temporal sandbox restrictions at module level
        import httpx

        from .errors import OpenBoxAuthError

        payload = {
            "workflow_id": workflow_id,
            "run_id": run_id,
            "activity_id": activity_id,
        }

        if self._workload_client is not None:
            try:
                result = await self._workload_client.apoll_approval(
                    workflow_id, run_id, activity_id
                )
            except Exception as exc:
                self._handle_workload_error(exc)
                return None
            return dict(result.raw) if result is not None else None

        if self._okta_identity is not None:
            from .request_signing import prepare_okta_signed_request

            headers, body = prepare_okta_signed_request(
                "POST",
                _APPROVAL_PATH_V2,
                payload,
                api_key=self._api_key,
                okta_identity=self._okta_identity,
            )
            approval_path = _APPROVAL_PATH_V2
        else:
            from .request_signing import prepare_signed_request

            headers, body = prepare_signed_request(
                "POST",
                _APPROVAL_PATH,
                payload,
                api_key=self._api_key,
                agent_did=self._agent_did,
                signer=self._signer,
            )
            approval_path = _APPROVAL_PATH

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._api_url}{approval_path}",
                    content=body,
                    headers=headers,
                )

                if self._okta_identity is not None and response.status_code in (
                    401,
                    403,
                ):
                    _raise_okta_auth_failure(response)

                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Approval status response: {data}")
                    _check_expiration(data)
                    return data

                logger.warning(
                    f"Failed to get approval status: HTTP {response.status_code}"
                )
                return None

        except OpenBoxAuthError:
            # Raised by `_raise_okta_auth_failure` above — must propagate
            # (proposal §13.6: never read a hard auth failure as "still
            # pending" and keep polling forever).
            raise
        except Exception as e:
            logger.warning(f"Failed to poll approval status: {e}")
            return None

    async def close(self) -> None:
        """Close the shared base client used for v3 workload requests."""

        if self._workload_client is not None:
            await self._workload_client.aclose()

    def _handle_workload_error(
        self, exc: Exception
    ) -> Optional[GovernanceVerdictResponse]:
        """Translate base-SDK v3 errors into this package's public errors."""

        from openbox_core.errors import GovernanceAPIError as CoreGovernanceAPIError
        from openbox_core.errors import OpenBoxAuthError as CoreOpenBoxAuthError
        from openbox_core.errors import OpenBoxNetworkError as CoreOpenBoxNetworkError
        from openbox_core.errors import OpenBoxSigningError as CoreOpenBoxSigningError

        from .errors import OpenBoxAuthError, OpenBoxSigningError

        if isinstance(exc, CoreOpenBoxSigningError):
            raise OpenBoxSigningError(
                str(exc), getattr(exc, "reason_code", None)
            ) from exc
        if isinstance(exc, CoreOpenBoxAuthError):
            raise OpenBoxAuthError(str(exc)) from exc
        if isinstance(exc, (CoreGovernanceAPIError, CoreOpenBoxNetworkError)):
            return self._handle_api_error(str(exc))
        raise exc

    def _handle_api_error(self, error_msg: str) -> Optional[GovernanceVerdictResponse]:
        """Apply on_api_error policy. Returns None (fail_open) or HALT (fail_closed)."""
        if self._on_api_error == "fail_closed":
            return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=error_msg)
        return None

    @staticmethod
    def halt_response(reason: str) -> GovernanceVerdictResponse:
        """Build a HALT verdict response."""
        return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=reason)
