"""Temporal compatibility façade over the shared Core EvaluationClient.

This module owns only Temporal result and fail-policy adaptation. HTTP
transport, exact-body signing, and successful-response parsing remain owned by
``openbox_core``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from .types import GovernanceVerdictResponse, Verdict

if TYPE_CHECKING:
    from openbox_core.client import EvaluationClient
    from openbox_core.contracts.results import EvaluationResult


def _check_expiration(data: dict) -> dict:
    """Compatibility wrapper around the shared Core expiration helper."""
    from openbox_core.client import check_expiration

    return check_expiration(data)


def _temporal_response(result: EvaluationResult) -> GovernanceVerdictResponse:
    instance = GovernanceVerdictResponse.__new__(GovernanceVerdictResponse)
    instance.__dict__.update(result.__dict__)
    return instance


class GovernanceClient:
    """Compatibility façade adapting one shared Core client."""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        timeout: float = 30.0,
        on_api_error: str = "fail_open",
        agent_did: Optional[str] = None,
        signer=None,
        _core_client: EvaluationClient | None = None,
    ):
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._on_api_error = on_api_error

        from .config import resolve_signing_defaults

        self._agent_did, self._signer = resolve_signing_defaults(agent_did, signer)
        if _core_client is None:
            from openbox_core.client import EvaluationClient
            from openbox_core.identity import AgentIdentity

            identity = (
                AgentIdentity(self._agent_did, self._signer)
                if self._agent_did and self._signer is not None
                else None
            )
            _core_client = EvaluationClient(
                self._api_url,
                self._api_key,
                timeout_seconds=timeout,
                on_api_error=on_api_error,
                identity=identity,
                sdk_engine="temporal",
            )
        self._core_client = _core_client

    @classmethod
    def _from_core_client(
        cls,
        core_client: EvaluationClient,
        *,
        on_api_error: str,
    ) -> GovernanceClient:
        """Adapt a borrowed runtime client without copying its credentials."""
        instance = cls.__new__(cls)
        instance._on_api_error = on_api_error
        instance._core_client = core_client
        return instance

    async def evaluate_event(
        self, payload: dict
    ) -> Optional[GovernanceVerdictResponse]:
        from openbox_core.errors import GovernanceAPIError as CoreGovernanceAPIError

        try:
            result = await self._core_client.aevaluate(payload)
        except CoreGovernanceAPIError as error:
            return self._handle_api_error(str(error))
        if result.fallback_used:
            return None
        return _temporal_response(result)

    async def poll_approval(
        self, workflow_id: str, run_id: str, activity_id: str
    ) -> Optional[dict]:
        result: Any = await self._core_client.apoll_approval(
            workflow_id, run_id, activity_id
        )
        return None if result is None else dict(result.raw)

    async def close(self) -> None:
        await self._core_client.aclose()

    def _handle_api_error(self, error_msg: str) -> Optional[GovernanceVerdictResponse]:
        if self._on_api_error == "fail_closed":
            return GovernanceVerdictResponse(
                verdict=Verdict.HALT,
                reason=error_msg,
                fallback_used=True,
            )
        return None

    @staticmethod
    def halt_response(reason: str) -> GovernanceVerdictResponse:
        return GovernanceVerdictResponse(verdict=Verdict.HALT, reason=reason)
