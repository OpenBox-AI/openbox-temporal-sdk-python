from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_core.contracts.results import ApprovalResult, EvaluationResult, Verdict
from openbox_core.errors import ContractError, GovernanceAPIError

from openbox.client import GovernanceClient
from openbox.types import GovernanceVerdictResponse


def core_client() -> MagicMock:
    client = MagicMock()
    client.aevaluate = AsyncMock()
    client.apoll_approval = AsyncMock()
    client.aclose = AsyncMock()
    return client


def test_constructor_delegates_transport_ownership_to_core_client() -> None:
    with patch("openbox_core.client.EvaluationClient") as client_type:
        value = GovernanceClient(
            api_url="https://core.example/",
            api_key="obx_test_key",
            timeout=12.0,
            on_api_error="fail_closed",
        )
    assert value._core_client._base is client_type.return_value
    client_type.assert_called_once_with(
        "https://core.example",
        "obx_test_key",
        timeout_seconds=12.0,
        on_api_error="fail_closed",
        identity=None,
        sdk_engine="temporal",
    )


def test_worker_owned_core_client_is_adapted_without_reconstruction() -> None:
    shared = core_client()
    with patch("openbox_core.client.EvaluationClient") as client_type:
        value = GovernanceClient._from_core_client(
            shared,
            api_url="https://core.example",
            api_key="obx_test_key",
            timeout=30.0,
            on_api_error="fail_open",
            agent_did=None,
            signer=None,
        )
    assert value._core_client is shared
    client_type.assert_not_called()


@pytest.mark.asyncio
async def test_evaluate_maps_shared_result_without_reparsing() -> None:
    shared = core_client()
    shared_result = EvaluationResult.from_dict(
        {
            "verdict": "constrain",
            "reason": "sandbox required",
            "policy_id": "policy-1",
            "constraints": ["run_in_sandbox"],
        }
    )
    shared_result.raw = {"unknown": "preserved"}
    shared.aevaluate.return_value = shared_result
    value = GovernanceClient._from_core_client(
        shared,
        api_url="https://core.example",
        api_key="obx_test_key",
        timeout=30.0,
        agent_did=None,
        signer=None,
        on_api_error="fail_open",
    )

    result = await value.evaluate_event({"event_type": "ActivityStarted"})

    assert isinstance(result, GovernanceVerdictResponse)
    assert result.verdict is Verdict.CONSTRAIN
    assert result.constraints == ["run_in_sandbox"]
    assert result.raw == {"unknown": "preserved"}
    shared.aevaluate.assert_awaited_once_with({"event_type": "ActivityStarted"})


@pytest.mark.asyncio
async def test_fail_open_fallback_remains_distinguishable_and_returns_none() -> None:
    shared = core_client()
    shared.aevaluate.return_value = EvaluationResult.fallback_allow("Core unavailable")
    value = GovernanceClient._from_core_client(
        shared,
        api_url="https://core.example",
        api_key="obx_test_key",
        timeout=30.0,
        agent_did=None,
        signer=None,
        on_api_error="fail_open",
    )
    assert await value.evaluate_event({}) is None


@pytest.mark.asyncio
async def test_fail_closed_network_error_maps_to_marked_halt() -> None:
    shared = core_client()
    shared.aevaluate.side_effect = GovernanceAPIError("Core unavailable")
    value = GovernanceClient._from_core_client(
        shared,
        api_url="https://core.example",
        api_key="obx_test_key",
        timeout=30.0,
        agent_did=None,
        signer=None,
        on_api_error="fail_closed",
    )
    result = await value.evaluate_event({})
    assert result is not None
    assert result.verdict is Verdict.HALT
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_contract_error_is_never_converted_to_fail_open() -> None:
    shared = core_client()
    error = ContractError("Malformed governance response", code="RESPONSE_INVALID")
    shared.aevaluate.side_effect = error
    value = GovernanceClient._from_core_client(
        shared,
        api_url="https://core.example",
        api_key="obx_test_key",
        timeout=30.0,
        agent_did=None,
        signer=None,
        on_api_error="fail_open",
    )
    with pytest.raises(ContractError) as exc_info:
        await value.evaluate_event({})
    assert exc_info.value is error


@pytest.mark.asyncio
async def test_approval_and_close_delegate_to_same_shared_client() -> None:
    shared = core_client()
    shared.apoll_approval.return_value = ApprovalResult(
        verdict=Verdict.ALLOW,
        raw={"verdict": "allow", "unknown": "preserved"},
    )
    value = GovernanceClient._from_core_client(
        shared,
        api_url="https://core.example",
        api_key="obx_test_key",
        timeout=30.0,
        agent_did=None,
        signer=None,
        on_api_error="fail_open",
    )

    result = await value.poll_approval("wf-1", "run-1", "activity-1")
    await value.close()

    assert result == {"verdict": "allow", "unknown": "preserved"}
    shared.apoll_approval.assert_awaited_once_with("wf-1", "run-1", "activity-1")
    shared.aclose.assert_awaited_once_with()
