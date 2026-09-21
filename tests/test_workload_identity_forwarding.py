"""Provider-neutral Keycloak workload identity forwarding."""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest
from openbox_core.contracts.results import EvaluationResult

from openbox.activities import GovernanceActivities
from openbox.client import GovernanceClient
from openbox.config import initialize
from openbox.errors import OpenBoxConfigError
from openbox.types import GovernanceVerdictResponse, Verdict

API_URL = "http://localhost:8086"
API_KEY = "obx_test_workload_agent"
WORKLOAD_PRIVATE_KEY = "workload-private-key"


@pytest.mark.parametrize(
    "workload_private_key",
    [
        "not-a-key",
        "-----BEGIN PRIVATE KEY-----\nsensitive-junk\n-----END PRIVATE KEY-----",
    ],
)
def test_initialize_names_malformed_workload_key_without_disclosing_it(
    workload_private_key,
):
    with pytest.raises(OpenBoxConfigError) as exc_info:
        initialize(
            API_URL,
            API_KEY,
            workload_private_key=workload_private_key,
        )

    assert str(exc_info.value) == (
        "Invalid workload_private_key: could not load a PKCS8 PEM RSA "
        "private key (key bytes not shown)."
    )
    assert workload_private_key not in str(exc_info.value)


def test_plugin_forwards_workload_key_to_every_governance_path():
    runtime = MagicMock()
    with (
        patch("openbox.plugin.validate_api_key") as validate,
        patch(
            "openbox.core_adapter.create_core_runtime", return_value=runtime
        ) as create_runtime,
        patch("openbox.plugin.GovernanceClient") as governance_client,
        patch("openbox.activities.build_governance_activities") as build_activities,
        patch("openbox.workflow_interceptor.GovernanceInterceptor"),
        patch("openbox.activity_interceptor.ActivityGovernanceInterceptor"),
    ):
        from openbox.plugin import OpenBoxPlugin

        OpenBoxPlugin(
            openbox_url=API_URL,
            openbox_api_key=API_KEY,
            workload_private_key=WORKLOAD_PRIVATE_KEY,
            enable_trace_propagation=False,
        )

    assert validate.call_args.kwargs["workload_private_key"] == WORKLOAD_PRIVATE_KEY
    assert (
        create_runtime.call_args.kwargs["workload_private_key"] == WORKLOAD_PRIVATE_KEY
    )
    assert (
        governance_client.call_args.kwargs["workload_private_key"]
        == WORKLOAD_PRIVATE_KEY
    )
    assert (
        build_activities.call_args.kwargs["workload_private_key"]
        == WORKLOAD_PRIVATE_KEY
    )


@pytest.mark.asyncio
async def test_governance_client_delegates_workload_requests_to_base_v3_client():
    base_client = MagicMock()
    base_client.aevaluate = AsyncMock(
        return_value=EvaluationResult.from_dict(
            {"verdict": "allow", "reason": "resource access allowed"}
        )
    )

    with patch(
        "openbox_core.client.EvaluationClient", return_value=base_client
    ) as evaluation_client:
        client = GovernanceClient(
            api_url=API_URL,
            api_key=API_KEY,
            on_api_error="fail_closed",
            workload_private_key=WORKLOAD_PRIVATE_KEY,
        )
        result = await client.evaluate_event({"event_type": "ActivityStarted"})

    assert result is not None
    assert result.verdict is Verdict.ALLOW
    assert result.reason == "resource access allowed"
    assert result.fallback_used is False
    evaluation_client.assert_called_once_with(
        API_URL,
        API_KEY,
        timeout_seconds=30.0,
        on_api_error="fail_closed",
        workload_private_key=WORKLOAD_PRIVATE_KEY,
        sdk_version=ANY,
    )
    base_client.aevaluate.assert_awaited_once_with({"event_type": "ActivityStarted"})


@pytest.mark.parametrize(
    "reason",
    [
        "Governance API error: HTTP 500",
        "Governance API returned unparseable body: invalid response",
    ],
)
@pytest.mark.asyncio
async def test_governance_client_preserves_workload_fail_open_marker(reason):
    base_client = MagicMock()
    base_client.aevaluate = AsyncMock(
        return_value=EvaluationResult.fallback_allow(reason)
    )

    with patch("openbox_core.client.EvaluationClient", return_value=base_client):
        client = GovernanceClient(
            api_url=API_URL,
            api_key=API_KEY,
            workload_private_key=WORKLOAD_PRIVATE_KEY,
        )
        result = await client.evaluate_event({"event_type": "ActivityStarted"})

    assert result is not None
    assert result.verdict is Verdict.ALLOW
    assert result.reason == reason
    assert result.fallback_used is True


@pytest.mark.asyncio
async def test_workflow_activity_uses_workload_aware_governance_client():
    workload_client = MagicMock()
    workload_client.evaluate_event = AsyncMock(
        return_value=GovernanceVerdictResponse(
            verdict=Verdict.ALLOW,
            reason="resource access allowed",
        )
    )
    workload_client.close = AsyncMock()

    with patch(
        "openbox.client.GovernanceClient", return_value=workload_client
    ) as governance_client:
        activities = GovernanceActivities(
            API_URL,
            API_KEY,
            workload_private_key=WORKLOAD_PRIVATE_KEY,
        )
        result = await activities.send_governance_event(
            {
                "payload": {"event_type": "WorkflowStarted"},
                "on_api_error": "fail_closed",
                "timeout": 12.0,
            }
        )

    assert result == {
        "success": True,
        "verdict": "allow",
        "action": "allow",
        "reason": "resource access allowed",
        "policy_id": None,
        "risk_score": 0.0,
    }
    governance_client.assert_called_once_with(
        api_url=API_URL,
        api_key=API_KEY,
        timeout=12.0,
        on_api_error="fail_closed",
        workload_private_key=WORKLOAD_PRIVATE_KEY,
    )
    workload_client.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_workflow_activity_forwards_workload_fail_open_marker():
    reason = "Governance API error: HTTP 500"
    workload_client = MagicMock()
    workload_client.evaluate_event = AsyncMock(
        return_value=GovernanceVerdictResponse(
            verdict=Verdict.ALLOW,
            reason=reason,
            fallback_used=True,
        )
    )
    workload_client.close = AsyncMock()

    with patch("openbox.client.GovernanceClient", return_value=workload_client):
        activities = GovernanceActivities(
            API_URL,
            API_KEY,
            workload_private_key=WORKLOAD_PRIVATE_KEY,
        )
        result = await activities.send_governance_event(
            {"payload": {"event_type": "WorkflowStarted"}}
        )

    assert result == {
        "success": True,
        "verdict": "allow",
        "action": "allow",
        "reason": reason,
        "policy_id": None,
        "risk_score": 0.0,
        "fallback_used": True,
    }
