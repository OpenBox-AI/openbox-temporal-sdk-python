"""Provider-neutral Keycloak workload identity forwarding."""

from __future__ import annotations

from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from openbox.activities import GovernanceActivities
from openbox.client import GovernanceClient
from openbox.types import GovernanceVerdictResponse, Verdict

API_URL = "http://localhost:8086"
API_KEY = "obx_test_workload_agent"
WORKLOAD_PRIVATE_KEY = "workload-private-key"


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
        return_value=MagicMock(
            raw={"verdict": "allow", "reason": "resource access allowed"}
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
    evaluation_client.assert_called_once_with(
        API_URL,
        API_KEY,
        timeout_seconds=30.0,
        on_api_error="fail_closed",
        workload_private_key=WORKLOAD_PRIVATE_KEY,
        sdk_version=ANY,
    )
    base_client.aevaluate.assert_awaited_once_with({"event_type": "ActivityStarted"})


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
