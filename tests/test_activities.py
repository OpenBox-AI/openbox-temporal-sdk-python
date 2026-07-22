"""
Comprehensive pytest tests for the OpenBox SDK activities module.

Tests cover:
- _rfc3339_now() timestamp formatting
- GovernanceAPIError exception
- raise_governance_block() and _terminate_workflow_for_halt() behavior
- send_governance_event() activity with various scenarios
"""

import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openbox_core.errors import ContractError
from temporalio.exceptions import ApplicationError

from openbox.activities import (
    GovernanceActivities,
    GovernanceAPIError,
    _rfc3339_now,
    _terminate_workflow_for_halt,
    raise_governance_block,
    send_governance_event as send_governance_event_compat,
)
from openbox.retryable_block import GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION
from openbox.types import GovernanceVerdictResponse, Verdict


class TestRfc3339Now:
    """Tests for the _rfc3339_now() function."""

    def test_returns_string(self):
        """Test that _rfc3339_now returns a string."""
        result = _rfc3339_now()
        assert isinstance(result, str)

    def test_ends_with_z(self):
        """Test that the timestamp ends with 'Z' (UTC indicator)."""
        result = _rfc3339_now()
        assert result.endswith("Z")

    def test_format_matches_rfc3339(self):
        """Test that the format matches YYYY-MM-DDTHH:MM:SS.sssZ."""
        result = _rfc3339_now()
        # RFC3339 pattern: 2024-01-15T10:30:45.123Z
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
        assert re.match(pattern, result), (
            f"Timestamp '{result}' does not match RFC3339 format"
        )

    def test_timestamp_is_valid_datetime(self):
        """Test that the timestamp can be parsed back to a valid datetime."""
        result = _rfc3339_now()
        # Remove trailing Z and parse
        dt_str = result[:-1]  # Remove 'Z'
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
        assert isinstance(dt, datetime)

    def test_timestamp_is_recent(self):
        """Test that the timestamp is approximately the current time."""
        before = datetime.now(timezone.utc)
        result = _rfc3339_now()
        after = datetime.now(timezone.utc)

        # Parse the result
        dt_str = result[:-1]  # Remove 'Z'
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
        dt = dt.replace(tzinfo=timezone.utc)

        # The function truncates to milliseconds, so we need to account for that.
        # Truncate 'before' to milliseconds as well for fair comparison.
        from datetime import timedelta

        # Allow 1 second tolerance since truncation can cause dt to be slightly before 'before'
        tolerance = timedelta(seconds=1)
        assert (before - tolerance) <= dt <= (after + tolerance)

    def test_millisecond_precision(self):
        """Test that the timestamp has exactly 3 decimal places (milliseconds)."""
        result = _rfc3339_now()
        # Extract the fractional seconds part
        match = re.search(r"\.(\d+)Z$", result)
        assert match is not None
        fractional = match.group(1)
        assert len(fractional) == 3, f"Expected 3 decimal places, got {len(fractional)}"


class TestGovernanceAPIError:
    """Tests for the GovernanceAPIError exception class."""

    def test_can_be_raised(self):
        """Test that GovernanceAPIError can be raised."""
        with pytest.raises(GovernanceAPIError):
            raise GovernanceAPIError("Test error")

    def test_can_be_caught(self):
        """Test that GovernanceAPIError can be caught."""
        try:
            raise GovernanceAPIError("Test error")
        except GovernanceAPIError as e:
            assert str(e) == "Test error"

    def test_inherits_from_exception(self):
        """Test that GovernanceAPIError inherits from Exception."""
        assert issubclass(GovernanceAPIError, Exception)

    def test_with_empty_message(self):
        """Test GovernanceAPIError with empty message."""
        with pytest.raises(GovernanceAPIError) as exc_info:
            raise GovernanceAPIError("")
        assert str(exc_info.value) == ""

    def test_can_be_caught_as_base_exception(self):
        """Test that GovernanceAPIError can be caught as Exception."""
        try:
            raise GovernanceAPIError("Test error")
        except Exception as e:
            assert isinstance(e, GovernanceAPIError)


class TestRaiseGovernanceBlock:
    """Tests for the raise_governance_block() function."""

    def test_raises_application_error(self):
        with pytest.raises(ApplicationError):
            raise_governance_block("Test reason")

    def test_error_message_format(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Policy violation detected")
        assert "Governance blocked: Policy violation detected" in str(exc_info.value)

    def test_error_type_is_governance_block(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Test reason")
        assert exc_info.value.type == "GovernanceBlock"

    def test_non_retryable_is_true(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Test reason")
        assert exc_info.value.non_retryable is True

    def test_includes_policy_id_in_details(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Test reason", policy_id="policy-123")
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["policy_id"] == "policy-123"

    def test_includes_risk_score_in_details(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Test reason", risk_score=0.85)
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["risk_score"] == 0.85

    def test_default_values_are_none(self):
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_block("Test reason")
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["policy_id"] is None
        assert details[0]["risk_score"] is None


class TestTerminateWorkflowForHalt:
    """Tests for the _terminate_workflow_for_halt() function."""

    @pytest.mark.asyncio
    async def test_calls_client_terminate_when_client_available(self):
        """HALT with client calls terminate() then raises ApplicationError to stop activity."""
        from openbox.activities import set_temporal_client

        mock_handle = MagicMock()
        mock_handle.terminate = AsyncMock()
        mock_client = MagicMock()
        mock_client.get_workflow_handle.return_value = mock_handle

        set_temporal_client(mock_client)
        try:
            with pytest.raises(ApplicationError) as exc_info:
                await _terminate_workflow_for_halt("wf-123", "policy violation")
            # Verify terminate was called before the raise
            mock_client.get_workflow_handle.assert_called_once_with("wf-123")
            mock_handle.terminate.assert_called_once_with(
                "Governance HALT: policy violation"
            )
            assert exc_info.value.type == "GovernanceHalt"
        finally:
            set_temporal_client(None)

    @pytest.mark.asyncio
    async def test_fallback_to_application_error_without_client(self):
        """HALT without client should raise ApplicationError as fallback."""
        from openbox.activities import set_temporal_client

        set_temporal_client(None)
        with pytest.raises(ApplicationError) as exc_info:
            await _terminate_workflow_for_halt("wf-123", "policy violation")
        assert exc_info.value.type == "GovernanceHalt"
        assert exc_info.value.non_retryable is True


class TestSendGovernanceEvent:
    def _activity(self, response):
        client = MagicMock()
        client.evaluate_event = AsyncMock(return_value=response)
        return (
            GovernanceActivities(
                "https://core.invalid",
                "obx_test_key",
                governance_client=client,
            ),
            client,
        )

    @pytest.mark.asyncio
    async def test_allow_uses_injected_shared_client_and_preserves_result_shape(self):
        activity_instance, client = self._activity(
            GovernanceVerdictResponse(
                verdict=Verdict.ALLOW,
                reason="allowed",
                policy_id="policy-1",
                risk_score=0.1,
            )
        )
        result = await activity_instance.send_governance_event(
            {"payload": {"event_type": "WorkflowStarted", "workflow_id": "wf-1"}}
        )
        assert result == {
            "success": True,
            "verdict": "allow",
            "action": "allow",
            "reason": "allowed",
            "policy_id": "policy-1",
            "risk_score": 0.1,
        }
        payload = client.evaluate_event.await_args.args[0]
        assert payload["event_type"] == "WorkflowStarted"
        assert payload["timestamp"].endswith("Z")

    @pytest.mark.asyncio
    async def test_fail_open_none_returns_error_without_second_transport(self):
        activity_instance, client = self._activity(None)
        result = await activity_instance.send_governance_event(
            {"payload": {"event_type": "WorkflowStarted"}}
        )
        assert result == {"success": False, "error": "Governance API unavailable"}
        client.evaluate_event.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_block_raises_temporal_application_error(self):
        activity_instance, _ = self._activity(
            GovernanceVerdictResponse(verdict=Verdict.BLOCK, reason="blocked")
        )
        with pytest.raises(ApplicationError) as exc_info:
            await activity_instance.send_governance_event(
                {"payload": {"event_type": "WorkflowStarted"}}
            )
        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    async def test_signal_block_returns_durable_result(self):
        activity_instance, _ = self._activity(
            GovernanceVerdictResponse(verdict=Verdict.BLOCK, reason="blocked")
        )
        result = await activity_instance.send_governance_event(
            {"payload": {"event_type": "SignalReceived"}}
        )
        assert result["verdict"] == "block"
        assert result["success"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verdict", [Verdict.CONSTRAIN, Verdict.REQUIRE_APPROVAL])
    async def test_nonterminal_verdicts_preserve_compatibility_shape(self, verdict):
        activity_instance, _ = self._activity(
            GovernanceVerdictResponse(
                verdict=verdict,
                reason="evaluated",
                policy_id="policy-2",
                risk_score=0.2,
            )
        )
        result = await activity_instance.send_governance_event(
            {"payload": {"event_type": "WorkflowCompleted"}}
        )
        assert result == {
            "success": True,
            "verdict": verdict.value,
            "action": verdict.value,
            "reason": "evaluated",
            "policy_id": "policy-2",
            "risk_score": 0.2,
        }

    @pytest.mark.asyncio
    async def test_halt_keeps_temporal_termination_behavior(self):
        from openbox.activities import set_temporal_client

        set_temporal_client(None)
        activity_instance, _ = self._activity(
            GovernanceVerdictResponse(verdict=Verdict.HALT, reason="halted")
        )
        with pytest.raises(ApplicationError) as exc_info:
            await activity_instance.send_governance_event(
                {
                    "payload": {
                        "event_type": "WorkflowCompleted",
                        "workflow_id": "wf-1",
                    }
                }
            )
        assert exc_info.value.type == "GovernanceHalt"
        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type",
        [
            "WorkflowStarted",
            "WorkflowCompleted",
            "WorkflowFailed",
            "SignalReceived",
            "Handoff",
        ],
    )
    async def test_retryable_block_precedes_plain_stop_handling(self, event_type):
        activity_instance, _ = self._activity(
            GovernanceVerdictResponse.from_dict(
                {
                    "verdict": "block",
                    "reason": "retry corrected input",
                    "governance_event_id": "evt-retry-1",
                    "retry_plan": {"new_input": {"query": "corrected"}},
                }
            )
        )
        with pytest.raises(ApplicationError) as exc_info:
            await activity_instance.send_governance_event(
                {"payload": {"event_type": event_type}}
            )
        assert exc_info.value.type == "GovernanceRetryableBlock"
        assert exc_info.value.non_retryable is True
        assert exc_info.value.details[0] == {
            "schema_version": GOVERNANCE_RETRYABLE_BLOCK_SCHEMA_VERSION,
            "new_input": {"query": "corrected"},
            "event_type": event_type,
            "governance_event_id": "evt-retry-1",
            "reason": "retry corrected input",
            "hook_trigger": False,
            "hook_stage": None,
        }

    @pytest.mark.asyncio
    async def test_malformed_success_never_becomes_fail_open(self):
        client = MagicMock()
        client.evaluate_event = AsyncMock(
            side_effect=ContractError(
                "Malformed governance response",
                code="RESPONSE_INVALID",
            )
        )
        activity_instance = GovernanceActivities(
            "https://core.invalid",
            "obx_test_key",
            governance_client=client,
        )
        with pytest.raises(GovernanceAPIError, match="Malformed governance response"):
            await activity_instance.send_governance_event(
                {
                    "payload": {"event_type": "WorkflowStarted"},
                    "on_api_error": "fail_open",
                }
            )

    @pytest.mark.asyncio
    async def test_shared_client_fail_closed_fallback_preserves_api_error_contract(
        self,
    ):
        client = MagicMock()
        client.evaluate_event = AsyncMock(
            return_value=GovernanceVerdictResponse(
                verdict=Verdict.HALT,
                reason="Core unavailable",
                fallback_used=True,
            )
        )
        activity_instance = GovernanceActivities(
            "https://core.invalid",
            "obx_test_key",
            on_api_error="fail_closed",
            governance_client=client,
        )
        with pytest.raises(GovernanceAPIError, match="Core unavailable"):
            await activity_instance.send_governance_event(
                {
                    "payload": {"event_type": "WorkflowStarted"},
                    "on_api_error": "fail_closed",
                }
            )

    @pytest.mark.asyncio
    async def test_unexpected_client_failure_respects_configured_fail_policy(self):
        client = MagicMock()
        client.evaluate_event = AsyncMock(side_effect=RuntimeError("transport failed"))
        fail_open_activity = GovernanceActivities(
            "https://core.invalid",
            "obx_test_key",
            on_api_error="fail_open",
            governance_client=client,
        )
        fail_open = await fail_open_activity.send_governance_event(
            {
                "payload": {"event_type": "WorkflowStarted"},
                "on_api_error": "fail_closed",
            }
        )
        assert fail_open == {"success": False, "error": "transport failed"}

        fail_closed_activity = GovernanceActivities(
            "https://core.invalid",
            "obx_test_key",
            on_api_error="fail_closed",
            governance_client=client,
        )
        with pytest.raises(GovernanceAPIError, match="transport failed"):
            await fail_closed_activity.send_governance_event(
                {
                    "payload": {"event_type": "WorkflowStarted"},
                    "on_api_error": "fail_open",
                }
            )

    @pytest.mark.asyncio
    async def test_compatibility_helper_closes_its_owned_client(self):
        client = MagicMock()
        client.evaluate_event = AsyncMock(
            return_value=GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        client.close = AsyncMock()
        with patch("openbox.client.GovernanceClient", return_value=client):
            result = await send_governance_event_compat(
                {
                    "api_url": "https://core.invalid",
                    "api_key": "obx_test_key",
                    "payload": {"event_type": "WorkflowStarted"},
                }
            )
        assert result is not None
        assert result["verdict"] == "allow"
        client.close.assert_awaited_once_with()
