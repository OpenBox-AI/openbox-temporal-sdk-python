# tests/test_activities.py
"""
Comprehensive pytest tests for the OpenBox SDK activities module.

Tests cover:
- _rfc3339_now() timestamp formatting
- GovernanceAPIError exception
- raise_governance_stop() behavior
- send_governance_event() activity with various scenarios
"""

import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

import httpx
from temporalio.exceptions import ApplicationError

from openbox.activities import (
    _rfc3339_now,
    GovernanceAPIError,
    raise_governance_stop,
    send_governance_event,
)


# =============================================================================
# Tests for _rfc3339_now()
# =============================================================================

class TestRfc3339Now:
    """Tests for the _rfc3339_now() function."""

    def test_returns_string(self):
        """Test that _rfc3339_now returns a string."""
        result = _rfc3339_now()
        assert isinstance(result, str)

    def test_ends_with_z(self):
        """Test that the timestamp ends with 'Z' (UTC indicator)."""
        result = _rfc3339_now()
        assert result.endswith('Z')

    def test_format_matches_rfc3339(self):
        """Test that the format matches YYYY-MM-DDTHH:MM:SS.sssZ."""
        result = _rfc3339_now()
        # RFC3339 pattern: 2024-01-15T10:30:45.123Z
        pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'
        assert re.match(pattern, result), f"Timestamp '{result}' does not match RFC3339 format"

    def test_timestamp_is_valid_datetime(self):
        """Test that the timestamp can be parsed back to a valid datetime."""
        result = _rfc3339_now()
        # Remove trailing Z and parse
        dt_str = result[:-1]  # Remove 'Z'
        dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
        assert isinstance(dt, datetime)

    def test_timestamp_is_recent(self):
        """Test that the timestamp is approximately the current time."""
        before = datetime.now(timezone.utc)
        result = _rfc3339_now()
        after = datetime.now(timezone.utc)

        # Parse the result
        dt_str = result[:-1]  # Remove 'Z'
        dt = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S.%f')
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
        match = re.search(r'\.(\d+)Z$', result)
        assert match is not None
        fractional = match.group(1)
        assert len(fractional) == 3, f"Expected 3 decimal places, got {len(fractional)}"


# =============================================================================
# Tests for GovernanceAPIError
# =============================================================================

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


# =============================================================================
# Tests for raise_governance_stop()
# =============================================================================

class TestRaiseGovernanceStop:
    """Tests for the raise_governance_stop() function."""

    def test_raises_application_error(self):
        """Test that raise_governance_stop raises ApplicationError."""
        with pytest.raises(ApplicationError):
            raise_governance_stop("Test reason")

    def test_error_message_format(self):
        """Test that the error message has the correct format."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Policy violation detected")
        assert "Governance blocked: Policy violation detected" in str(exc_info.value)

    def test_error_type_is_governance_stop(self):
        """Test that the error type is 'GovernanceStop'."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason")
        assert exc_info.value.type == "GovernanceStop"

    def test_non_retryable_is_true(self):
        """Test that non_retryable is set to True."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason")
        assert exc_info.value.non_retryable is True

    def test_includes_policy_id_in_details(self):
        """Test that policy_id is included in error details."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason", policy_id="policy-123")
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["policy_id"] == "policy-123"

    def test_includes_risk_score_in_details(self):
        """Test that risk_score is included in error details."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason", risk_score=0.85)
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["risk_score"] == 0.85

    def test_includes_both_policy_id_and_risk_score(self):
        """Test that both policy_id and risk_score are included in details."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason", policy_id="policy-456", risk_score=0.95)
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["policy_id"] == "policy-456"
        assert details[0]["risk_score"] == 0.95

    def test_default_values_are_none(self):
        """Test that default values for policy_id and risk_score are None."""
        with pytest.raises(ApplicationError) as exc_info:
            raise_governance_stop("Test reason")
        details = exc_info.value.details
        assert len(details) == 1
        assert details[0]["policy_id"] is None
        assert details[0]["risk_score"] is None


# =============================================================================
# Tests for send_governance_event() activity
# =============================================================================

class TestSendGovernanceEvent:
    """Tests for the send_governance_event() activity function."""

    @pytest.fixture
    def base_input(self):
        """Provide a base input dict for tests."""
        return {
            "api_url": "https://api.openbox.ai",
            "api_key": "test-api-key",
            "payload": {
                "event_type": "WorkflowStarted",
                "workflow_id": "test-workflow-123",
            },
            "timeout": 30.0,
            "on_api_error": "fail_open",
        }

    @pytest.fixture
    def mock_response_allow(self):
        """Create a mock response for 'allow' verdict."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "verdict": "allow",
            "reason": "Policy passed",
            "policy_id": "policy-001",
            "risk_score": 0.1,
        }
        return response

    @pytest.fixture
    def mock_response_block(self):
        """Create a mock response for 'block' verdict."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "verdict": "block",
            "reason": "High risk detected",
            "policy_id": "policy-002",
            "risk_score": 0.9,
        }
        return response

    @pytest.fixture
    def mock_response_v1_continue(self):
        """Create a mock response for v1.0 'continue' action."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "action": "continue",
            "reason": "Allowed by policy",
            "policy_id": "policy-v1",
            "risk_score": 0.2,
        }
        return response

    @pytest.fixture
    def mock_response_v1_stop(self):
        """Create a mock response for v1.0 'stop' action."""
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "action": "stop",
            "reason": "Blocked by v1 policy",
            "policy_id": "policy-v1-stop",
            "risk_score": 0.95,
        }
        return response

    # -------------------------------------------------------------------------
    # Successful API call tests
    # -------------------------------------------------------------------------

    async def test_successful_api_call_returns_result_with_verdict(
        self, base_input, mock_response_allow
    ):
        """Test that a successful API call returns a result with verdict."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "allow"
            assert result["reason"] == "Policy passed"
            assert result["policy_id"] == "policy-001"
            assert result["risk_score"] == 0.1

    async def test_adds_timestamp_to_payload(self, base_input, mock_response_allow):
        """Test that the activity adds a timestamp to the payload."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await send_governance_event(base_input)

            # Get the actual call arguments
            call_args = mock_client.post.call_args
            sent_payload = call_args.kwargs["json"]

            # Verify timestamp was added
            assert "timestamp" in sent_payload
            # Verify timestamp format
            pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$'
            assert re.match(pattern, sent_payload["timestamp"])

    async def test_preserves_original_payload_fields(self, base_input, mock_response_allow):
        """Test that original payload fields are preserved."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await send_governance_event(base_input)

            call_args = mock_client.post.call_args
            sent_payload = call_args.kwargs["json"]

            assert sent_payload["event_type"] == "WorkflowStarted"
            assert sent_payload["workflow_id"] == "test-workflow-123"

    # -------------------------------------------------------------------------
    # v1.0 compatibility tests
    # -------------------------------------------------------------------------

    async def test_v1_action_continue_maps_to_verdict_allow(
        self, base_input, mock_response_v1_continue
    ):
        """Test that v1.0 action='continue' maps to verdict='allow'."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_v1_continue
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "allow"
            assert result["action"] == "allow"  # backward compat field

    async def test_v1_action_stop_raises_governance_stop(
        self, base_input, mock_response_v1_stop
    ):
        """Test that v1.0 action='stop' raises GovernanceStop (non-SignalReceived)."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_v1_stop
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ApplicationError) as exc_info:
                await send_governance_event(base_input)

            assert exc_info.value.type == "GovernanceStop"
            assert exc_info.value.non_retryable is True
            assert "Blocked by v1 policy" in str(exc_info.value)

    # -------------------------------------------------------------------------
    # SignalReceived special handling tests
    # -------------------------------------------------------------------------

    async def test_signal_received_with_stop_returns_result_instead_of_raising(
        self, base_input, mock_response_block
    ):
        """Test that SignalReceived with action='stop' returns result instead of raising."""
        # Modify input to be a SignalReceived event
        base_input["payload"]["event_type"] = "SignalReceived"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_block
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Should NOT raise, should return result
            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "block"
            assert result["reason"] == "High risk detected"
            assert result["policy_id"] == "policy-002"
            assert result["risk_score"] == 0.9

    async def test_signal_received_with_allow_returns_normally(
        self, base_input, mock_response_allow
    ):
        """Test that SignalReceived with 'allow' verdict returns normally."""
        base_input["payload"]["event_type"] = "SignalReceived"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "allow"

    # -------------------------------------------------------------------------
    # HTTP error tests
    # -------------------------------------------------------------------------

    async def test_http_error_with_fail_open_returns_error_dict(self, base_input):
        """Test that HTTP error with fail_open returns error dict."""
        base_input["on_api_error"] = "fail_open"

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is False
            assert "error" in result
            assert "500" in result["error"]
            assert "Internal Server Error" in result["error"]

    async def test_http_error_with_fail_closed_raises_governance_api_error(self, base_input):
        """Test that HTTP error with fail_closed raises GovernanceAPIError."""
        base_input["on_api_error"] = "fail_closed"

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GovernanceAPIError) as exc_info:
                await send_governance_event(base_input)

            assert "503" in str(exc_info.value)
            assert "Service Unavailable" in str(exc_info.value)

    async def test_http_404_with_fail_open(self, base_input):
        """Test HTTP 404 error with fail_open returns error dict."""
        base_input["on_api_error"] = "fail_open"

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is False
            assert "404" in result["error"]

    # -------------------------------------------------------------------------
    # Network error tests
    # -------------------------------------------------------------------------

    async def test_network_error_with_fail_open_returns_error_dict(self, base_input):
        """Test that network error with fail_open returns error dict."""
        base_input["on_api_error"] = "fail_open"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is False
            assert "error" in result
            assert "Connection refused" in result["error"]

    async def test_network_error_with_fail_closed_raises_governance_api_error(self, base_input):
        """Test that network error with fail_closed raises GovernanceAPIError."""
        base_input["on_api_error"] = "fail_closed"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("Connection refused")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GovernanceAPIError) as exc_info:
                await send_governance_event(base_input)

            assert "Connection refused" in str(exc_info.value)

    async def test_timeout_error_with_fail_open(self, base_input):
        """Test timeout error with fail_open returns error dict."""
        base_input["on_api_error"] = "fail_open"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is False
            assert "error" in result

    async def test_timeout_error_with_fail_closed(self, base_input):
        """Test timeout error with fail_closed raises GovernanceAPIError."""
        base_input["on_api_error"] = "fail_closed"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.TimeoutException("Request timed out")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GovernanceAPIError):
                await send_governance_event(base_input)

    # -------------------------------------------------------------------------
    # API URL and headers tests
    # -------------------------------------------------------------------------

    async def test_correct_api_endpoint_is_called(self, base_input, mock_response_allow):
        """Test that the correct API endpoint is called."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await send_governance_event(base_input)

            call_args = mock_client.post.call_args
            called_url = call_args.args[0]
            assert called_url == "https://api.openbox.ai/api/v1/governance/evaluate"

    async def test_authorization_header_is_set(self, base_input, mock_response_allow):
        """Test that the Authorization header is correctly set."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await send_governance_event(base_input)

            call_args = mock_client.post.call_args
            headers = call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-api-key"
            assert headers["Content-Type"] == "application/json"

    # -------------------------------------------------------------------------
    # Verdict types tests
    # -------------------------------------------------------------------------

    async def test_block_verdict_raises_governance_stop(self, base_input, mock_response_block):
        """Test that 'block' verdict raises GovernanceStop for non-SignalReceived."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_block
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ApplicationError) as exc_info:
                await send_governance_event(base_input)

            assert exc_info.value.type == "GovernanceStop"

    async def test_halt_verdict_raises_governance_stop(self, base_input):
        """Test that 'halt' verdict raises GovernanceStop."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "verdict": "halt",
            "reason": "Emergency halt",
            "policy_id": "emergency-policy",
            "risk_score": 1.0,
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ApplicationError) as exc_info:
                await send_governance_event(base_input)

            assert exc_info.value.type == "GovernanceStop"
            assert "Emergency halt" in str(exc_info.value)

    async def test_constrain_verdict_returns_result(self, base_input):
        """Test that 'constrain' verdict returns result (does not raise)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "verdict": "constrain",
            "reason": "Apply constraints",
            "policy_id": "constrain-policy",
            "risk_score": 0.5,
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "constrain"

    async def test_require_approval_verdict_returns_result(self, base_input):
        """Test that 'require_approval' verdict returns result (does not raise)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "verdict": "require_approval",
            "reason": "Human approval required",
            "policy_id": "approval-policy",
            "risk_score": 0.7,
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            assert result["success"] is True
            assert result["verdict"] == "require_approval"

    # -------------------------------------------------------------------------
    # Default values tests
    # -------------------------------------------------------------------------

    async def test_default_on_api_error_is_fail_open(self, mock_response_allow):
        """Test that default on_api_error is 'fail_open'."""
        minimal_input = {
            "api_url": "https://api.openbox.ai",
            "api_key": "key",
            "payload": {"event_type": "WorkflowStarted"},
        }

        mock_error_response = MagicMock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Error"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_error_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Should not raise (fail_open is default)
            result = await send_governance_event(minimal_input)
            assert result["success"] is False

    async def test_default_timeout(self, mock_response_allow):
        """Test that default timeout is 30.0 seconds."""
        minimal_input = {
            "api_url": "https://api.openbox.ai",
            "api_key": "key",
            "payload": {"event_type": "WorkflowStarted"},
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            await send_governance_event(minimal_input)

            # Check that AsyncClient was initialized with timeout=30.0
            mock_client_class.assert_called_once_with(timeout=30.0)

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    async def test_empty_payload(self, mock_response_allow):
        """Test handling of empty payload."""
        input_data = {
            "api_url": "https://api.openbox.ai",
            "api_key": "key",
            "payload": {},
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(input_data)

            assert result["success"] is True
            # Timestamp should still be added
            call_args = mock_client.post.call_args
            sent_payload = call_args.kwargs["json"]
            assert "timestamp" in sent_payload

    async def test_missing_reason_in_response_uses_default(self, base_input):
        """Test that missing reason in stop response uses 'No reason provided'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "verdict": "block",
            "policy_id": "policy-no-reason",
            "risk_score": 0.9,
            # No "reason" field
        }

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ApplicationError) as exc_info:
                await send_governance_event(base_input)

            assert "No reason provided" in str(exc_info.value)

    async def test_application_error_is_reraised(self, base_input):
        """Test that ApplicationError is re-raised without modification."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = ApplicationError(
                "Custom app error",
                type="CustomType",
                non_retryable=True,
            )
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(ApplicationError) as exc_info:
                await send_governance_event(base_input)

            assert exc_info.value.type == "CustomType"

    async def test_governance_api_error_is_reraised(self, base_input):
        """Test that GovernanceAPIError is re-raised without modification."""
        base_input["on_api_error"] = "fail_closed"

        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.side_effect = GovernanceAPIError("API failed")
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(GovernanceAPIError) as exc_info:
                await send_governance_event(base_input)

            assert "API failed" in str(exc_info.value)

    async def test_backward_compat_action_field_in_result(
        self, base_input, mock_response_allow
    ):
        """Test that result includes 'action' field for backward compatibility."""
        with patch("openbox.activities.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response_allow
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await send_governance_event(base_input)

            # Both verdict and action should be present
            assert "verdict" in result
            assert "action" in result
            assert result["verdict"] == result["action"]
