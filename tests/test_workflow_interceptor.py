# tests/test_workflow_interceptor.py
"""
Comprehensive pytest tests for the OpenBox SDK workflow_interceptor module.

Tests cover:
1. _serialize_value() function
2. GovernanceHaltError exception
3. _send_governance_event() helper
4. GovernanceInterceptor class
5. _Inbound interceptor class (inner class)
"""

import base64
import json
import pytest
from dataclasses import dataclass, asdict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from openbox.workflow_interceptor import (
    _serialize_value,
    GovernanceHaltError,
    _send_governance_event,
    GovernanceInterceptor,
)
from openbox.types import Verdict
from openbox.config import GovernanceConfig


# =============================================================================
# Tests for _serialize_value()
# =============================================================================


class TestSerializeValue:
    """Tests for the _serialize_value() function."""

    def test_none_returns_none(self):
        """Test that None returns None."""
        assert _serialize_value(None) is None

    def test_string_passes_through(self):
        """Test that string primitives pass through unchanged."""
        assert _serialize_value("hello") == "hello"
        assert _serialize_value("") == ""
        assert _serialize_value("unicode: \u4e2d\u6587") == "unicode: \u4e2d\u6587"

    def test_int_passes_through(self):
        """Test that int primitives pass through unchanged."""
        assert _serialize_value(42) == 42
        assert _serialize_value(0) == 0
        assert _serialize_value(-123) == -123

    def test_float_passes_through(self):
        """Test that float primitives pass through unchanged."""
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(0.0) == 0.0
        assert _serialize_value(-2.5) == -2.5

    def test_bool_passes_through(self):
        """Test that bool primitives pass through unchanged."""
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False

    def test_bytes_decode_utf8(self):
        """Test that bytes decode to UTF-8 string."""
        result = _serialize_value(b"hello world")
        assert result == "hello world"

    def test_bytes_decode_utf8_unicode(self):
        """Test that bytes with unicode decode correctly."""
        result = _serialize_value("unicode: \u4e2d\u6587".encode('utf-8'))
        assert result == "unicode: \u4e2d\u6587"

    def test_bytes_fallback_to_base64(self):
        """Test that bytes fallback to base64 when not valid UTF-8."""
        # Invalid UTF-8 byte sequence
        invalid_bytes = b"\xff\xfe\x00\x01"
        result = _serialize_value(invalid_bytes)
        # Should be base64 encoded
        expected = base64.b64encode(invalid_bytes).decode('ascii')
        assert result == expected

    def test_dataclass_converts_to_dict(self):
        """Test that dataclass converts to dict via asdict()."""
        @dataclass
        class SampleData:
            name: str
            value: int

        data = SampleData(name="test", value=42)
        result = _serialize_value(data)
        assert result == {"name": "test", "value": 42}

    def test_dataclass_nested(self):
        """Test that nested dataclass converts correctly."""
        @dataclass
        class Inner:
            x: int

        @dataclass
        class Outer:
            inner: Inner
            label: str

        data = Outer(inner=Inner(x=10), label="outer")
        result = _serialize_value(data)
        assert result == {"inner": {"x": 10}, "label": "outer"}

    def test_dataclass_type_not_instance(self):
        """Test that dataclass type (not instance) is handled differently."""
        @dataclass
        class SampleData:
            name: str

        # The type itself (not an instance) should go through the fallback path
        result = _serialize_value(SampleData)
        # Should be stringified
        assert isinstance(result, str)
        assert "SampleData" in result

    def test_list_recursively_serializes(self):
        """Test that list recursively serializes its elements."""
        result = _serialize_value([1, "hello", 3.14, None])
        assert result == [1, "hello", 3.14, None]

    def test_list_with_bytes(self):
        """Test that list with bytes elements serializes correctly."""
        result = _serialize_value([b"hello", b"world"])
        assert result == ["hello", "world"]

    def test_list_nested(self):
        """Test that nested lists serialize correctly."""
        result = _serialize_value([[1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_tuple_recursively_serializes(self):
        """Test that tuple recursively serializes its elements."""
        result = _serialize_value((1, "hello", 3.14))
        assert result == [1, "hello", 3.14]

    def test_dict_recursively_serializes(self):
        """Test that dict recursively serializes its values."""
        result = _serialize_value({"key": "value", "num": 42})
        assert result == {"key": "value", "num": 42}

    def test_dict_with_bytes_value(self):
        """Test that dict with bytes values serializes correctly."""
        result = _serialize_value({"data": b"binary"})
        assert result == {"data": "binary"}

    def test_dict_nested(self):
        """Test that nested dicts serialize correctly."""
        result = _serialize_value({"outer": {"inner": "value"}})
        assert result == {"outer": {"inner": "value"}}

    def test_dict_with_dataclass_value(self):
        """Test that dict with dataclass values serializes correctly."""
        @dataclass
        class Item:
            id: int

        result = _serialize_value({"item": Item(id=123)})
        assert result == {"item": {"id": 123}}

    def test_other_object_fallback_to_str(self):
        """Test that other objects fallback to str()."""
        class CustomObject:
            def __str__(self):
                return "CustomObject<test>"

        result = _serialize_value(CustomObject())
        assert result == "CustomObject<test>"

    def test_complex_nested_structure(self):
        """Test serialization of complex nested structures."""
        @dataclass
        class Result:
            status: str
            data: dict

        value = {
            "results": [
                Result(status="ok", data={"key": "value"}),
                Result(status="error", data={"error": "message"}),
            ],
            "metadata": {
                "binary": b"test",
                "nested": {"level": 2},
            },
        }
        result = _serialize_value(value)
        assert result == {
            "results": [
                {"status": "ok", "data": {"key": "value"}},
                {"status": "error", "data": {"error": "message"}},
            ],
            "metadata": {
                "binary": "test",
                "nested": {"level": 2},
            },
        }

    def test_json_serializable_object_via_json_dumps(self):
        """Test objects that are JSON serializable via json.dumps default=str."""
        from datetime import datetime

        # datetime is not directly JSON serializable but json.dumps with default=str handles it
        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _serialize_value(dt)
        # Should be the string representation
        assert isinstance(result, str)


# =============================================================================
# Tests for GovernanceHaltError
# =============================================================================


class TestGovernanceHaltError:
    """Tests for the GovernanceHaltError exception class."""

    def test_can_be_raised_with_message(self):
        """Test that GovernanceHaltError can be raised with a message."""
        with pytest.raises(GovernanceHaltError) as exc_info:
            raise GovernanceHaltError("Governance blocked execution")
        assert str(exc_info.value) == "Governance blocked execution"

    def test_inherits_from_exception(self):
        """Test that GovernanceHaltError inherits from Exception."""
        assert issubclass(GovernanceHaltError, Exception)

    def test_can_be_caught_as_exception(self):
        """Test that GovernanceHaltError can be caught as Exception."""
        try:
            raise GovernanceHaltError("test error")
        except Exception as e:
            assert isinstance(e, GovernanceHaltError)
            assert str(e) == "test error"

    def test_empty_message(self):
        """Test GovernanceHaltError with empty message."""
        with pytest.raises(GovernanceHaltError) as exc_info:
            raise GovernanceHaltError("")
        assert str(exc_info.value) == ""

    def test_message_with_special_characters(self):
        """Test GovernanceHaltError with special characters in message."""
        msg = "Error: policy 'test-policy' blocked\nDetails: high risk"
        with pytest.raises(GovernanceHaltError) as exc_info:
            raise GovernanceHaltError(msg)
        assert str(exc_info.value) == msg


# =============================================================================
# Tests for _send_governance_event()
# =============================================================================


class TestSendGovernanceEvent:
    """Tests for the _send_governance_event() helper function."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow module."""
        with patch("openbox.workflow_interceptor.workflow") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_calls_execute_activity_with_correct_args(self, mock_workflow):
        """Test that _send_governance_event calls workflow.execute_activity with correct args."""
        mock_workflow.execute_activity = AsyncMock(return_value={"verdict": "allow"})

        result = await _send_governance_event(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            payload={"event_type": "WorkflowStarted"},
            timeout=30.0,
            on_api_error="fail_open",
        )

        mock_workflow.execute_activity.assert_called_once()
        call_args = mock_workflow.execute_activity.call_args

        # Check activity name
        assert call_args.args[0] == "send_governance_event"

        # Check args parameter contains expected data
        activity_input = call_args.kwargs["args"][0]
        assert activity_input["api_url"] == "https://api.openbox.ai"
        assert activity_input["api_key"] == "test-key"
        assert activity_input["payload"] == {"event_type": "WorkflowStarted"}
        assert activity_input["timeout"] == 30.0
        assert activity_input["on_api_error"] == "fail_open"

        assert result == {"verdict": "allow"}

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self, mock_workflow):
        """Test that _send_governance_event returns result on success."""
        expected_result = {
            "verdict": "allow",
            "reason": "Policy passed",
            "policy_id": "policy-001",
        }
        mock_workflow.execute_activity = AsyncMock(return_value=expected_result)

        result = await _send_governance_event(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            payload={},
            timeout=30.0,
        )

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_application_error(self, mock_workflow):
        """Test that ApplicationError raises GovernanceHaltError."""
        # Create a custom ApplicationError class to simulate the real one
        class ApplicationError(Exception):
            pass

        error = ApplicationError("Governance blocked: Policy violation")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                api_url="https://api.openbox.ai",
                api_key="test-key",
                payload={},
                timeout=30.0,
            )

        assert "Governance blocked:" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_governance_blocked_message(self, mock_workflow):
        """Test that error with 'Governance blocked:' message raises GovernanceHaltError."""
        error = Exception("Governance blocked: High risk detected")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                api_url="https://api.openbox.ai",
                api_key="test-key",
                payload={},
                timeout=30.0,
            )

        assert "Governance blocked: High risk detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_governance_api_error(self, mock_workflow):
        """Test that GovernanceAPIError raises GovernanceHaltError."""
        error = Exception("GovernanceAPIError: API unreachable")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                api_url="https://api.openbox.ai",
                api_key="test-key",
                payload={},
                timeout=30.0,
            )

        assert "GovernanceAPIError" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_governance_api_error_type(self, mock_workflow):
        """Test that exception with GovernanceAPIError in type name raises GovernanceHaltError."""
        # Create a custom GovernanceAPIError class to simulate the real one
        class GovernanceAPIError(Exception):
            pass

        error = GovernanceAPIError("API failed")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                api_url="https://api.openbox.ai",
                api_key="test-key",
                payload={},
                timeout=30.0,
            )

        assert "API failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_returns_none_for_other_errors_with_fail_open(self, mock_workflow):
        """Test that other errors with fail_open return None."""
        error = RuntimeError("Network timeout")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            payload={},
            timeout=30.0,
            on_api_error="fail_open",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_default_on_api_error_is_fail_open(self, mock_workflow):
        """Test that default on_api_error is fail_open."""
        error = RuntimeError("Some error")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            payload={},
            timeout=30.0,
            # on_api_error not specified, should default to fail_open
        )

        assert result is None  # fail_open returns None on non-governance errors

    @pytest.mark.asyncio
    async def test_timeout_is_passed_correctly(self, mock_workflow):
        """Test that start_to_close_timeout is timeout + 5 seconds."""
        from datetime import timedelta

        mock_workflow.execute_activity = AsyncMock(return_value={})

        await _send_governance_event(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            payload={},
            timeout=25.0,
        )

        call_args = mock_workflow.execute_activity.call_args
        expected_timeout = timedelta(seconds=30.0)  # 25 + 5
        assert call_args.kwargs["start_to_close_timeout"] == expected_timeout


# =============================================================================
# Tests for GovernanceInterceptor
# =============================================================================


class TestGovernanceInterceptor:
    """Tests for the GovernanceInterceptor class."""

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        mock_span_processor = MagicMock()
        config = GovernanceConfig(
            api_timeout=60.0,
            on_api_error="fail_closed",
            send_start_event=False,
            skip_workflow_types={"SkipWorkflow"},
            skip_signals={"skip_signal"},
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-api-key",
            span_processor=mock_span_processor,
            config=config,
        )

        assert interceptor.api_url == "https://api.openbox.ai"
        assert interceptor.api_key == "test-api-key"
        assert interceptor.span_processor is mock_span_processor
        assert interceptor.api_timeout == 60.0
        assert interceptor.on_api_error == "fail_closed"
        assert interceptor.send_start_event is False
        assert interceptor.skip_workflow_types == {"SkipWorkflow"}
        assert interceptor.skip_signals == {"skip_signal"}

    def test_api_url_trailing_slash_is_stripped(self):
        """Test that trailing slash is stripped from api_url."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai/",
            api_key="test-key",
        )
        assert interceptor.api_url == "https://api.openbox.ai"

        interceptor2 = GovernanceInterceptor(
            api_url="https://api.openbox.ai///",
            api_key="test-key",
        )
        assert interceptor2.api_url == "https://api.openbox.ai"

    def test_default_values_without_config(self):
        """Test default values when config is None."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=None,
            config=None,
        )

        assert interceptor.api_timeout == 30.0
        assert interceptor.on_api_error == "fail_open"
        assert interceptor.send_start_event is True
        assert interceptor.skip_workflow_types == set()
        assert interceptor.skip_signals == set()
        assert interceptor.span_processor is None

    def test_default_values_from_config(self):
        """Test default values are read from config."""
        config = GovernanceConfig()  # Use default config values

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        assert interceptor.api_timeout == 30.0  # Default from GovernanceConfig
        assert interceptor.on_api_error == "fail_open"
        assert interceptor.send_start_event is True
        assert interceptor.skip_workflow_types == set()
        assert interceptor.skip_signals == set()

    def test_workflow_interceptor_class_returns_interceptor_class(self):
        """Test that workflow_interceptor_class() returns the _Inbound class."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
        )

        # Create mock input
        mock_input = MagicMock()
        result = interceptor.workflow_interceptor_class(mock_input)

        # Should return a class
        assert result is not None
        assert isinstance(result, type)
        # The class should be named _Inbound
        assert result.__name__ == "_Inbound"


# =============================================================================
# Tests for _Inbound interceptor class (inner class)
# =============================================================================


class TestInboundInterceptor:
    """Tests for the _Inbound interceptor class."""

    @pytest.fixture
    def mock_workflow_info(self):
        """Create a mock workflow info."""
        info = MagicMock()
        info.workflow_id = "wf-123"
        info.run_id = "run-456"
        info.workflow_type = "TestWorkflow"
        info.task_queue = "test-queue"
        return info

    @pytest.fixture
    def mock_workflow_module(self, mock_workflow_info):
        """Create a mock workflow module with patched methods."""
        with patch("openbox.workflow_interceptor.workflow") as mock:
            mock.info.return_value = mock_workflow_info
            mock.patched.return_value = True
            mock.execute_activity = AsyncMock(return_value={"verdict": "allow"})
            yield mock

    @pytest.fixture
    def governance_interceptor(self):
        """Create a GovernanceInterceptor for testing."""
        return GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=MagicMock(),
            config=GovernanceConfig(),
        )

    @pytest.fixture
    def inbound_class(self, governance_interceptor):
        """Get the _Inbound class from the interceptor."""
        mock_input = MagicMock()
        return governance_interceptor.workflow_interceptor_class(mock_input)

    @pytest.fixture
    def inbound_instance(self, inbound_class):
        """Create an instance of the _Inbound class."""
        mock_next_interceptor = MagicMock()
        return inbound_class(mock_next_interceptor)

    # -------------------------------------------------------------------------
    # execute_workflow() tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_execute_workflow_skips_if_workflow_type_in_skip_list(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test execute_workflow skips governance if workflow_type is in skip list."""
        mock_workflow_info.workflow_type = "SkipThisWorkflow"

        config = GovernanceConfig(skip_workflow_types={"SkipThisWorkflow"})
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(return_value="workflow_result")
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()
        result = await inbound.execute_workflow(execute_input)

        # Should call super().execute_workflow without sending events
        mock_next.execute_workflow.assert_called_once_with(execute_input)
        # execute_activity should NOT be called (no governance events)
        mock_workflow_module.execute_activity.assert_not_called()
        assert result == "workflow_result"

    @pytest.mark.asyncio
    async def test_execute_workflow_sends_started_event_if_send_start_event_true(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test execute_workflow sends WorkflowStarted event if send_start_event=True."""
        config = GovernanceConfig(send_start_event=True)
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(return_value="result")
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()
        await inbound.execute_workflow(execute_input)

        # Check that execute_activity was called with WorkflowStarted event
        calls = mock_workflow_module.execute_activity.call_args_list
        assert len(calls) >= 1  # At least WorkflowStarted

        # Find the WorkflowStarted call
        started_calls = [
            c for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowStarted"
        ]
        assert len(started_calls) == 1
        started_payload = started_calls[0].kwargs["args"][0]["payload"]
        assert started_payload["workflow_id"] == "wf-123"
        assert started_payload["run_id"] == "run-456"
        assert started_payload["workflow_type"] == "TestWorkflow"

    @pytest.mark.asyncio
    async def test_execute_workflow_does_not_send_started_event_if_send_start_event_false(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test execute_workflow does not send WorkflowStarted event if send_start_event=False."""
        config = GovernanceConfig(send_start_event=False)
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(return_value="result")
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()
        await inbound.execute_workflow(execute_input)

        # Check that no WorkflowStarted event was sent
        calls = mock_workflow_module.execute_activity.call_args_list
        started_calls = [
            c for c in calls
            if c.kwargs.get("args", [[{}]])[0].get("payload", {}).get("event_type") == "WorkflowStarted"
        ]
        assert len(started_calls) == 0

    @pytest.mark.asyncio
    async def test_execute_workflow_sends_completed_event_on_success(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test execute_workflow sends WorkflowCompleted event on success."""
        @dataclass
        class WorkflowResult:
            status: str
            data: dict

        workflow_result = WorkflowResult(status="success", data={"key": "value"})

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=GovernanceConfig(send_start_event=False),  # Disable start to simplify test
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(return_value=workflow_result)
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()
        result = await inbound.execute_workflow(execute_input)

        assert result == workflow_result

        # Find the WorkflowCompleted call
        calls = mock_workflow_module.execute_activity.call_args_list
        completed_calls = [
            c for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowCompleted"
        ]
        assert len(completed_calls) == 1
        completed_payload = completed_calls[0].kwargs["args"][0]["payload"]
        assert completed_payload["workflow_id"] == "wf-123"
        assert completed_payload["workflow_type"] == "TestWorkflow"
        # Check serialized output
        assert completed_payload["workflow_output"] == {"status": "success", "data": {"key": "value"}}

    @pytest.mark.asyncio
    async def test_execute_workflow_sends_failed_event_on_failure(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test execute_workflow sends WorkflowFailed event on failure."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=GovernanceConfig(send_start_event=False),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(side_effect=ValueError("Something went wrong"))
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()

        with pytest.raises(ValueError):
            await inbound.execute_workflow(execute_input)

        # Find the WorkflowFailed call
        calls = mock_workflow_module.execute_activity.call_args_list
        failed_calls = [
            c for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowFailed"
        ]
        assert len(failed_calls) == 1
        failed_payload = failed_calls[0].kwargs["args"][0]["payload"]
        assert failed_payload["workflow_id"] == "wf-123"
        assert failed_payload["workflow_type"] == "TestWorkflow"
        assert failed_payload["error"]["type"] == "ValueError"
        assert "Something went wrong" in failed_payload["error"]["message"]

    @pytest.mark.asyncio
    async def test_execute_workflow_error_includes_cause_chain(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test that error includes cause chain for ActivityError."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=GovernanceConfig(send_start_event=False),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        # Create an exception with a cause (simulating ActivityError)
        root_cause = Exception("Root cause error")
        middle_cause = Exception("Middle cause")
        middle_cause.__cause__ = root_cause
        top_error = Exception("Activity failed")
        top_error.cause = middle_cause  # Temporal uses .cause property

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(side_effect=top_error)
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()

        with pytest.raises(Exception):
            await inbound.execute_workflow(execute_input)

        # Find the WorkflowFailed call
        calls = mock_workflow_module.execute_activity.call_args_list
        failed_calls = [
            c for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowFailed"
        ]
        assert len(failed_calls) == 1
        error_info = failed_calls[0].kwargs["args"][0]["payload"]["error"]

        # Check cause chain
        assert error_info["type"] == "Exception"
        assert "Activity failed" in error_info["message"]
        assert "cause" in error_info
        assert error_info["cause"]["type"] == "Exception"
        assert "Middle cause" in error_info["cause"]["message"]
        # Check root cause
        assert "root_cause" in error_info
        assert error_info["root_cause"]["type"] == "Exception"
        assert "Root cause error" in error_info["root_cause"]["message"]

    @pytest.mark.asyncio
    async def test_execute_workflow_error_includes_application_error_details(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test that ApplicationError details are included in error info."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=GovernanceConfig(send_start_event=False),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        # Create an exception with ApplicationError-like cause
        cause = MagicMock()
        cause.__class__.__name__ = "ApplicationError"
        cause.type = "GovernanceStop"
        cause.non_retryable = True
        cause.__str__ = MagicMock(return_value="Governance stopped")

        top_error = Exception("Activity error")
        top_error.cause = cause

        mock_next = AsyncMock()
        mock_next.execute_workflow = AsyncMock(side_effect=top_error)
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()

        with pytest.raises(Exception):
            await inbound.execute_workflow(execute_input)

        # Find the WorkflowFailed call
        calls = mock_workflow_module.execute_activity.call_args_list
        failed_calls = [
            c for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowFailed"
        ]
        error_info = failed_calls[0].kwargs["args"][0]["payload"]["error"]

        assert "cause" in error_info
        assert error_info["cause"]["error_type"] == "GovernanceStop"
        assert error_info["cause"]["non_retryable"] is True

    # -------------------------------------------------------------------------
    # handle_signal() tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_handle_signal_skips_if_signal_in_skip_list(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal skips if signal is in skip_signals list."""
        config = GovernanceConfig(skip_signals={"skip_this_signal"})
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "skip_this_signal"
        signal_input.args = ["arg1"]

        await inbound.handle_signal(signal_input)

        # Should call super().handle_signal
        mock_next.handle_signal.assert_called_once_with(signal_input)
        # execute_activity should NOT be called
        mock_workflow_module.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_signal_skips_if_workflow_type_in_skip_list(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal skips if workflow_type is in skip_workflow_types list."""
        mock_workflow_info.workflow_type = "SkipWorkflow"
        config = GovernanceConfig(skip_workflow_types={"SkipWorkflow"})
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=config,
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "test_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        mock_next.handle_signal.assert_called_once_with(signal_input)
        mock_workflow_module.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_signal_sends_signal_received_event(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal sends SignalReceived event."""
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "my_signal"
        signal_input.args = ["arg1", {"key": "value"}]

        await inbound.handle_signal(signal_input)

        # Check that SignalReceived event was sent
        calls = mock_workflow_module.execute_activity.call_args_list
        assert len(calls) == 1

        payload = calls[0].kwargs["args"][0]["payload"]
        assert payload["event_type"] == "SignalReceived"
        assert payload["workflow_id"] == "wf-123"
        assert payload["run_id"] == "run-456"
        assert payload["workflow_type"] == "TestWorkflow"
        assert payload["signal_name"] == "my_signal"
        assert payload["signal_args"] == ["arg1", {"key": "value"}]

        # Should also call next handler
        mock_next.handle_signal.assert_called_once_with(signal_input)

    @pytest.mark.asyncio
    async def test_handle_signal_stores_verdict_in_span_processor_if_block(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal stores verdict in span_processor if BLOCK verdict."""
        mock_span_processor = MagicMock()
        mock_workflow_module.execute_activity = AsyncMock(return_value={
            "verdict": "block",
            "reason": "High risk signal",
        })

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "risky_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        # Check that set_verdict was called on span_processor
        mock_span_processor.set_verdict.assert_called_once_with(
            "wf-123",
            Verdict.BLOCK,
            "High risk signal",
            "run-456",  # run_id
        )

    @pytest.mark.asyncio
    async def test_handle_signal_stores_verdict_in_span_processor_if_halt(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal stores verdict in span_processor if HALT verdict."""
        mock_span_processor = MagicMock()
        mock_workflow_module.execute_activity = AsyncMock(return_value={
            "verdict": "halt",
            "reason": "Critical alert",
        })

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "critical_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        mock_span_processor.set_verdict.assert_called_once_with(
            "wf-123",
            Verdict.HALT,
            "Critical alert",
            "run-456",
        )

    @pytest.mark.asyncio
    async def test_handle_signal_does_not_store_verdict_if_allow(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal does not store verdict if ALLOW verdict."""
        mock_span_processor = MagicMock()
        mock_workflow_module.execute_activity = AsyncMock(return_value={
            "verdict": "allow",
            "reason": "Signal approved",
        })

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "safe_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        # set_verdict should NOT be called for ALLOW
        mock_span_processor.set_verdict.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_signal_does_not_store_verdict_if_no_span_processor(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal does not fail if no span_processor."""
        mock_workflow_module.execute_activity = AsyncMock(return_value={
            "verdict": "block",
            "reason": "High risk",
        })

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=None,  # No span processor
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "test_signal"
        signal_input.args = []

        # Should not raise even though verdict is BLOCK and no span_processor
        await inbound.handle_signal(signal_input)

        mock_next.handle_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_signal_uses_action_field_for_v1_compat(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal uses action field if verdict not present (v1.0 compat)."""
        mock_span_processor = MagicMock()
        mock_workflow_module.execute_activity = AsyncMock(return_value={
            "action": "stop",  # v1.0 style
            "reason": "Blocked by v1 policy",
        })

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "test_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        # "stop" should map to HALT
        mock_span_processor.set_verdict.assert_called_once_with(
            "wf-123",
            Verdict.HALT,
            "Blocked by v1 policy",
            "run-456",
        )

    @pytest.mark.asyncio
    async def test_handle_signal_defaults_to_allow_if_no_result(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal defaults to ALLOW if result is None."""
        mock_span_processor = MagicMock()
        mock_workflow_module.execute_activity = AsyncMock(return_value=None)

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        mock_input = MagicMock()
        inbound_class = interceptor.workflow_interceptor_class(mock_input)

        mock_next = AsyncMock()
        mock_next.handle_signal = AsyncMock()
        inbound = inbound_class(mock_next)

        signal_input = MagicMock()
        signal_input.signal = "test_signal"
        signal_input.args = []

        await inbound.handle_signal(signal_input)

        # Should not store verdict since default is ALLOW
        mock_span_processor.set_verdict.assert_not_called()


# =============================================================================
# Integration-style tests (testing closures and captured variables)
# =============================================================================


class TestInterceptorClosures:
    """Tests to verify that closures capture variables correctly."""

    @pytest.mark.asyncio
    async def test_interceptor_captures_config_values(self):
        """Test that the inner _Inbound class captures config values via closures."""
        with patch("openbox.workflow_interceptor.workflow") as mock_workflow:
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-closure-test"
            mock_info.run_id = "run-closure"
            mock_info.workflow_type = "ClosureWorkflow"
            mock_info.task_queue = "closure-queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.return_value = True
            mock_workflow.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            config = GovernanceConfig(
                api_timeout=45.0,
                on_api_error="fail_closed",
            )
            interceptor = GovernanceInterceptor(
                api_url="https://custom.api.url",
                api_key="custom-api-key",
                config=config,
            )

            inbound_class = interceptor.workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.execute_workflow = AsyncMock(return_value="result")
            inbound = inbound_class(mock_next)

            execute_input = MagicMock()
            await inbound.execute_workflow(execute_input)

            # Verify the captured values were used
            calls = mock_workflow.execute_activity.call_args_list
            if calls:
                activity_input = calls[0].kwargs["args"][0]
                assert activity_input["api_url"] == "https://custom.api.url"
                assert activity_input["api_key"] == "custom-api-key"
                assert activity_input["timeout"] == 45.0
                assert activity_input["on_api_error"] == "fail_closed"

    @pytest.mark.asyncio
    async def test_multiple_interceptor_instances_are_independent(self):
        """Test that multiple interceptor instances don't share state."""
        interceptor1 = GovernanceInterceptor(
            api_url="https://api1.openbox.ai",
            api_key="key1",
            config=GovernanceConfig(skip_workflow_types={"Skip1"}),
        )
        interceptor2 = GovernanceInterceptor(
            api_url="https://api2.openbox.ai",
            api_key="key2",
            config=GovernanceConfig(skip_workflow_types={"Skip2"}),
        )

        # Verify they have different configurations
        assert interceptor1.api_url == "https://api1.openbox.ai"
        assert interceptor2.api_url == "https://api2.openbox.ai"
        assert interceptor1.skip_workflow_types == {"Skip1"}
        assert interceptor2.skip_workflow_types == {"Skip2"}


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_serialize_value_with_empty_list(self):
        """Test _serialize_value with empty list."""
        assert _serialize_value([]) == []

    def test_serialize_value_with_empty_dict(self):
        """Test _serialize_value with empty dict."""
        assert _serialize_value({}) == {}

    def test_serialize_value_with_deeply_nested_structure(self):
        """Test _serialize_value with deeply nested structure."""
        deep = {"level": 1}
        for i in range(2, 10):
            deep = {"level": i, "nested": deep}

        result = _serialize_value(deep)
        assert result["level"] == 9
        assert result["nested"]["level"] == 8

    @pytest.mark.asyncio
    async def test_execute_workflow_handles_none_result(self):
        """Test execute_workflow handles None result."""
        with patch("openbox.workflow_interceptor.workflow") as mock_workflow:
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-none"
            mock_info.run_id = "run-none"
            mock_info.workflow_type = "NoneWorkflow"
            mock_info.task_queue = "queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.return_value = True
            mock_workflow.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            interceptor = GovernanceInterceptor(
                api_url="https://api.openbox.ai",
                api_key="test-key",
                config=GovernanceConfig(send_start_event=False),
            )

            inbound_class = interceptor.workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.execute_workflow = AsyncMock(return_value=None)
            inbound = inbound_class(mock_next)

            result = await inbound.execute_workflow(MagicMock())
            assert result is None

            # Check WorkflowCompleted was sent with None output
            calls = mock_workflow.execute_activity.call_args_list
            completed_calls = [
                c for c in calls
                if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowCompleted"
            ]
            assert completed_calls[0].kwargs["args"][0]["payload"]["workflow_output"] is None

    @pytest.mark.asyncio
    async def test_handle_signal_with_empty_args(self):
        """Test handle_signal with empty args list."""
        with patch("openbox.workflow_interceptor.workflow") as mock_workflow:
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-signal"
            mock_info.run_id = "run-signal"
            mock_info.workflow_type = "SignalWorkflow"
            mock_info.task_queue = "queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.return_value = True
            mock_workflow.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            interceptor = GovernanceInterceptor(
                api_url="https://api.openbox.ai",
                api_key="test-key",
            )

            inbound_class = interceptor.workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.handle_signal = AsyncMock()
            inbound = inbound_class(mock_next)

            signal_input = MagicMock()
            signal_input.signal = "empty_args_signal"
            signal_input.args = []

            await inbound.handle_signal(signal_input)

            calls = mock_workflow.execute_activity.call_args_list
            payload = calls[0].kwargs["args"][0]["payload"]
            assert payload["signal_args"] == []

    def test_governance_interceptor_is_temporal_interceptor(self):
        """Test that GovernanceInterceptor is a Temporal Interceptor."""
        from temporalio.worker import Interceptor

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
        )
        assert isinstance(interceptor, Interceptor)
