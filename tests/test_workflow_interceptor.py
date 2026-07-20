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
from dataclasses import asdict, dataclass
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from openbox.config import GovernanceConfig
from openbox.governance_state import TemporalGovernanceState
from openbox.types import Verdict
from openbox.workflow_interceptor import (
    _RETRYABLE_BLOCK_PATCH,
    GovernanceHaltError,
    GovernanceInterceptor,
    _is_halt,
    _send_governance_event,
    _serialize_value,
)


def _legacy_patched(marker, *args, **kwargs):
    """Simulate an old history: every pre-existing ``openbox-v2-*`` marker is
    present, but the retryable-block marker is NOT — so execute_workflow takes the
    unchanged legacy path. Existing execute_workflow tests use this to assert the
    pre-feature behavior is preserved."""
    return marker != _RETRYABLE_BLOCK_PATCH


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
        result = _serialize_value("unicode: \u4e2d\u6587".encode("utf-8"))
        assert result == "unicode: \u4e2d\u6587"

    def test_bytes_fallback_to_base64(self):
        """Test that bytes fallback to base64 when not valid UTF-8."""
        # Invalid UTF-8 byte sequence
        invalid_bytes = b"\xff\xfe\x00\x01"
        result = _serialize_value(invalid_bytes)
        # Should be base64 encoded
        expected = base64.b64encode(invalid_bytes).decode("ascii")
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

    def test_pydantic_style_model_dump_converts_to_dict(self):
        """Pydantic v2 models retain their structure instead of using str()."""

        class RefundExecutionInput:
            def model_dump(self, *, mode):
                assert mode == "json"
                return {
                    "item": {"id": "game-001", "price": 300, "price_unit": "cent"},
                    "refund": {
                        "amount": 300,
                        "currency": "USD",
                        "amount_unit": "dollar",
                    },
                }

            def __str__(self):
                return "item=Item(...) refund=Refund(...)"

        result = _serialize_value([RefundExecutionInput()])

        assert result == [
            {
                "item": {"id": "game-001", "price": 300, "price_unit": "cent"},
                "refund": {
                    "amount": 300,
                    "currency": "USD",
                    "amount_unit": "dollar",
                },
            }
        ]

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


class TestSendGovernanceEvent:
    """Tests for the _send_governance_event() helper function."""

    @pytest.fixture
    def mock_workflow(self):
        """Create a mock workflow module."""
        with patch("openbox.workflow_interceptor.workflow") as mock:
            yield mock

    @pytest.mark.asyncio
    async def test_calls_execute_activity_with_correct_args(self, mock_workflow):
        """Activity input must carry payload/timeout/policy — never credentials."""
        mock_workflow.execute_activity = AsyncMock(return_value={"verdict": "allow"})

        result = await _send_governance_event(
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
        assert activity_input["payload"] == {"event_type": "WorkflowStarted"}
        assert activity_input["timeout"] == 30.0
        assert activity_input["on_api_error"] == "fail_open"
        # Credentials must NOT be in the activity input (would leak via workflow history)
        assert "api_url" not in activity_input
        assert "api_key" not in activity_input

        assert result == {"verdict": "allow"}

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self, mock_workflow):
        expected_result = {
            "verdict": "allow",
            "reason": "Policy passed",
            "policy_id": "policy-001",
        }
        mock_workflow.execute_activity = AsyncMock(return_value=expected_result)

        result = await _send_governance_event(
            payload={},
            timeout=30.0,
        )

        assert result == expected_result

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_application_error(
        self, mock_workflow
    ):
        """ApplicationError.type == 'GovernanceHalt' must raise GovernanceHaltError."""
        from temporalio.exceptions import ApplicationError

        error = ApplicationError(
            "Governance HALT: Policy violation", type="GovernanceHalt"
        )
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                payload={},
                timeout=30.0,
            )

        assert "Policy violation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_legacy_governance_stop_type_raises_halt(self, mock_workflow):
        """Legacy ApplicationError.type == 'GovernanceStop' still routes to halt."""
        from temporalio.exceptions import ApplicationError

        error = ApplicationError("legacy halt", type="GovernanceStop")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError):
            await _send_governance_event(payload={}, timeout=30.0)

    @pytest.mark.asyncio
    async def test_governance_block_returns_none(self, mock_workflow):
        """ApplicationError.type == 'GovernanceBlock' returns None (activity-level block)."""
        from temporalio.exceptions import ApplicationError

        error = ApplicationError("blocked", type="GovernanceBlock")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(payload={}, timeout=30.0)
        assert result is None

    @pytest.mark.asyncio
    async def test_raises_governance_halt_error_for_governance_api_error_type(
        self, mock_workflow
    ):
        """ApplicationError.type == 'GovernanceAPIError' (fail_closed+API down) raises halt."""
        from temporalio.exceptions import ApplicationError

        error = ApplicationError("API failed", type="GovernanceAPIError")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        with pytest.raises(GovernanceHaltError) as exc_info:
            await _send_governance_event(
                payload={},
                timeout=30.0,
            )

        assert "API failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_string_matching_on_message_is_ignored(self, mock_workflow):
        """Messages containing 'GovernanceHalt' must NOT trigger halt without proper type."""
        error = Exception("GovernanceHalt: user happens to mention this string")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(
            payload={},
            timeout=30.0,
            on_api_error="fail_open",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_other_errors_with_fail_open(self, mock_workflow):
        """Non-governance errors with fail_open return None."""
        error = RuntimeError("Network timeout")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(
            payload={},
            timeout=30.0,
            on_api_error="fail_open",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_default_on_api_error_is_fail_open(self, mock_workflow):
        """Default on_api_error is fail_open."""
        error = RuntimeError("Some error")
        mock_workflow.execute_activity = AsyncMock(side_effect=error)

        result = await _send_governance_event(
            payload={},
            timeout=30.0,
            # on_api_error not specified, should default to fail_open
        )

        assert result is None  # fail_open returns None on non-governance errors

    @pytest.mark.asyncio
    async def test_timeout_is_passed_correctly(self, mock_workflow):
        """start_to_close_timeout is timeout + 5 seconds."""
        from datetime import timedelta

        mock_workflow.execute_activity = AsyncMock(return_value={})

        await _send_governance_event(
            payload={},
            timeout=25.0,
        )

        call_args = mock_workflow.execute_activity.call_args
        expected_timeout = timedelta(seconds=30.0)  # 25 + 5
        assert call_args.kwargs["start_to_close_timeout"] == expected_timeout


class TestGovernanceInterceptor:
    """Tests for the GovernanceInterceptor class."""

    def test_initialization_with_all_parameters(self):
        """Test initialization with all parameters."""
        state = TemporalGovernanceState()
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
            state=state,
            config=config,
        )

        assert interceptor.api_url == "https://api.openbox.ai"
        assert interceptor.api_key == "test-api-key"
        assert interceptor.state is state
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
            state=None,
            config=None,
        )

        assert interceptor.api_timeout == 30.0
        assert interceptor.on_api_error == "fail_open"
        assert interceptor.send_start_event is True
        assert interceptor.skip_workflow_types == set()
        assert interceptor.skip_signals == set()
        assert interceptor.state is None

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
        with (
            patch("openbox.workflow_interceptor.workflow") as mock,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            mock.info.return_value = mock_workflow_info
            mock.patched.side_effect = _legacy_patched
            mock.execute_activity = AsyncMock(return_value={"verdict": "allow"})
            yield mock

    @pytest.fixture
    def governance_interceptor(self):
        """Create a GovernanceInterceptor for testing."""
        return GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=TemporalGovernanceState(),
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
            c
            for c in calls
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
            c
            for c in calls
            if c.kwargs.get("args", [[{}]])[0].get("payload", {}).get("event_type")
            == "WorkflowStarted"
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
            config=GovernanceConfig(
                send_start_event=False
            ),  # Disable start to simplify test
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
            c
            for c in calls
            if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowCompleted"
        ]
        assert len(completed_calls) == 1
        completed_payload = completed_calls[0].kwargs["args"][0]["payload"]
        assert completed_payload["workflow_id"] == "wf-123"
        assert completed_payload["workflow_type"] == "TestWorkflow"
        # Check serialized output
        assert completed_payload["workflow_output"] == {
            "status": "success",
            "data": {"key": "value"},
        }

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
        mock_next.execute_workflow = AsyncMock(
            side_effect=ValueError("Something went wrong")
        )
        inbound = inbound_class(mock_next)

        execute_input = MagicMock()

        with pytest.raises(ValueError):
            await inbound.execute_workflow(execute_input)

        # Find the WorkflowFailed call
        calls = mock_workflow_module.execute_activity.call_args_list
        failed_calls = [
            c
            for c in calls
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
            c
            for c in calls
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
            c
            for c in calls
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
    async def test_handle_signal_stores_verdict_in_state_if_block(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal records a run-scoped signal verdict on BLOCK."""
        state = TemporalGovernanceState()
        mock_workflow_module.execute_activity = AsyncMock(
            return_value={
                "verdict": "block",
                "reason": "High risk signal",
            }
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state,
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

        # The next activity in THIS run enforces the recorded verdict.
        assert state.get_signal_verdict("wf-123", "run-456") == (
            Verdict.BLOCK,
            "High risk signal",
        )

    @pytest.mark.asyncio
    async def test_handle_signal_stores_verdict_in_state_if_halt(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal records a run-scoped signal verdict on HALT."""
        state = TemporalGovernanceState()
        mock_workflow_module.execute_activity = AsyncMock(
            return_value={
                "verdict": "halt",
                "reason": "Critical alert",
            }
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state,
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

        assert state.get_signal_verdict("wf-123", "run-456") == (
            Verdict.HALT,
            "Critical alert",
        )

    @pytest.mark.asyncio
    async def test_handle_signal_does_not_store_verdict_if_allow(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal does not store verdict if ALLOW verdict."""
        state = TemporalGovernanceState()
        mock_workflow_module.execute_activity = AsyncMock(
            return_value={
                "verdict": "allow",
                "reason": "Signal approved",
            }
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state,
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

        # No verdict should be recorded for ALLOW.
        assert state.get_signal_verdict("wf-123", "run-456") is None

    @pytest.mark.asyncio
    async def test_handle_signal_does_not_fail_if_no_state(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal does not fail if state is None."""
        mock_workflow_module.execute_activity = AsyncMock(
            return_value={
                "verdict": "block",
                "reason": "High risk",
            }
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=None,  # No governance state
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

        # Should not raise even though verdict is BLOCK and there is no state.
        await inbound.handle_signal(signal_input)

        mock_next.handle_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_signal_uses_action_field_for_v1_compat(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal uses action field if verdict not present (v1.0 compat)."""
        state = TemporalGovernanceState()
        mock_workflow_module.execute_activity = AsyncMock(
            return_value={
                "action": "stop",  # v1.0 style
                "reason": "Blocked by v1 policy",
            }
        )

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state,
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
        assert state.get_signal_verdict("wf-123", "run-456") == (
            Verdict.HALT,
            "Blocked by v1 policy",
        )

    @pytest.mark.asyncio
    async def test_handle_signal_defaults_to_allow_if_no_result(
        self, mock_workflow_module, mock_workflow_info
    ):
        """Test handle_signal defaults to ALLOW if result is None."""
        state = TemporalGovernanceState()
        mock_workflow_module.execute_activity = AsyncMock(return_value=None)

        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state,
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

        # Should not store verdict since default is ALLOW.
        assert state.get_signal_verdict("wf-123", "run-456") is None


class TestInterceptorClosures:
    """Tests to verify that closures capture variables correctly."""

    @pytest.mark.asyncio
    async def test_interceptor_captures_config_values(self):
        """Test that the inner _Inbound class captures config values via closures."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_workflow,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-closure-test"
            mock_info.run_id = "run-closure"
            mock_info.workflow_type = "ClosureWorkflow"
            mock_info.task_queue = "closure-queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.side_effect = _legacy_patched
            mock_workflow.execute_activity = AsyncMock(
                return_value={"verdict": "allow"}
            )

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

            # Verify the captured values were used. Credentials are no longer
            # passed through activity input (they live on the activity instance
            # itself), so the activity input only carries payload/timeout/policy.
            calls = mock_workflow.execute_activity.call_args_list
            if calls:
                activity_input = calls[0].kwargs["args"][0]
                assert "api_url" not in activity_input
                assert "api_key" not in activity_input
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
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_workflow,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-none"
            mock_info.run_id = "run-none"
            mock_info.workflow_type = "NoneWorkflow"
            mock_info.task_queue = "queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.side_effect = _legacy_patched
            mock_workflow.execute_activity = AsyncMock(
                return_value={"verdict": "allow"}
            )

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
                c
                for c in calls
                if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowCompleted"
            ]
            assert (
                completed_calls[0].kwargs["args"][0]["payload"]["workflow_output"]
                is None
            )

    @pytest.mark.asyncio
    async def test_handle_signal_with_empty_args(self):
        """Test handle_signal with empty args list."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_workflow,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            mock_info = MagicMock()
            mock_info.workflow_id = "wf-signal"
            mock_info.run_id = "run-signal"
            mock_info.workflow_type = "SignalWorkflow"
            mock_info.task_queue = "queue"
            mock_workflow.info.return_value = mock_info
            mock_workflow.patched.side_effect = _legacy_patched
            mock_workflow.execute_activity = AsyncMock(
                return_value={"verdict": "allow"}
            )

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


class TestMultiAgentSessionPropagation:
    """Workflow interceptor tags events + stamps header from the memo session id."""

    def _interceptor(self):
        return GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=TemporalGovernanceState(),
            config=GovernanceConfig(),
        )

    def _mock_info(self):
        info = MagicMock()
        info.workflow_id = "wf-ma"
        info.run_id = "run-ma"
        info.workflow_type = "MaWorkflow"
        info.task_queue = "ma-queue"
        return info

    @pytest.mark.asyncio
    async def test_workflow_events_tagged_when_memo_set(self):
        """Memo set → WorkflowStarted/Completed payloads carry multi_agent_session_id."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_wf,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo",
                return_value="sess-abc",
            ),
        ):
            mock_wf.info.return_value = self._mock_info()
            mock_wf.patched.side_effect = _legacy_patched
            mock_wf.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.execute_workflow = AsyncMock(return_value="ok")
            inbound = inbound_class(mock_next)

            await inbound.execute_workflow(MagicMock())

            payloads = [
                c.kwargs["args"][0]["payload"]
                for c in mock_wf.execute_activity.call_args_list
            ]
            assert payloads, "expected at least one governance event"
            for payload in payloads:
                assert payload["multi_agent_session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_workflow_failed_tagged_when_memo_set(self):
        """Memo set → WorkflowFailed payload carries multi_agent_session_id."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_wf,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo",
                return_value="sess-abc",
            ),
        ):
            mock_wf.info.return_value = self._mock_info()
            mock_wf.patched.side_effect = _legacy_patched
            mock_wf.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.execute_workflow = AsyncMock(side_effect=ValueError("boom"))
            inbound = inbound_class(mock_next)

            with pytest.raises(ValueError):
                await inbound.execute_workflow(MagicMock())

            failed = [
                c.kwargs["args"][0]["payload"]
                for c in mock_wf.execute_activity.call_args_list
                if c.kwargs["args"][0]["payload"]["event_type"] == "WorkflowFailed"
            ]
            assert failed and failed[0]["multi_agent_session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_signal_tagged_when_memo_set(self):
        """Memo set → SignalReceived payload carries multi_agent_session_id."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_wf,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo",
                return_value="sess-abc",
            ),
        ):
            mock_wf.info.return_value = self._mock_info()
            mock_wf.patched.side_effect = _legacy_patched
            mock_wf.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.handle_signal = AsyncMock()
            inbound = inbound_class(mock_next)

            signal_input = MagicMock()
            signal_input.signal = "sig"
            signal_input.args = []
            await inbound.handle_signal(signal_input)

            payload = mock_wf.execute_activity.call_args_list[0].kwargs["args"][0][
                "payload"
            ]
            assert payload["multi_agent_session_id"] == "sess-abc"

    @pytest.mark.asyncio
    async def test_events_not_tagged_when_memo_absent(self):
        """Memo absent → no multi_agent_session_id key in any payload."""
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_wf,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            mock_wf.info.return_value = self._mock_info()
            mock_wf.patched.side_effect = _legacy_patched
            mock_wf.execute_activity = AsyncMock(return_value={"verdict": "allow"})

            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())
            mock_next = AsyncMock()
            mock_next.execute_workflow = AsyncMock(return_value="ok")
            inbound = inbound_class(mock_next)

            await inbound.execute_workflow(MagicMock())

            for c in mock_wf.execute_activity.call_args_list:
                payload = c.kwargs["args"][0]["payload"]
                assert "multi_agent_session_id" not in payload

    @pytest.mark.asyncio
    async def test_outbound_stamps_header_when_memo_set(self):
        """Outbound interceptor stamps HEADER_KEY on scheduled activities."""
        from temporalio.converter import default as _default_converter

        from openbox.multi_agent import HEADER_KEY

        conv = _default_converter().payload_converter
        with (
            patch("openbox.workflow_interceptor.workflow") as mock_wf,
            patch(
                "openbox.workflow_interceptor.read_session_from_memo",
                return_value="sess-out",
            ),
        ):
            mock_wf.payload_converter.return_value = conv

            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())

            # Capture the _Outbound instance the inbound wraps during init().
            captured = {}
            mock_next = MagicMock()
            mock_next.init = lambda outbound: captured.__setitem__("outbound", outbound)
            inbound = inbound_class(mock_next)
            inbound.init(MagicMock())
            outbound = captured["outbound"]

            activity_input = MagicMock()
            activity_input.headers = {}
            outbound.start_activity(activity_input)

            assert HEADER_KEY in activity_input.headers
            assert (
                conv.from_payload(activity_input.headers[HEADER_KEY], str) == "sess-out"
            )

    @pytest.mark.asyncio
    async def test_outbound_no_header_when_memo_absent(self):
        """Outbound interceptor leaves headers untouched when no session id."""
        from openbox.multi_agent import HEADER_KEY

        with (
            patch("openbox.workflow_interceptor.workflow"),
            patch(
                "openbox.workflow_interceptor.read_session_from_memo", return_value=None
            ),
        ):
            inbound_class = self._interceptor().workflow_interceptor_class(MagicMock())
            captured = {}
            mock_next = MagicMock()
            mock_next.init = lambda outbound: captured.__setitem__("outbound", outbound)
            inbound = inbound_class(mock_next)
            inbound.init(MagicMock())
            outbound = captured["outbound"]

            activity_input = MagicMock()
            activity_input.headers = {}
            outbound.start_activity(activity_input)

            assert HEADER_KEY not in activity_input.headers


# --------------------------------------------------------------------------- #
# Phase 5 — retryable-block execute_workflow behavior (patched path)
# --------------------------------------------------------------------------- #

import asyncio  # noqa: E402
from contextlib import contextmanager  # noqa: E402

from temporalio.exceptions import ApplicationError  # noqa: E402

from openbox.errors import (  # noqa: E402
    GOVERNANCE_HALT_ERROR_TYPE,
    GOVERNANCE_RETRY_LIMIT_EXCEEDED_ERROR_TYPE,
    GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
)
from openbox.retry_coordinator import RetryableBlockCoordinator  # noqa: E402
from openbox.retryable_block import RetryableBlockRequest  # noqa: E402


class _ContinueAsNewSignal(Exception):
    """Stand-in for temporalio's ContinueAsNewError (control flow, Exception-based)
    so tests can assert continue_as_new was invoked without a live Temporal loop.
    Being an ``Exception`` also guards that no ``except Exception`` swallows the
    real control error."""


class _FakeActivityError(Exception):
    """Stand-in for temporalio ActivityError, which exposes a ``.cause`` property."""

    def __init__(self, cause):
        super().__init__("activity failed")
        self.cause = cause


def _mk_retry_request(new_input="corrected", event_type="WorkflowStarted"):
    return RetryableBlockRequest(
        schema_version=1,
        new_input=new_input,
        governance_event_id="evt",
        reason="retry",
        event_type=event_type,
        hook_trigger=False,
        hook_stage=None,
    )


def _retryable_app_error(req):
    return ApplicationError(
        "Governance requested workflow restart",
        req.to_dict(),
        type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
        non_retryable=True,
    )


def _activity_script(script):
    """Build an execute_activity side_effect from {event_type: ("return"|"raise", v)}.

    Unlisted event types return an ALLOW verdict dict.
    """

    def _fn(name, *, args, **kwargs):
        event_type = args[0]["payload"]["event_type"]
        action, value = script.get(event_type, ("return", {"verdict": "allow"}))
        if action == "raise":
            raise value
        return value

    return _fn


async def _real_wait_condition(predicate, *args, **kwargs):
    """A deterministic wait_condition for tests: poll the predicate on the real
    event loop until it is satisfied."""
    for _ in range(100000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("wait_condition predicate never satisfied")


@contextmanager
def _patched_retryable_workflow(script=None, *, memo_value=0, patched_v1=True):
    # The interceptor and the coordinator/budget module each hold their own
    # `workflow` reference; both must be patched. next_restart_memo (budget) reads
    # openbox.retry_coordinator.workflow.memo / memo_value.
    with (
        patch("openbox.workflow_interceptor.workflow") as mock,
        patch("openbox.retry_coordinator.workflow") as rc_mock,
        patch("openbox.workflow_interceptor.read_session_from_memo", return_value=None),
    ):
        mock.info.return_value = MagicMock(
            workflow_id="wf-1",
            run_id="run-1",
            workflow_type="TestWorkflow",
            task_queue="q",
        )
        if patched_v1:
            mock.patched.return_value = True
        else:
            mock.patched.side_effect = _legacy_patched
        mock.execute_activity = AsyncMock(side_effect=_activity_script(script or {}))
        mock.wait_condition = _real_wait_condition
        mock.all_handlers_finished = MagicMock(return_value=True)
        mock.continue_as_new = MagicMock(side_effect=_ContinueAsNewSignal())
        # Restart-budget memo lives on the coordinator module's workflow ref.
        rc_mock.memo = MagicMock(return_value={})
        rc_mock.memo_value = MagicMock(return_value=memo_value)
        yield mock


def _make_inbound(next_execute, config=None):
    interceptor = GovernanceInterceptor(
        api_url="https://api.openbox.ai",
        api_key="test-key",
        state=TemporalGovernanceState(),
        config=config or GovernanceConfig(),
    )
    inbound_class = interceptor.workflow_interceptor_class(MagicMock())
    mock_next = AsyncMock()
    mock_next.execute_workflow = next_execute
    return inbound_class(mock_next)


class TestIsHalt:
    """Unit tests for the HALT-detection helper used for HALT-over-retry dominance."""

    def test_none_is_not_halt(self):
        assert _is_halt(None) is False

    def test_typed_halt_error_detected(self):
        assert _is_halt(GovernanceHaltError("stop")) is True

    def test_application_error_halt_and_stop_types(self):
        assert _is_halt(ApplicationError("x", type="GovernanceHalt")) is True
        assert _is_halt(ApplicationError("x", type="GovernanceStop")) is True

    def test_plain_exception_not_halt(self):
        assert _is_halt(ValueError("x")) is False

    def test_block_type_not_halt(self):
        assert _is_halt(ApplicationError("x", type="GovernanceBlock")) is False

    def test_halt_in_cause_chain(self):
        assert _is_halt(_FakeActivityError(GovernanceHaltError("deep"))) is True


class TestRetryableBlockExecuteWorkflow:
    """Phase-5 retryable-block Continue-As-New behavior in execute_workflow."""

    @pytest.mark.asyncio
    async def test_unpatched_takes_legacy_path_no_can(self):
        """patched(retryable-block-v1)=False → legacy path: no coordinator race,
        no Continue-As-New."""
        user = AsyncMock(return_value="ok")
        with _patched_retryable_workflow({}, patched_v1=False) as mock:
            inbound = _make_inbound(user)
            result = await inbound.execute_workflow(MagicMock(args=[]))
        assert result == "ok"
        mock.continue_as_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_started_block_plan_cans_before_user_code(self):
        req = _mk_retry_request(new_input={"q": "fixed"}, event_type="WorkflowStarted")
        script = {"WorkflowStarted": ("raise", _retryable_app_error(req))}
        user = AsyncMock(return_value="should-not-run")
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=[]))
        user.assert_not_called()  # user code never runs
        assert mock.continue_as_new.call_args.args == ({"q": "fixed"},)

    @pytest.mark.asyncio
    async def test_workflow_completed_block_plan_cans_instead_of_returning(self):
        req = _mk_retry_request(new_input="fixed", event_type="WorkflowCompleted")
        script = {"WorkflowCompleted": ("raise", _retryable_app_error(req))}
        user = AsyncMock(return_value="user-result")
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=["orig"]))
        user.assert_awaited_once()
        assert mock.continue_as_new.call_args.args == ("fixed",)

    @pytest.mark.asyncio
    async def test_activity_origin_retry_extracted_and_cans(self):
        req = _mk_retry_request(new_input=[1, 2], event_type="ActivityCompleted")
        user = AsyncMock(side_effect=_FakeActivityError(_retryable_app_error(req)))
        with _patched_retryable_workflow({}) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=[]))
        # A list new_input is passed as ONE workflow argument.
        assert mock.continue_as_new.call_args.args == ([1, 2],)
        # An activity-origin retry is NOT reported as WorkflowFailed.
        event_types = [
            c.kwargs["args"][0]["payload"]["event_type"]
            for c in mock.execute_activity.call_args_list
        ]
        assert "WorkflowFailed" not in event_types

    @pytest.mark.asyncio
    async def test_new_input_none_reuses_current_args(self):
        req = _mk_retry_request(new_input=None, event_type="WorkflowStarted")
        script = {"WorkflowStarted": ("raise", _retryable_app_error(req))}
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(AsyncMock())
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=["a", "b"]))
        assert mock.continue_as_new.call_args.args == ()
        assert mock.continue_as_new.call_args.kwargs["args"] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_budget_exhausted_raises_limit_and_no_can(self):
        req = _mk_retry_request(new_input="x", event_type="WorkflowStarted")
        script = {"WorkflowStarted": ("raise", _retryable_app_error(req))}
        # memo counter already at the default cap (3) → next would be 4 > 3.
        with _patched_retryable_workflow(script, memo_value=3) as mock:
            inbound = _make_inbound(AsyncMock())
            with pytest.raises(ApplicationError) as exc:
                await inbound.execute_workflow(MagicMock(args=[]))
        assert exc.value.type == GOVERNANCE_RETRY_LIMIT_EXCEEDED_ERROR_TYPE
        mock.continue_as_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_genuine_failure_reports_workflow_failed_and_reraises(self):
        user = AsyncMock(side_effect=ValueError("boom"))
        with _patched_retryable_workflow({}) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(ValueError):
                await inbound.execute_workflow(MagicMock(args=[]))
        event_types = [
            c.kwargs["args"][0]["payload"]["event_type"]
            for c in mock.execute_activity.call_args_list
        ]
        assert "WorkflowFailed" in event_types
        mock.continue_as_new.assert_not_called()


class TestSendGovernanceEventRetryable:
    """Dispatcher-level retryable-block conversion + legacy degrade."""

    @pytest.fixture
    def mock_workflow(self):
        with patch("openbox.workflow_interceptor.workflow") as mock:
            yield mock

    def _err(self, event_type="SignalReceived", schema_version=1):
        return ApplicationError(
            "restart",
            {
                "schema_version": schema_version,
                "new_input": "x",
                "event_type": event_type,
                "hook_trigger": False,
                "hook_stage": None,
                "governance_event_id": None,
                "reason": None,
            },
            type=GOVERNANCE_RETRYABLE_BLOCK_ERROR_TYPE,
            non_retryable=True,
        )

    @pytest.mark.asyncio
    async def test_enabled_returns_request(self, mock_workflow):
        mock_workflow.execute_activity = AsyncMock(
            side_effect=self._err("WorkflowStarted")
        )
        result = await _send_governance_event(
            {"event_type": "WorkflowStarted"}, 30.0, retryable_block_enabled=True
        )
        assert isinstance(result, RetryableBlockRequest)
        assert result.new_input == "x"

    @pytest.mark.asyncio
    async def test_disabled_degrades_signal_to_block_dict(self, mock_workflow):
        """Legacy/unpatched caller: a retryable signal degrades to a blocking dict
        (never ALLOW), so the legacy signal path still blocks."""
        mock_workflow.execute_activity = AsyncMock(
            side_effect=self._err("SignalReceived")
        )
        result = await _send_governance_event(
            {"event_type": "SignalReceived"}, 30.0, retryable_block_enabled=False
        )
        assert result == {
            "success": True,
            "verdict": "block",
            "reason": "Governance blocked",
        }

    @pytest.mark.asyncio
    async def test_disabled_degrades_lifecycle_to_none(self, mock_workflow):
        mock_workflow.execute_activity = AsyncMock(
            side_effect=self._err("WorkflowStarted")
        )
        result = await _send_governance_event(
            {"event_type": "WorkflowStarted"}, 30.0, retryable_block_enabled=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_enabled_extraction_failure_degrades_signal_to_block(
        self, mock_workflow
    ):
        """Enabled caller but malformed envelope (unknown schema): fail safe as a
        plain BLOCK via the event-specific degrade — never ALLOW."""
        mock_workflow.execute_activity = AsyncMock(
            side_effect=self._err("SignalReceived", schema_version=999)
        )
        result = await _send_governance_event(
            {"event_type": "SignalReceived"}, 30.0, retryable_block_enabled=True
        )
        assert result == {
            "success": True,
            "verdict": "block",
            "reason": "Governance blocked",
        }


class TestRetryableBlockSignals:
    """Phase-6 signal handling: retryable signals submit to the coordinator and
    never invoke the user handler; plain/legacy stops keep the existing bridge."""

    def _inbound_with_coordinator(self, coordinator, state=None):
        interceptor = GovernanceInterceptor(
            api_url="https://api.openbox.ai",
            api_key="test-key",
            state=state if state is not None else TemporalGovernanceState(),
            config=GovernanceConfig(),
        )
        inbound_class = interceptor.workflow_interceptor_class(MagicMock())
        mock_next = AsyncMock()
        inbound = inbound_class(mock_next)
        inbound._coordinator = coordinator
        return inbound, mock_next

    @pytest.mark.asyncio
    async def test_retryable_signal_submits_and_skips_user_handler(self):
        req = _mk_retry_request(new_input="fix", event_type="SignalReceived")
        script = {"SignalReceived": ("raise", _retryable_app_error(req))}
        coord = RetryableBlockCoordinator()
        with _patched_retryable_workflow(script):
            inbound, mock_next = self._inbound_with_coordinator(coord)
            await inbound.handle_signal(MagicMock(signal="s", args=[]))
        assert coord.get_request() == req
        mock_next.handle_signal.assert_not_called()  # user handler skipped

    @pytest.mark.asyncio
    async def test_retryable_signal_without_coordinator_falls_back_to_block(self):
        req = _mk_retry_request(new_input="fix", event_type="SignalReceived")
        script = {"SignalReceived": ("raise", _retryable_app_error(req))}
        state = TemporalGovernanceState()
        with _patched_retryable_workflow(script):
            inbound, mock_next = self._inbound_with_coordinator(None, state=state)
            await inbound.handle_signal(MagicMock(signal="s", args=[]))
        # Fail safe: enforced as a plain block via the activity bridge.
        assert state.get_signal_verdict("wf-1", "run-1") == (Verdict.BLOCK, "retry")
        mock_next.handle_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_plain_block_signal_on_patched_path_records_verdict_and_runs_handler(
        self,
    ):
        script = {"SignalReceived": ("return", {"verdict": "block", "reason": "r"})}
        state = TemporalGovernanceState()
        coord = RetryableBlockCoordinator()
        with _patched_retryable_workflow(script):
            inbound, mock_next = self._inbound_with_coordinator(coord, state=state)
            await inbound.handle_signal(MagicMock(signal="s", args=[]))
        # Plain BLOCK keeps the existing bridge behavior + the user handler runs.
        assert state.get_signal_verdict("wf-1", "run-1") == (Verdict.BLOCK, "r")
        assert coord.get_request() is None
        mock_next.handle_signal.assert_awaited_once()


class TestRetryableBlockWorkflowFailedAndCoordinator:
    """Phase-6 WorkflowFailed-origin retry, HALT dominance, coordinator-win."""

    @pytest.mark.asyncio
    async def test_workflow_failed_with_plan_cans_overriding_failure(self):
        req = _mk_retry_request(new_input="recovered", event_type="WorkflowFailed")
        script = {"WorkflowFailed": ("raise", _retryable_app_error(req))}
        user = AsyncMock(side_effect=ValueError("boom"))
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=[]))
        assert mock.continue_as_new.call_args.args == ("recovered",)

    @pytest.mark.asyncio
    async def test_workflow_failed_reporting_halt_does_not_shadow_original(self):
        """A HALT raised while REPORTING WorkflowFailed is swallowed at the
        interceptor so it never shadows the original workflow exception. (A genuine
        policy HALT still terminates the workflow via the reporting activity's
        client.terminate(), which is out-of-band from this interceptor-level
        unit.)"""
        halt = ApplicationError(
            "halt", type=GOVERNANCE_HALT_ERROR_TYPE, non_retryable=True
        )
        script = {"WorkflowFailed": ("raise", halt)}
        user = AsyncMock(side_effect=ValueError("original"))
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(ValueError, match="original"):
                await inbound.execute_workflow(MagicMock(args=[]))
        mock.continue_as_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_workflow_failed_reporting_api_error_does_not_shadow_original(self):
        """A fail_closed governance-API outage while reporting WorkflowFailed (the
        dispatcher maps it to GovernanceHaltError) is swallowed so the ORIGINAL
        workflow exception is preserved for the caller."""
        api_err = ApplicationError("api down", type="GovernanceAPIError")
        script = {"WorkflowFailed": ("raise", api_err)}
        user = AsyncMock(side_effect=ValueError("original"))
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(ValueError, match="original"):
                await inbound.execute_workflow(MagicMock(args=[]))
        mock.continue_as_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_halt_dominates_coordinator_retry_request(self):
        """User code submits a retry to the coordinator, then the same turn raises
        HALT — HALT wins (priority: HALT > retryable BLOCK), no Continue-As-New."""
        req = _mk_retry_request(new_input="x", event_type="Handoff")

        async def user(_input):
            from openbox.retry_coordinator import get_coordinator

            get_coordinator().submit(req)
            raise ApplicationError(
                "halt", type=GOVERNANCE_HALT_ERROR_TYPE, non_retryable=True
            )

        with _patched_retryable_workflow({}) as mock:
            inbound = _make_inbound(AsyncMock(side_effect=user))
            with pytest.raises(ApplicationError) as exc:
                await inbound.execute_workflow(MagicMock(args=[]))
        assert exc.value.type == GOVERNANCE_HALT_ERROR_TYPE
        mock.continue_as_new.assert_not_called()

    @pytest.mark.asyncio
    async def test_coordinator_win_continues_as_new_after_submission(self):
        """A submission reachable via the run-scoped ContextVar (handoff-style)
        drives Continue-As-New from the main path."""
        req = _mk_retry_request(new_input="via-coord", event_type="Handoff")

        async def user(_input):
            from openbox.retry_coordinator import get_coordinator

            get_coordinator().submit(req)
            return "done"

        with _patched_retryable_workflow({}) as mock:
            inbound = _make_inbound(AsyncMock(side_effect=user))
            with pytest.raises(_ContinueAsNewSignal):
                await inbound.execute_workflow(MagicMock(args=[]))
        assert mock.continue_as_new.call_args.args == ("via-coord",)

    @pytest.mark.asyncio
    async def test_halt_in_user_exception_not_overridden_by_failed_retry(self):
        """A genuine HALT surfacing through user code dominates: it is re-raised and
        NEVER overridden by a WorkflowFailed retry plan (priority HALT > retryable
        BLOCK; acceptance: HALT never restarts). The WorkflowFailed evaluation must
        not even be reached."""
        halt = ApplicationError(
            "halt", type=GOVERNANCE_HALT_ERROR_TYPE, non_retryable=True
        )
        req = _mk_retry_request(new_input="x", event_type="WorkflowFailed")
        # WorkflowFailed would return a plan if reached — it must NOT be reached.
        script = {"WorkflowFailed": ("raise", _retryable_app_error(req))}
        user = AsyncMock(side_effect=halt)
        with _patched_retryable_workflow(script) as mock:
            inbound = _make_inbound(user)
            with pytest.raises(ApplicationError) as exc_info:
                await inbound.execute_workflow(MagicMock(args=[]))
        assert exc_info.value.type == GOVERNANCE_HALT_ERROR_TYPE
        mock.continue_as_new.assert_not_called()
        # WorkflowFailed evaluation was short-circuited by the HALT guard.
        event_types = [
            c.kwargs["args"][0]["payload"]["event_type"]
            for c in mock.execute_activity.call_args_list
        ]
        assert "WorkflowFailed" not in event_types
