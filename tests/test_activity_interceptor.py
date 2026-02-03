# tests/test_activity_interceptor.py
"""Comprehensive tests for the OpenBox SDK activity_interceptor module."""

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import sys

import pytest

from openbox.activity_interceptor import (
    _rfc3339_now,
    _deep_update_dataclass,
    _serialize_value,
    ActivityGovernanceInterceptor,
    _ActivityInterceptor,
)
from openbox.types import (
    Verdict,
    WorkflowSpanBuffer,
    GovernanceVerdictResponse,
    GuardrailsCheckResult,
)
from openbox.config import GovernanceConfig


# =============================================================================
# Helper Fixtures and Dataclasses for Testing
# =============================================================================


@dataclass
class NestedData:
    """Nested dataclass for testing _deep_update_dataclass."""
    value: str = ""
    count: int = 0


@dataclass
class OuterData:
    """Outer dataclass with nested dataclass for testing."""
    name: str = ""
    nested: NestedData = field(default_factory=NestedData)
    items: List[str] = field(default_factory=list)


@dataclass
class DataWithList:
    """Dataclass with list of nested dataclasses."""
    entries: List[NestedData] = field(default_factory=list)


@dataclass
class ActivityInput:
    """Sample activity input dataclass for testing redaction."""
    prompt: str = ""
    user_id: str = ""
    metadata: dict = field(default_factory=dict)


class MockTemporalPayload:
    """Mock Temporal Payload object for testing serialization."""
    def __init__(self, data: bytes, metadata: Optional[dict] = None):
        self.data = data
        self.metadata = metadata or {}


class NonSerializableObject:
    """Object that can't be JSON serialized."""
    def __init__(self, value):
        self._value = value

    def __str__(self):
        return f"NonSerializable({self._value})"


@pytest.fixture
def mock_activity_info():
    """Create a mock activity.info() return value."""
    info = MagicMock()
    info.workflow_id = "test-workflow-id"
    info.workflow_run_id = "test-run-id"
    info.workflow_type = "TestWorkflow"
    info.activity_id = "test-activity-id"
    info.activity_type = "test_activity"
    info.task_queue = "test-queue"
    info.attempt = 1
    return info


@pytest.fixture
def mock_span_processor():
    """Create a mock WorkflowSpanProcessor."""
    processor = MagicMock()
    processor.get_buffer.return_value = None
    processor.get_verdict.return_value = None
    processor.get_pending_body.return_value = None
    return processor


@pytest.fixture
def governance_config():
    """Create a default GovernanceConfig."""
    return GovernanceConfig()


def create_mock_httpx_client(response_data, status_code=200):
    """Create a mock httpx async client with specified response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_data

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client, mock_client_instance


# =============================================================================
# Tests for _rfc3339_now()
# =============================================================================


class TestRfc3339Now:
    """Tests for _rfc3339_now() function."""

    def test_returns_string(self):
        """Test that _rfc3339_now returns a string."""
        result = _rfc3339_now()
        assert isinstance(result, str)

    def test_ends_with_z_suffix(self):
        """Test that result ends with 'Z' suffix (UTC indicator)."""
        result = _rfc3339_now()
        assert result.endswith("Z")

    def test_rfc3339_format(self):
        """Test that result matches RFC3339 format."""
        result = _rfc3339_now()
        # RFC3339 format: YYYY-MM-DDTHH:MM:SS.mmmZ
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
        assert re.match(pattern, result), f"'{result}' doesn't match RFC3339 format"

    def test_has_milliseconds(self):
        """Test that result includes milliseconds (3 digits before Z)."""
        result = _rfc3339_now()
        # Extract milliseconds part
        ms_part = result.split(".")[1][:3]
        assert len(ms_part) == 3
        assert ms_part.isdigit()

    def test_returns_recent_time(self):
        """Test that returned time is within recent timeframe."""
        from datetime import datetime, timezone, timedelta

        result = _rfc3339_now()

        # Parse the result (handle 3-digit milliseconds)
        # Result is like "2026-02-02T17:35:50.719Z"
        result_no_z = result[:-1]  # Remove Z
        # Pad milliseconds to 6 digits for fromisoformat
        parts = result_no_z.split(".")
        if len(parts) == 2:
            parts[1] = parts[1].ljust(6, '0')
            result_padded = ".".join(parts)
        else:
            result_padded = result_no_z

        result_time = datetime.fromisoformat(result_padded)
        result_time = result_time.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        # The result should be within 1 second of now
        assert abs((now - result_time).total_seconds()) < 1


# =============================================================================
# Tests for _deep_update_dataclass()
# =============================================================================


class TestDeepUpdateDataclass:
    """Tests for _deep_update_dataclass() function."""

    def test_updates_simple_fields(self):
        """Test updating simple dataclass fields."""
        data = NestedData(value="original", count=1)
        update = {"value": "updated", "count": 42}

        _deep_update_dataclass(data, update)

        assert data.value == "updated"
        assert data.count == 42

    def test_recursively_updates_nested_dataclass(self):
        """Test recursively updating nested dataclass fields."""
        data = OuterData(
            name="outer",
            nested=NestedData(value="inner", count=10),
        )
        update = {
            "name": "new_outer",
            "nested": {"value": "new_inner", "count": 99},
        }

        _deep_update_dataclass(data, update)

        assert data.name == "new_outer"
        assert data.nested.value == "new_inner"
        assert data.nested.count == 99

    def test_updates_list_of_dataclasses(self):
        """Test updating list of dataclasses."""
        data = DataWithList(
            entries=[
                NestedData(value="first", count=1),
                NestedData(value="second", count=2),
            ]
        )
        update = {
            "entries": [
                {"value": "updated_first", "count": 100},
                {"value": "updated_second", "count": 200},
            ]
        }

        _deep_update_dataclass(data, update)

        assert data.entries[0].value == "updated_first"
        assert data.entries[0].count == 100
        assert data.entries[1].value == "updated_second"
        assert data.entries[1].count == 200

    def test_skips_fields_not_in_data(self):
        """Test that fields not in update dict are not modified."""
        data = NestedData(value="original", count=42)
        update = {"value": "updated"}  # count not included

        _deep_update_dataclass(data, update)

        assert data.value == "updated"
        assert data.count == 42  # Unchanged

    def test_handles_non_dataclass_objects(self):
        """Test that non-dataclass objects are not modified (no-op)."""
        obj = {"key": "value"}
        original = {"key": "value"}
        update = {"key": "new_value"}

        # Should be a no-op for non-dataclass objects
        _deep_update_dataclass(obj, update)

        assert obj == original

    def test_handles_dataclass_type_not_instance(self):
        """Test that dataclass types (not instances) are not modified."""
        update = {"value": "test"}

        # Should be a no-op when passing the class itself
        _deep_update_dataclass(NestedData, update)

        # Class should be unchanged
        new_instance = NestedData()
        assert new_instance.value == ""

    def test_partial_nested_update(self):
        """Test partial updates to nested dataclass."""
        data = OuterData(
            name="outer",
            nested=NestedData(value="inner", count=10),
        )
        update = {
            "nested": {"count": 999},  # Only update count, not value
        }

        _deep_update_dataclass(data, update)

        assert data.name == "outer"  # Unchanged
        assert data.nested.value == "inner"  # Unchanged
        assert data.nested.count == 999  # Updated

    def test_update_with_logger(self):
        """Test that logger is called when provided."""
        mock_logger = MagicMock()
        data = NestedData(value="original", count=1)
        update = {"value": "updated"}

        _deep_update_dataclass(data, update, _logger=mock_logger)

        assert data.value == "updated"
        mock_logger.info.assert_called()

    def test_handles_empty_update_dict(self):
        """Test that empty update dict doesn't modify anything."""
        data = NestedData(value="original", count=42)
        update = {}

        _deep_update_dataclass(data, update)

        assert data.value == "original"
        assert data.count == 42

    def test_list_with_primitive_values(self):
        """Test updating list with primitive values (not dataclasses)."""
        data = OuterData(name="test", items=["a", "b", "c"])
        update = {"items": ["x", "y", "z"]}

        _deep_update_dataclass(data, update)

        # List items should be replaced in-place
        assert data.items == ["x", "y", "z"]


# =============================================================================
# Tests for _serialize_value()
# =============================================================================


class TestSerializeValue:
    """Tests for _serialize_value() function."""

    def test_none_returns_none(self):
        """Test that None returns None."""
        assert _serialize_value(None) is None

    def test_string_passes_through(self):
        """Test that strings pass through unchanged."""
        assert _serialize_value("hello") == "hello"
        assert _serialize_value("") == ""

    def test_int_passes_through(self):
        """Test that integers pass through unchanged."""
        assert _serialize_value(42) == 42
        assert _serialize_value(0) == 0
        assert _serialize_value(-100) == -100

    def test_float_passes_through(self):
        """Test that floats pass through unchanged."""
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(0.0) == 0.0

    def test_bool_passes_through(self):
        """Test that booleans pass through unchanged."""
        assert _serialize_value(True) is True
        assert _serialize_value(False) is False

    def test_bytes_decode_to_utf8(self):
        """Test that bytes decode to UTF-8 string."""
        data = b"hello world"
        assert _serialize_value(data) == "hello world"

    def test_bytes_fallback_to_base64(self):
        """Test that non-UTF-8 bytes fallback to base64 encoding."""
        # Invalid UTF-8 sequence
        data = b"\xff\xfe\x00\x01"
        result = _serialize_value(data)
        expected = base64.b64encode(data).decode("ascii")
        assert result == expected

    def test_dataclass_converts_to_dict(self):
        """Test that dataclass converts to dict."""
        data = NestedData(value="test", count=42)
        result = _serialize_value(data)

        assert result == {"value": "test", "count": 42}

    def test_nested_dataclass_converts_to_nested_dict(self):
        """Test that nested dataclass converts to nested dict."""
        data = OuterData(
            name="outer",
            nested=NestedData(value="inner", count=10),
            items=["a", "b"],
        )
        result = _serialize_value(data)

        assert result == {
            "name": "outer",
            "nested": {"value": "inner", "count": 10},
            "items": ["a", "b"],
        }

    def test_list_recursively_serializes(self):
        """Test that list elements are recursively serialized."""
        data = [
            NestedData(value="first", count=1),
            "string",
            42,
            None,
        ]
        result = _serialize_value(data)

        assert result == [
            {"value": "first", "count": 1},
            "string",
            42,
            None,
        ]

    def test_tuple_recursively_serializes(self):
        """Test that tuple elements are recursively serialized."""
        data = (NestedData(value="test", count=1), "hello")
        result = _serialize_value(data)

        assert result == [{"value": "test", "count": 1}, "hello"]

    def test_dict_recursively_serializes(self):
        """Test that dict values are recursively serialized."""
        data = {
            "nested": NestedData(value="test", count=1),
            "items": [1, 2, 3],
            "primitive": "hello",
        }
        result = _serialize_value(data)

        assert result == {
            "nested": {"value": "test", "count": 1},
            "items": [1, 2, 3],
            "primitive": "hello",
        }

    def test_temporal_payload_decoded_json(self):
        """Test that Temporal Payload objects with JSON data are decoded."""
        payload = MockTemporalPayload(
            data=b'{"key": "value"}',
            metadata={"encoding": "json"},
        )
        result = _serialize_value(payload)

        assert result == {"key": "value"}

    def test_temporal_payload_binary_fallback(self):
        """Test that Temporal Payload with invalid data returns description."""
        payload = MockTemporalPayload(
            data=b"\xff\xfe",  # Invalid UTF-8
            metadata={},
        )
        result = _serialize_value(payload)

        assert "<Payload:" in result
        assert "bytes>" in result

    def test_fallback_to_str_for_other_objects(self):
        """Test that other objects fallback to str() representation."""
        obj = NonSerializableObject("test")
        result = _serialize_value(obj)

        assert result == "NonSerializable(test)"

    def test_deeply_nested_structures(self):
        """Test serialization of deeply nested structures."""
        data = {
            "level1": {
                "level2": [
                    {"level3": NestedData(value="deep", count=999)},
                ],
            },
        }
        result = _serialize_value(data)

        assert result == {
            "level1": {
                "level2": [
                    {"level3": {"value": "deep", "count": 999}},
                ],
            },
        }


# =============================================================================
# Tests for ActivityGovernanceInterceptor class
# =============================================================================


class TestActivityGovernanceInterceptor:
    """Tests for ActivityGovernanceInterceptor class."""

    def test_initialization(self, mock_span_processor):
        """Test interceptor initialization with all parameters."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=GovernanceConfig(),
        )

        assert interceptor.api_url == "http://localhost:8086"
        assert interceptor.api_key == "obx_test_key123"
        assert interceptor.span_processor is mock_span_processor
        assert isinstance(interceptor.config, GovernanceConfig)

    def test_initialization_with_default_config(self, mock_span_processor):
        """Test interceptor initialization with default config."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
        )

        assert isinstance(interceptor.config, GovernanceConfig)

    def test_api_url_trailing_slash_stripped(self, mock_span_processor):
        """Test that trailing slash is stripped from api_url."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086/",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
        )

        assert interceptor.api_url == "http://localhost:8086"

    def test_api_url_multiple_trailing_slashes_stripped(self, mock_span_processor):
        """Test that multiple trailing slashes are stripped."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086///",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
        )

        assert interceptor.api_url == "http://localhost:8086"

    def test_intercept_activity_returns_activity_interceptor(self, mock_span_processor):
        """Test that intercept_activity returns _ActivityInterceptor."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
        )
        mock_next = MagicMock()

        result = interceptor.intercept_activity(mock_next)

        assert isinstance(result, _ActivityInterceptor)


# =============================================================================
# Tests for _ActivityInterceptor class
# =============================================================================


class TestActivityInterceptor:
    """Tests for _ActivityInterceptor class."""

    @pytest.fixture
    def interceptor(self, mock_span_processor, governance_config):
        """Create an _ActivityInterceptor instance."""
        mock_next = AsyncMock()
        return _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

    # =========================================================================
    # Tests for execute_activity()
    # =========================================================================

    @pytest.mark.asyncio
    async def test_skips_if_activity_type_in_skip_list(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that activity is skipped if activity_type is in skip_activity_types."""
        config = GovernanceConfig(skip_activity_types={"test_activity"})
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="activity_result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = ["arg1"]

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor.execute_activity(mock_input)

        assert result == "activity_result"
        mock_next.execute_activity.assert_called_once_with(mock_input)

    @pytest.mark.asyncio
    async def test_checks_pending_verdict_raises_governance_stop(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that pending BLOCK verdict raises GovernanceStop."""
        mock_span_processor.get_verdict.return_value = {
            "verdict": "block",
            "reason": "Blocked by policy",
            "run_id": "test-run-id",
        }

        config = GovernanceConfig()
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "GovernanceStop"
            assert "Governance blocked" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_clears_stale_buffer_from_previous_run(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that stale buffer from previous run is cleared."""
        stale_buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="old-run-id",  # Different from current
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )
        mock_span_processor.get_buffer.return_value = stale_buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        # Create mock httpx module
        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        # Verify stale buffer was cleared
        mock_span_processor.unregister_workflow.assert_called_with("test-workflow-id")

    @pytest.mark.asyncio
    async def test_clears_stale_verdict_from_previous_run(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that stale verdict from previous run is cleared."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = {
            "verdict": "block",
            "reason": "Old verdict",
            "run_id": "old-run-id",  # Different from current
        }

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        mock_span_processor.clear_verdict.assert_called_with("test-workflow-id")

    @pytest.mark.asyncio
    async def test_registers_buffer_if_not_exists(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that buffer is registered if it doesn't exist."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        mock_span_processor.register_workflow.assert_called()
        call_args = mock_span_processor.register_workflow.call_args
        assert call_args[0][0] == "test-workflow-id"
        assert isinstance(call_args[0][1], WorkflowSpanBuffer)

    @pytest.mark.asyncio
    async def test_sends_activity_started_event_if_enabled(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that ActivityStarted event is sent if send_activity_start_event=True."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = ["test_arg"]

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        # Verify API was called at least twice (ActivityStarted + ActivityCompleted)
        assert mock_client_instance.post.call_count >= 2
        calls = mock_client_instance.post.call_args_list
        # First call should be ActivityStarted
        first_call = calls[0]
        payload = first_call[1]["json"]
        assert payload["event_type"] == "ActivityStarted"

    @pytest.mark.asyncio
    async def test_raises_governance_stop_on_block_verdict(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that GovernanceStop is raised for BLOCK verdict."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "block",
            "reason": "Policy violation",
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "GovernanceStop"
            assert "Policy violation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_guardrails_validation_failed_if_validation_passed_false(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that GuardrailsValidationFailed is raised when validation_passed=False."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "allow",
            "guardrails_result": {
                "redacted_input": {},
                "input_type": "activity_input",
                "validation_passed": False,
                "reasons": [{"reason": "PII detected"}],
            },
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "GuardrailsValidationFailed"
            assert "PII detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_applies_guardrails_redaction_to_dataclass_input(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that guardrails redaction is applied to dataclass input."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        # Create dataclass input
        input_data = ActivityInput(
            prompt="original prompt with PII",
            user_id="user123",
        )
        mock_input = MagicMock()
        mock_input.args = [input_data]

        # Track call count for different responses
        call_count = [0]

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            mock_response.status_code = 200
            if call_count[0] == 1:  # ActivityStarted
                mock_response.json.return_value = {
                    "verdict": "allow",
                    "guardrails_result": {
                        "redacted_input": [{"prompt": "[REDACTED]", "user_id": "user123", "metadata": {}}],
                        "input_type": "activity_input",
                        "validation_passed": True,
                    },
                }
            else:  # ActivityCompleted
                mock_response.json.return_value = {"verdict": "allow"}
            return mock_response

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.post = mock_post
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        # Verify the dataclass was updated in place
        assert input_data.prompt == "[REDACTED]"
        assert input_data.user_id == "user123"

    @pytest.mark.asyncio
    async def test_applies_guardrails_redaction_to_dict_input(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that guardrails redaction is applied to dict input for completed event."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        # Create dict input (not dataclass)
        input_data = {"prompt": "original", "user_id": "user123"}
        mock_input = MagicMock()
        mock_input.args = [input_data]

        call_count = [0]
        captured_payloads = []

        async def mock_post(*args, **kwargs):
            call_count[0] += 1
            captured_payloads.append(kwargs.get("json", {}))
            mock_response = MagicMock()
            mock_response.status_code = 200
            if call_count[0] == 1:
                mock_response.json.return_value = {
                    "verdict": "allow",
                    "guardrails_result": {
                        "redacted_input": [{"prompt": "[REDACTED]", "user_id": "user123"}],
                        "input_type": "activity_input",
                        "validation_passed": True,
                    },
                }
            else:
                mock_response.json.return_value = {"verdict": "allow"}
            return mock_response

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.post = mock_post
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        # The ActivityCompleted event (second call) should have redacted input
        assert len(captured_payloads) >= 2
        completed_payload = captured_payloads[1]
        assert completed_payload["event_type"] == "ActivityCompleted"
        # The activity_input in the completed event should show redacted values
        assert completed_payload["activity_input"] == [{"prompt": "[REDACTED]", "user_id": "user123"}]

    @pytest.mark.asyncio
    async def test_sends_activity_completed_event(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that ActivityCompleted event is sent with input/output."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value={"result": "success"})

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = [{"input": "data"}]

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            await interceptor.execute_activity(mock_input)

        # Verify ActivityCompleted event was sent
        call_args = mock_client_instance.post.call_args
        payload = call_args[1]["json"]
        assert payload["event_type"] == "ActivityCompleted"
        assert payload["activity_input"] == [{"input": "data"}]
        assert payload["activity_output"] == {"result": "success"}
        assert payload["status"] == "completed"

    @pytest.mark.asyncio
    async def test_require_approval_sets_pending_and_raises_retryable(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that REQUIRE_APPROVAL sets pending_approval and raises retryable error."""
        buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="test-run-id",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
        )
        mock_span_processor.get_buffer.return_value = buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=True, hitl_enabled=True)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "require_approval",
            "reason": "Needs human review",
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "ApprovalPending"
            assert exc_info.value.non_retryable is False  # Retryable
            assert buffer.pending_approval is True

    @pytest.mark.asyncio
    async def test_approval_polling_on_retry_when_pending(
        self, mock_span_processor, mock_activity_info
    ):
        """Test approval polling on retry when pending_approval=True."""
        buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="test-run-id",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
            pending_approval=True,
        )
        mock_span_processor.get_buffer.return_value = buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        async def mock_post(url, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "approval" in url:
                mock_response.json.return_value = {"verdict": "allow"}
            else:
                mock_response.json.return_value = {"verdict": "allow"}
            return mock_response

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.post = mock_post
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            result = await interceptor.execute_activity(mock_input)

        assert result == "result"
        assert buffer.pending_approval is False

    @pytest.mark.asyncio
    async def test_approval_rejected_raises_non_retryable(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that rejected approval raises non-retryable error."""
        buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="test-run-id",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
            pending_approval=True,
        )
        mock_span_processor.get_buffer.return_value = buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        async def mock_post(url, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "approval" in url:
                mock_response.json.return_value = {
                    "verdict": "block",
                    "reason": "Request denied by admin",
                }
            return mock_response

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.post = mock_post
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "ApprovalRejected"
            assert exc_info.value.non_retryable is True
            assert "Request denied by admin" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_approval_expired_raises_non_retryable(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that expired approval raises non-retryable error."""
        buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="test-run-id",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
            pending_approval=True,
        )
        mock_span_processor.get_buffer.return_value = buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        async def mock_post(url, *args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 200
            if "approval" in url:
                mock_response.json.return_value = {
                    "verdict": "require_approval",
                    "expired": True,
                }
            return mock_response

        mock_httpx = MagicMock()
        mock_client_instance = MagicMock()
        mock_client_instance.post = mock_post
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "ApprovalExpired"
            assert exc_info.value.non_retryable is True

    # =========================================================================
    # Tests for _send_activity_event()
    # =========================================================================

    @pytest.mark.asyncio
    async def test_send_activity_event_correct_payload(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that _send_activity_event sends correct payload."""
        config = GovernanceConfig()
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "allow",
            "reason": "OK",
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityStarted",
                activity_input=["test"],
            )

        # Verify the call
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://localhost:8086/api/v1/governance/evaluate"
        payload = call_args[1]["json"]
        assert payload["source"] == "workflow-telemetry"
        assert payload["event_type"] == "ActivityStarted"
        assert payload["workflow_id"] == "test-workflow-id"
        assert payload["run_id"] == "test-run-id"
        assert payload["activity_id"] == "test-activity-id"
        assert payload["activity_type"] == "test_activity"
        assert payload["activity_input"] == ["test"]
        assert "timestamp" in payload

        # Verify headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer obx_test_key123"

        # Verify result
        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.ALLOW

    @pytest.mark.asyncio
    async def test_send_activity_event_serializes_extra_fields(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that extra fields are serialized properly."""
        config = GovernanceConfig()
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            # Pass complex extra fields
            extra_data = NestedData(value="test", count=42)
            await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityCompleted",
                activity_output=extra_data,
                spans=[{"name": "span1"}],
            )

        payload = mock_client_instance.post.call_args[1]["json"]
        assert payload["activity_output"] == {"value": "test", "count": 42}
        assert payload["spans"] == [{"name": "span1"}]

    @pytest.mark.asyncio
    async def test_send_activity_event_returns_none_on_fail_open(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that None is returned on API error with fail_open policy."""
        config = GovernanceConfig(on_api_error="fail_open")
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({}, status_code=500)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityStarted",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_send_activity_event_returns_halt_on_fail_closed(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that HALT verdict is returned on API error with fail_closed policy."""
        config = GovernanceConfig(on_api_error="fail_closed")
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({}, status_code=503)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityStarted",
            )

        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.HALT
        assert "Governance API error" in result.reason

    @pytest.mark.asyncio
    async def test_send_activity_event_handles_exception_fail_open(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that exceptions are handled with fail_open policy."""
        config = GovernanceConfig(on_api_error="fail_open")
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection error"))
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityStarted",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_send_activity_event_handles_exception_fail_closed(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that exceptions return HALT with fail_closed policy."""
        config = GovernanceConfig(on_api_error="fail_closed")
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection error"))
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            result = await interceptor._send_activity_event(
                mock_activity_info,
                "ActivityStarted",
            )

        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.HALT
        assert "Connection error" in result.reason

    # =========================================================================
    # Tests for _poll_approval_status()
    # =========================================================================

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_status(
        self, mock_span_processor, governance_config
    ):
        """Test that _poll_approval_status returns approval status dict."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "allow",
            "reason": "Approved by admin",
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.logger = MagicMock()

            result = await interceptor._poll_approval_status(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result == {"verdict": "allow", "reason": "Approved by admin"}

        # Verify the request
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://localhost:8086/api/v1/governance/approval"
        payload = call_args[1]["json"]
        assert payload["workflow_id"] == "wf-123"
        assert payload["run_id"] == "run-456"
        assert payload["activity_id"] == "act-789"

    @pytest.mark.asyncio
    async def test_poll_approval_status_checks_expiration(
        self, mock_span_processor, governance_config
    ):
        """Test that _poll_approval_status checks expiration and sets expired=True."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "require_approval",
            "approval_expiration_time": "2020-01-01T00:00:00Z",  # Past date
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.logger = MagicMock()

            result = await interceptor._poll_approval_status(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result["expired"] is True

    @pytest.mark.asyncio
    async def test_poll_approval_status_handles_various_timestamp_formats(
        self, mock_span_processor, governance_config
    ):
        """Test that _poll_approval_status handles various timestamp formats."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        test_cases = [
            "2020-01-01T00:00:00Z",  # ISO with Z
            "2020-01-01T00:00:00+00:00",  # ISO with offset
            "2020-01-01 00:00:00",  # Space-separated
        ]

        for timestamp in test_cases:
            mock_httpx = MagicMock()
            mock_client, mock_client_instance = create_mock_httpx_client({
                "verdict": "require_approval",
                "approval_expiration_time": timestamp,
            })
            mock_httpx.AsyncClient.return_value = mock_client

            with patch("openbox.activity_interceptor.activity") as mock_activity, \
                 patch.dict(sys.modules, {"httpx": mock_httpx}):
                mock_activity.logger = MagicMock()

                result = await interceptor._poll_approval_status(
                    workflow_id="wf-123",
                    run_id="run-456",
                    activity_id="act-789",
                )

            assert result["expired"] is True, f"Failed for timestamp: {timestamp}"

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_none_on_api_error(
        self, mock_span_processor, governance_config
    ):
        """Test that _poll_approval_status returns None on API error."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({}, status_code=500)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.logger = MagicMock()

            result = await interceptor._poll_approval_status(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_none_on_exception(
        self, mock_span_processor, governance_config
    ):
        """Test that _poll_approval_status returns None on exception."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.logger = MagicMock()

            result = await interceptor._poll_approval_status(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_approval_status_null_expiration_not_expired(
        self, mock_span_processor, governance_config
    ):
        """Test that null/empty expiration time does not set expired."""
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "require_approval",
            "approval_expiration_time": None,  # No expiration
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.logger = MagicMock()

            result = await interceptor._poll_approval_status(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert "expired" not in result or result.get("expired") is not True


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_serialize_value_empty_list(self):
        """Test serializing empty list."""
        assert _serialize_value([]) == []

    def test_serialize_value_empty_dict(self):
        """Test serializing empty dict."""
        assert _serialize_value({}) == {}

    def test_serialize_value_nested_none(self):
        """Test serializing structure with nested None values."""
        data = {"key": None, "nested": {"inner": None}}
        result = _serialize_value(data)
        assert result == {"key": None, "nested": {"inner": None}}

    def test_deep_update_with_none_values(self):
        """Test _deep_update_dataclass with None values in update."""
        data = NestedData(value="original", count=42)
        update = {"value": None}

        _deep_update_dataclass(data, update)

        assert data.value is None
        assert data.count == 42

    @pytest.mark.asyncio
    async def test_execute_activity_with_none_args(
        self, mock_span_processor, mock_activity_info
    ):
        """Test execute_activity with None args."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value="result")

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = None

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            result = await interceptor.execute_activity(mock_input)

        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_activity_handles_activity_exception(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that activity exceptions are properly propagated."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(side_effect=ValueError("Activity failed"))

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({"verdict": "allow"})
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            with pytest.raises(ValueError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert str(exc_info.value) == "Activity failed"

    @pytest.mark.asyncio
    async def test_buffer_verdict_blocks_activity(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that buffer.verdict blocks activity execution."""
        buffer = WorkflowSpanBuffer(
            workflow_id="test-workflow-id",
            run_id="test-run-id",
            workflow_type="TestWorkflow",
            task_queue="test-queue",
            verdict=Verdict.HALT,
            verdict_reason="Workflow halted by policy",
        )
        mock_span_processor.get_buffer.return_value = buffer
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig()
        mock_next = AsyncMock()

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)

            assert exc_info.value.type == "GovernanceStop"
            assert "Workflow halted by policy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_output_redaction_applied(
        self, mock_span_processor, mock_activity_info
    ):
        """Test that output redaction is applied from ActivityCompleted verdict."""
        mock_span_processor.get_buffer.return_value = None
        mock_span_processor.get_verdict.return_value = None

        config = GovernanceConfig(send_activity_start_event=False)
        output_data = ActivityInput(prompt="secret data", user_id="user123")
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(return_value=output_data)

        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=mock_span_processor,
            config=config,
        )

        mock_input = MagicMock()
        mock_input.args = []

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client({
            "verdict": "allow",
            "guardrails_result": {
                "redacted_input": {"prompt": "[REDACTED]", "user_id": "user123", "metadata": {}},
                "input_type": "activity_output",
                "validation_passed": True,
            },
        })
        mock_httpx.AsyncClient.return_value = mock_client

        with patch("openbox.activity_interceptor.activity") as mock_activity, \
             patch("openbox.activity_interceptor.trace") as mock_trace, \
             patch.dict(sys.modules, {"httpx": mock_httpx}):
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()

            mock_span = MagicMock()
            mock_span.get_span_context.return_value.trace_id = 123
            mock_span.get_span_context.return_value.span_id = 456
            mock_tracer = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(return_value=False)
            mock_trace.get_tracer.return_value = mock_tracer

            result = await interceptor.execute_activity(mock_input)

        # Verify output was redacted in place
        assert output_data.prompt == "[REDACTED]"
        assert result.prompt == "[REDACTED]"
