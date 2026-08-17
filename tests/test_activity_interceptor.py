"""Comprehensive tests for the OpenBox SDK activity_interceptor module."""

import asyncio
import base64
import enum as _enum
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from openbox.activity_interceptor import (
    ActivityGovernanceInterceptor,
    _ActivityInterceptor,
    _deep_update_dataclass,
    _rfc3339_now,
    _serialize_value,
)
from openbox.config import GovernanceConfig
from openbox.core_adapter import (
    TemporalFrameworkAdapter,
    build_core_activity_context,
    get_core_context_store,
)
from openbox.governance_state import TemporalGovernanceState
from openbox.patch import GOVERNANCE_PATCH_SCHEMA_VERSION, PatchRequest
from openbox.span_processor import WorkflowSpanProcessor
from openbox.types import EvaluationResult, GovernanceVerdictResponse, Verdict

from .conftest import posted_payload


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
    items: list[str] = field(default_factory=list)


@dataclass
class DataWithList:
    """Dataclass with list of nested dataclasses."""

    entries: list[NestedData] = field(default_factory=list)


@dataclass
class ActivityInput:
    """Sample activity input dataclass for testing redaction."""

    prompt: str = ""
    user_id: str = ""
    metadata: dict = field(default_factory=dict)


class MockTemporalPayload:
    """Mock Temporal Payload object for testing serialization."""

    def __init__(self, data: bytes, metadata: dict | None = None):
        self.data = data
        self.metadata = metadata or {}


class NonSerializableObject:
    """Object that can't be JSON serialized."""

    def __init__(self, value):
        self._value = value

    def __str__(self):
        return f"NonSerializable({self._value})"


@pytest.fixture(autouse=True)
def reset_core_context_store():
    """Isolate the process-wide core ContextStore between tests.

    ``core_adapter`` binds one module-global ContextStore that the interceptor
    reads for the within-activity abort flag. Reset it so a completed-stop /
    abort flag set in one test never bleeds into the next.
    """
    get_core_context_store().clear()
    yield
    get_core_context_store().clear()


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
def state():
    """A real TemporalGovernanceState shared across interceptors."""
    return TemporalGovernanceState()


@pytest.fixture
def governance_config():
    """Create a default GovernanceConfig."""
    return GovernanceConfig()


def make_verdict_client(verdict_response=None, approval_response=None) -> MagicMock:
    """Build a mock GovernanceClient.

    evaluate_event returns ``verdict_response`` (a GovernanceVerdictResponse or
    None); poll_approval returns ``approval_response`` (a dict or None). Both
    default to a plain ALLOW / None so activities run without governance stops.
    """
    client = MagicMock()
    if verdict_response is None:
        verdict_response = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
    client.evaluate_event = AsyncMock(return_value=verdict_response)
    client.poll_approval = AsyncMock(return_value=approval_response)
    return client


def make_interceptor(state, config=None, *, next_result="result", client=None):
    """Build an _ActivityInterceptor with a mock next + mock client."""
    config = config or GovernanceConfig()
    mock_next = AsyncMock()
    mock_next.execute_activity = AsyncMock(return_value=next_result)
    return _ActivityInterceptor(
        next_interceptor=mock_next,
        api_url="http://localhost:8086",
        api_key="obx_test_key123",
        span_processor=WorkflowSpanProcessor(),
        state=state,
        config=config,
        client=client or make_verdict_client(),
    )


def make_input(args=None):
    """Build an ExecuteActivityInput-like mock with empty headers.

    Empty headers => read_session_from_header returns None (no session tag), so
    payload assertions stay free of a spurious multi_agent_session_id.
    """
    mock_input = MagicMock()
    mock_input.args = args if args is not None else []
    mock_input.headers = {}
    return mock_input


def make_patch_request(**overrides) -> PatchRequest:
    """Build a PatchRequest for seeding a completed-hook stop entry
    directly (bypassing the normalizer — these tests target the interceptor's
    OWN priority handling of an already-produced request)."""
    base = dict(
        schema_version=GOVERNANCE_PATCH_SCHEMA_VERSION,
        new_input={"query": "patch"},
        governance_event_id="evt_completed",
        reason="post-hoc patch",
        event_type="ActivityStarted",
        hook_trigger=True,
        hook_stage="completed",
    )
    base.update(overrides)
    return PatchRequest(**base)


def create_mock_httpx_client(response_data, status_code=200):
    """Create a mock httpx async client with specified response.

    Used only by the low-level GovernanceClient HTTP tests
    (_send_activity_event / poll_approval), which exercise the real client's
    transport via a patched httpx module.
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.content = json.dumps(response_data).encode()
    mock_response.json.return_value = response_data

    mock_client_instance = AsyncMock()
    mock_client_instance.post = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.post = mock_client_instance.post
    mock_client.aclose = AsyncMock(return_value=None)
    # The merged client uses the httpx client as an async context manager
    # (``async with client as actual``), so the mock must yield ITSELF on
    # enter — otherwise ``actual`` is an auto-created child mock.
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    return mock_client, mock_client_instance


def patched_activity(mock_activity_info):
    """Patch the interceptor's ``activity`` module with info + logger."""
    ctx = patch("openbox.activity_interceptor.activity")
    mock_activity = ctx.start()
    mock_activity.info.return_value = mock_activity_info
    mock_activity.logger = MagicMock()
    return ctx, mock_activity


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
        from datetime import datetime

        result = _rfc3339_now()

        # Parse the result (handle 3-digit milliseconds)
        # Result is like "2026-02-02T17:35:50.719Z"
        result_no_z = result[:-1]  # Remove Z
        # Pad milliseconds to 6 digits for fromisoformat
        parts = result_no_z.split(".")
        if len(parts) == 2:
            parts[1] = parts[1].ljust(6, "0")
            result_padded = ".".join(parts)
        else:
            result_padded = result_no_z

        result_time = datetime.fromisoformat(result_padded)
        result_time = result_time.replace(tzinfo=UTC)

        now = datetime.now(UTC)

        # The result should be within 1 second of now
        assert abs((now - result_time).total_seconds()) < 1


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


class TestSignalVerdictRunScoping:
    """Signal verdicts are run-scoped: a verdict left by a PRIOR run with the
    same workflow_id must be ignored (and cleared) rather than enforced on a
    new run. This replaces the legacy span-processor stale-buffer/stale-verdict
    cleanup that the interceptor used to do explicitly."""

    def test_current_run_verdict_returned(self):
        state = TemporalGovernanceState()
        state.set_signal_verdict("wf", "run-1", Verdict.BLOCK, "blocked")

        entry = state.get_signal_verdict("wf", "run-1")
        assert entry is not None
        verdict, reason = entry
        assert verdict == Verdict.BLOCK
        assert reason == "blocked"

    def test_stale_run_verdict_ignored_and_cleared(self):
        state = TemporalGovernanceState()
        state.set_signal_verdict("wf", "old-run", Verdict.BLOCK, "old")

        # A new run asks — stale verdict is ignored.
        assert state.get_signal_verdict("wf", "new-run") is None
        # And cleared, so even the original run no longer sees it.
        assert state.get_signal_verdict("wf", "old-run") is None


class TestActivityGovernanceInterceptor:
    """Tests for ActivityGovernanceInterceptor class."""

    def test_initialization(self, state):
        """Test interceptor initialization with all parameters."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=GovernanceConfig(),
        )

        assert interceptor.api_url == "http://localhost:8086"
        assert interceptor.api_key == "obx_test_key123"
        assert interceptor._state is state
        assert isinstance(interceptor.config, GovernanceConfig)

    def test_initialization_with_default_config(self, state):
        """Test interceptor initialization with default config."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
        )

        assert isinstance(interceptor.config, GovernanceConfig)

    def test_api_url_trailing_slash_stripped(self, state):
        """Test that trailing slash is stripped from api_url."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086/",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
        )

        assert interceptor.api_url == "http://localhost:8086"

    def test_api_url_multiple_trailing_slashes_stripped(self, state):
        """Test that multiple trailing slashes are stripped."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086///",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
        )

        assert interceptor.api_url == "http://localhost:8086"

    def test_intercept_activity_returns_activity_interceptor(self, state):
        """Test that intercept_activity returns _ActivityInterceptor."""
        interceptor = ActivityGovernanceInterceptor(
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
        )
        mock_next = MagicMock()

        result = interceptor.intercept_activity(mock_next)

        assert isinstance(result, _ActivityInterceptor)
        # The shared state flows into the per-activity interceptor.
        assert result._state is state


class TestActivityInterceptor:
    """Tests for _ActivityInterceptor class."""

    @pytest.mark.asyncio
    async def test_skips_if_activity_type_in_skip_list(self, state, mock_activity_info):
        """Test that activity is skipped if activity_type is in skip_activity_types."""
        config = GovernanceConfig(skip_activity_types={"test_activity"})
        interceptor = make_interceptor(state, config, next_result="activity_result")

        mock_input = make_input(["arg1"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "activity_result"
        interceptor.next.execute_activity.assert_called_once_with(mock_input)

    @pytest.mark.asyncio
    async def test_checks_pending_signal_verdict_raises_governance_block(
        self, state, mock_activity_info
    ):
        """A SignalReceived BLOCK recorded for this run fails the next activity
        with a non-retryable GovernanceBlock (via _check_pending_verdicts)."""
        state.set_signal_verdict(
            "test-workflow-id", "test-run-id", Verdict.BLOCK, "Blocked by policy"
        )

        interceptor = make_interceptor(state, GovernanceConfig())
        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True
        assert "Governance blocked" in str(exc_info.value)
        # Blocked before running user code.
        interceptor.next.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_signal_verdict_halt_terminates_workflow(
        self, state, mock_activity_info
    ):
        """A SignalReceived HALT recorded for this run terminates the workflow
        (GovernanceHalt) before the activity runs."""
        state.set_signal_verdict(
            "test-workflow-id", "test-run-id", Verdict.HALT, "Workflow halted by policy"
        )

        interceptor = make_interceptor(state, GovernanceConfig())
        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceHalt"
        assert "Workflow halted by policy" in str(exc_info.value)
        interceptor.next.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_activity_started_event_if_enabled(
        self, state, mock_activity_info
    ):
        """Test that ActivityStarted event is sent if send_activity_start_event=True."""
        config = GovernanceConfig(send_activity_start_event=True)
        client = make_verdict_client()
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input(["test_arg"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # Two events: ActivityStarted + ActivityCompleted.
        assert client.evaluate_event.call_count >= 2
        first_payload = client.evaluate_event.call_args_list[0].args[0]
        assert first_payload["event_type"] == "ActivityStarted"
        assert first_payload["activity_input"] == ["test_arg"]

    @pytest.mark.asyncio
    async def test_raises_governance_block_on_block_verdict(
        self, state, mock_activity_info
    ):
        """Test that GovernanceBlock is raised for BLOCK verdict on ActivityStarted."""
        config = GovernanceConfig(send_activity_start_event=True)
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.BLOCK, reason="Policy violation")
        )
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True
        assert "Policy violation" in str(exc_info.value)
        interceptor.next.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_guardrails_validation_failed_if_validation_passed_false(
        self, state, mock_activity_info
    ):
        """Test that GuardrailsValidationFailed is raised when validation_passed=False."""
        config = GovernanceConfig(send_activity_start_event=True)
        verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "allow",
                "guardrails_result": {
                    "redacted_input": {},
                    "input_type": "activity_input",
                    "validation_passed": False,
                    "reasons": [{"reason": "PII detected"}],
                },
            }
        )
        client = make_verdict_client(verdict)
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GuardrailsValidationFailed"
        assert "PII detected" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_applies_guardrails_redaction_to_dataclass_input(
        self, state, mock_activity_info
    ):
        """Test that guardrails redaction is applied to dataclass input in place."""
        config = GovernanceConfig(send_activity_start_event=True)

        started_verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "allow",
                "guardrails_result": {
                    "redacted_input": [
                        {
                            "prompt": "[REDACTED]",
                            "user_id": "user123",
                            "metadata": {},
                        }
                    ],
                    "input_type": "activity_input",
                    "validation_passed": True,
                },
            }
        )
        completed_verdict = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        client = MagicMock()
        client.evaluate_event = AsyncMock(
            side_effect=[started_verdict, completed_verdict]
        )
        client.poll_approval = AsyncMock(return_value=None)
        interceptor = make_interceptor(state, config, client=client)

        input_data = ActivityInput(
            prompt="original prompt with PII",
            user_id="user123",
        )
        mock_input = make_input([input_data])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # The dataclass was updated in place.
        assert input_data.prompt == "[REDACTED]"
        assert input_data.user_id == "user123"

    @pytest.mark.asyncio
    async def test_applies_guardrails_redaction_to_dict_input(
        self, state, mock_activity_info
    ):
        """Test that redacted input flows into the ActivityCompleted event payload."""
        config = GovernanceConfig(send_activity_start_event=True)

        started_verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "allow",
                "guardrails_result": {
                    "redacted_input": [{"prompt": "[REDACTED]", "user_id": "user123"}],
                    "input_type": "activity_input",
                    "validation_passed": True,
                },
            }
        )
        completed_verdict = GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        client = MagicMock()
        client.evaluate_event = AsyncMock(
            side_effect=[started_verdict, completed_verdict]
        )
        client.poll_approval = AsyncMock(return_value=None)
        interceptor = make_interceptor(state, config, client=client)

        input_data = {"prompt": "original", "user_id": "user123"}
        mock_input = make_input([input_data])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # Second event = ActivityCompleted, carrying the redacted input.
        assert client.evaluate_event.call_count >= 2
        completed_payload = client.evaluate_event.call_args_list[1].args[0]
        assert completed_payload["event_type"] == "ActivityCompleted"
        assert completed_payload["activity_input"] == [
            {"prompt": "[REDACTED]", "user_id": "user123"}
        ]

    @pytest.mark.asyncio
    async def test_sends_activity_completed_event(self, state, mock_activity_info):
        """Test that ActivityCompleted event is sent with input/output."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()
        interceptor = make_interceptor(
            state, config, next_result={"result": "success"}, client=client
        )

        mock_input = make_input([{"input": "data"}])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # send_activity_start_event=False, so the only event is ActivityCompleted.
        payload = client.evaluate_event.call_args.args[0]
        assert payload["event_type"] == "ActivityCompleted"
        assert payload["activity_input"] == [{"input": "data"}]
        assert payload["activity_output"] == {"result": "success"}
        assert payload["status"] == "completed"

    @pytest.mark.asyncio
    async def test_require_approval_marks_pending_and_raises_retryable(
        self, state, mock_activity_info
    ):
        """REQUIRE_APPROVAL marks pending approval in state and raises retryable
        ApprovalPending."""
        config = GovernanceConfig(send_activity_start_event=True, hitl_enabled=True)
        client = make_verdict_client(
            GovernanceVerdictResponse(
                verdict=Verdict.REQUIRE_APPROVAL, reason="Needs human review"
            )
        )
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "ApprovalPending"
        assert exc_info.value.non_retryable is False  # Retryable
        assert state.has_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

    @pytest.mark.asyncio
    async def test_approval_polling_on_retry_when_pending(
        self, state, mock_activity_info
    ):
        """On retry with a pending marker, the interceptor polls approval; an
        ALLOW clears the marker and the activity proceeds."""
        state.mark_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        client = make_verdict_client(approval_response={"verdict": "allow"})
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "result"
        client.poll_approval.assert_called_once()
        assert not state.has_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

    @pytest.mark.asyncio
    async def test_approval_rejected_raises_non_retryable(
        self, state, mock_activity_info
    ):
        """A polled BLOCK verdict (human rejection) raises non-retryable
        ApprovalRejected."""
        state.mark_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        client = make_verdict_client(
            approval_response={
                "verdict": "block",
                "reason": "Request denied by admin",
            }
        )
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "ApprovalRejected"
        assert exc_info.value.non_retryable is True
        assert "Request denied by admin" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_approval_expired_raises_non_retryable(
        self, state, mock_activity_info
    ):
        """A polled response flagged expired raises non-retryable ApprovalExpired."""
        state.mark_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        client = make_verdict_client(
            approval_response={"verdict": "require_approval", "expired": True}
        )
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "ApprovalExpired"
        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    async def test_approval_patch_raises_governance_patch(
        self, state, mock_activity_info
    ):
        """A non-expired exact BLOCK with a valid patch on the approval
        poll requests a workflow restart instead of the plain ApprovalRejected."""
        state.mark_pending_approval(
            "test-workflow-id", "test-run-id", "test-activity-id"
        )

        config = GovernanceConfig(hitl_enabled=True, send_activity_start_event=False)
        client = make_verdict_client(
            approval_response={
                "verdict": "block",
                "reason": "admin patch",
                "id": "evt_a",
                "patch": {"new_input": {"k": "v"}},
            }
        )
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernancePatch"
        assert exc_info.value.non_retryable is True
        details = exc_info.value.details[0]
        assert details["new_input"] == {"k": "v"}
        assert details["governance_event_id"] == "evt_a"
        assert details["event_type"] == "ActivityStarted"

    @pytest.mark.asyncio
    async def test_send_activity_event_correct_payload(self, state, mock_activity_info):
        """_send_activity_event builds the correct payload and posts it via the
        real GovernanceClient (signed transport -> content= bytes)."""
        config = GovernanceConfig()
        # Real client so the HTTP transport/headers are genuinely exercised.
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client(
            {"verdict": "allow", "reason": "OK"}
        )
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityStarted",
                    activity_input=["test"],
                )
        finally:
            ctx.stop()

        # Verify the request target + payload.
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://localhost:8086/api/v1/governance/evaluate"
        payload = posted_payload(call_args)
        assert payload["source"] == "workflow-telemetry"
        assert payload["event_type"] == "ActivityStarted"
        assert payload["workflow_id"] == "test-workflow-id"
        assert payload["run_id"] == "test-run-id"
        assert payload["activity_id"] == "test-activity-id"
        assert payload["activity_type"] == "test_activity"
        assert payload["activity_input"] == ["test"]
        assert "timestamp" in payload

        # Verify auth header.
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer obx_test_key123"

        # Verify parsed result.
        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.ALLOW

    @pytest.mark.asyncio
    async def test_send_activity_event_serializes_extra_fields(
        self, state, mock_activity_info
    ):
        """Test that extra fields are serialized properly."""
        config = GovernanceConfig()
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client(
            {"verdict": "allow"}
        )
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                extra_data = NestedData(value="test", count=42)
                await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityCompleted",
                    activity_output=extra_data,
                    spans=[{"name": "span1"}],
                )
        finally:
            ctx.stop()

        payload = posted_payload(mock_client_instance.post.call_args)
        assert payload["activity_output"] == {"value": "test", "count": 42}
        assert payload["spans"] == [{"name": "span1"}]

    @pytest.mark.asyncio
    async def test_send_activity_event_returns_none_on_fail_open(
        self, state, mock_activity_info
    ):
        """Test that None is returned on API error with fail_open policy."""
        config = GovernanceConfig(on_api_error="fail_open")
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, _ = create_mock_httpx_client({}, status_code=500)
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityStarted",
                )
        finally:
            ctx.stop()

        assert result is None

    @pytest.mark.asyncio
    async def test_send_activity_event_returns_halt_on_fail_closed(
        self, state, mock_activity_info
    ):
        """Test that HALT verdict is returned on API error with fail_closed policy."""
        config = GovernanceConfig(on_api_error="fail_closed")
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client, _ = create_mock_httpx_client({}, status_code=503)
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityStarted",
                )
        finally:
            ctx.stop()

        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.HALT
        assert "Governance API error" in result.reason

    @pytest.mark.asyncio
    async def test_send_activity_event_handles_exception_fail_open(
        self, state, mock_activity_info
    ):
        """Test that exceptions are handled with fail_open policy."""
        config = GovernanceConfig(on_api_error="fail_open")
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
        mock_client.aclose = AsyncMock(return_value=None)
        # The merged client enters the transport as an async context manager.
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityStarted",
                )
        finally:
            ctx.stop()

        assert result is None

    @pytest.mark.asyncio
    async def test_send_activity_event_handles_exception_fail_closed(
        self, state, mock_activity_info
    ):
        """Test that exceptions return HALT with fail_closed policy."""
        config = GovernanceConfig(on_api_error="fail_closed")
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection error"))
        mock_client.aclose = AsyncMock(return_value=None)
        # The merged client enters the transport as an async context manager.
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_httpx.AsyncClient.return_value = mock_client

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._send_activity_event(
                    mock_activity_info,
                    "ActivityStarted",
                )
        finally:
            ctx.stop()

        assert isinstance(result, GovernanceVerdictResponse)
        assert result.verdict == Verdict.HALT
        assert "Connection error" in result.reason

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_status(self, state, governance_config):
        """Test that _client.poll_approval returns approval status dict."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, mock_client_instance = create_mock_httpx_client(
            {"verdict": "allow", "reason": "Approved by admin"}
        )
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await interceptor._client.poll_approval(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result == {"verdict": "allow", "reason": "Approved by admin"}

        # Verify the request target + payload.
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://localhost:8086/api/v1/governance/approval"
        payload = posted_payload(call_args)
        assert payload["workflow_id"] == "wf-123"
        assert payload["run_id"] == "run-456"
        assert payload["activity_id"] == "act-789"

    @pytest.mark.asyncio
    async def test_poll_approval_status_checks_expiration(
        self, state, governance_config
    ):
        """Test that _client.poll_approval checks expiration and sets expired=True."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, _ = create_mock_httpx_client(
            {
                "verdict": "require_approval",
                "approval_expiration_time": "2020-01-01T00:00:00Z",  # Past date
            }
        )
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await interceptor._client.poll_approval(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result["expired"] is True

    @pytest.mark.asyncio
    async def test_poll_approval_status_handles_various_timestamp_formats(
        self, state, governance_config
    ):
        """Test that _client.poll_approval handles various timestamp formats."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        test_cases = [
            "2020-01-01T00:00:00Z",  # ISO with Z
            "2020-01-01T00:00:00+00:00",  # ISO with offset
            "2020-01-01 00:00:00",  # Space-separated
        ]

        for timestamp in test_cases:
            mock_httpx = MagicMock()
            mock_client, _ = create_mock_httpx_client(
                {
                    "verdict": "require_approval",
                    "approval_expiration_time": timestamp,
                }
            )
            mock_httpx.AsyncClient.return_value = mock_client

            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                result = await interceptor._client.poll_approval(
                    workflow_id="wf-123",
                    run_id="run-456",
                    activity_id="act-789",
                )

            assert result["expired"] is True, f"Failed for timestamp: {timestamp}"

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_none_on_api_error(
        self, state, governance_config
    ):
        """Test that _client.poll_approval returns None on API error."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, _ = create_mock_httpx_client({}, status_code=500)
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await interceptor._client.poll_approval(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_approval_status_returns_none_on_exception(
        self, state, governance_config
    ):
        """Test that _client.poll_approval returns None on exception."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Network error"))
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await interceptor._client.poll_approval(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_poll_approval_status_null_expiration_not_expired(
        self, state, governance_config
    ):
        """Test that null/empty expiration time does not set expired."""
        interceptor = _ActivityInterceptor(
            next_interceptor=AsyncMock(),
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=governance_config,
        )

        mock_httpx = MagicMock()
        mock_client, _ = create_mock_httpx_client(
            {
                "verdict": "require_approval",
                "approval_expiration_time": None,  # No expiration
            }
        )
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = await interceptor._client.poll_approval(
                workflow_id="wf-123",
                run_id="run-456",
                activity_id="act-789",
            )

        assert "expired" not in result or result.get("expired") is not True


class TestActivityLifecyclePatchRestart:
    """``_enforce_verdict`` checks the BLOCK-with-patch normalizer BEFORE the
    generic BLOCK/HALT/guardrails mapping, for both the ActivityStarted and
    ActivityCompleted lifecycle verdicts (real activity events — hook_trigger
    stays False)."""

    @pytest.mark.asyncio
    async def test_started_block_with_valid_patch_raises_governance_patch(
        self, state, mock_activity_info
    ):
        config = GovernanceConfig(send_activity_start_event=True)
        verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "block",
                "reason": "patch with corrected input",
                "governance_event_id": "evt_1",
                "patch": {"new_input": {"query": "corrected"}},
            }
        )
        client = make_verdict_client(verdict)
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernancePatch"
        assert exc_info.value.non_retryable is True
        details = exc_info.value.details[0]
        assert details["schema_version"] == GOVERNANCE_PATCH_SCHEMA_VERSION
        assert details["new_input"] == {"query": "corrected"}
        assert details["governance_event_id"] == "evt_1"
        assert details["event_type"] == "ActivityStarted"
        assert details["hook_trigger"] is False
        # Blocked before running user code.
        interceptor.next.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_completed_block_with_valid_patch_raises_governance_patch(
        self, state, mock_activity_info
    ):
        """ActivityCompleted verdict with a patch raises the versioned restart
        error tagged with event_type=ActivityCompleted; new_input=None is the
        valid 'reuse current input' directive."""
        config = GovernanceConfig(send_activity_start_event=False)
        verdict = GovernanceVerdictResponse.from_dict(
            {"verdict": "block", "patch": {"new_input": None}}
        )
        client = make_verdict_client(verdict)
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernancePatch"
        details = exc_info.value.details[0]
        assert details["new_input"] is None
        assert details["event_type"] == "ActivityCompleted"
        assert details["hook_trigger"] is False

    @pytest.mark.asyncio
    async def test_plain_block_without_patch_still_raises_governance_block(
        self, state, mock_activity_info
    ):
        """No patch -> unchanged plain GovernanceBlock mapping (regression
        guard for the patch-first check added in front of enforce_verdict)."""
        config = GovernanceConfig(send_activity_start_event=True)
        verdict = GovernanceVerdictResponse.from_dict(
            {"verdict": "block", "reason": "no patch here"}
        )
        client = make_verdict_client(verdict)
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceBlock"
        assert exc_info.value.non_retryable is True

    @pytest.mark.asyncio
    async def test_halt_with_synthetic_patch_never_produces_governance_patch(
        self, state, mock_activity_info
    ):
        """Defense-in-depth: a HALT verdict carrying a patch never
        produces a patch request — the base normalizer gates strictly on
        verdict == BLOCK, so HALT always terminates instead."""
        config = GovernanceConfig(send_activity_start_event=True)
        verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "halt",
                "reason": "emergency",
                "patch": {"new_input": "x"},
            }
        )
        client = make_verdict_client(verdict)
        interceptor = make_interceptor(state, config, client=client)

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceHalt"


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
    async def test_execute_activity_with_none_args(self, state, mock_activity_info):
        """Test execute_activity with None args."""
        config = GovernanceConfig(send_activity_start_event=False)
        interceptor = make_interceptor(state, config)

        mock_input = make_input(None)  # args = None

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_activity_handles_activity_exception(
        self, state, mock_activity_info
    ):
        """Test that activity exceptions are properly propagated."""
        config = GovernanceConfig(send_activity_start_event=False)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(
            side_effect=ValueError("Activity failed")
        )
        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
            client=make_verdict_client(),
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with pytest.raises(ValueError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert str(exc_info.value) == "Activity failed"

    @pytest.mark.asyncio
    async def test_completed_hook_halt_terminates_workflow(
        self, state, mock_activity_info
    ):
        """A completed-hook HALT recorded by the adapter reaches Temporal's
        terminate path after user code returns (GovernanceHalt)."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()
        interceptor = make_interceptor(state, config, client=client)

        # The base adapter records a completed-hook stop run-scoped; simulate that
        # by seeding the state the way on_completed_hook_result would.
        state.record_completed_stop(
            "test-workflow-id",
            "test-run-id",
            "test-activity-id",
            Verdict.HALT,
            "Workflow halted by policy",
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceHalt"
        assert "Workflow halted by policy" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_completed_hook_block_skips_completed_event(
        self, state, mock_activity_info
    ):
        """A completed-hook BLOCK recorded by the adapter suppresses the duplicate
        ActivityCompleted event (the operation already ran)."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()
        interceptor = make_interceptor(state, config, client=client)

        state.record_completed_stop(
            "test-workflow-id",
            "test-run-id",
            "test-activity-id",
            Verdict.BLOCK,
            "post-hoc block",
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "result"
        # No ActivityCompleted event sent (aborted by hook governance).
        client.evaluate_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_base_abort_flag_skips_completed_event(
        self, state, mock_activity_info
    ):
        """A within-activity abort flag (started-hook BLOCK the user swallowed)
        suppresses the ActivityCompleted event and is cleared afterward."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()

        async def mark_abort_then_return(input):
            # Simulate a base started-hook BLOCK the user code caught + swallowed.
            get_core_context_store().mark_activity_aborted(
                "test-workflow-id", "test-activity-id"
            )
            return "result"

        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(side_effect=mark_abort_then_return)
        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=config,
            client=client,
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "result"
        client.evaluate_event.assert_not_called()
        # Abort flag cleared so it doesn't bleed into a retry.
        assert not get_core_context_store().is_activity_aborted(
            "test-workflow-id", "test-activity-id"
        )

    @pytest.mark.asyncio
    async def test_output_redaction_applied(self, state, mock_activity_info):
        """Test that output redaction is applied from ActivityCompleted verdict."""
        config = GovernanceConfig(send_activity_start_event=False)
        output_data = ActivityInput(prompt="secret data", user_id="user123")
        completed_verdict = GovernanceVerdictResponse.from_dict(
            {
                "verdict": "allow",
                "guardrails_result": {
                    "redacted_input": {
                        "prompt": "[REDACTED]",
                        "user_id": "user123",
                        "metadata": {},
                    },
                    "input_type": "activity_output",
                    "validation_passed": True,
                },
            }
        )
        client = make_verdict_client(completed_verdict)
        interceptor = make_interceptor(
            state, config, next_result=output_data, client=client
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # Output redacted in place + returned.
        assert output_data.prompt == "[REDACTED]"
        assert result.prompt == "[REDACTED]"


class TestActivityMultiAgentSession:
    """Activity interceptor reads the session header and tags activity events,
    and binds the session id onto the core ActivityContext so hook events
    inherit it."""

    def _converter(self):
        from temporalio.converter import default as _default_converter

        return _default_converter().payload_converter

    async def _run(self, state, mock_activity_info, headers):
        conv = self._converter()
        client = make_verdict_client(verdict_response=None)
        # verdict_response None => evaluate_event returns None (no governance stop).
        client.evaluate_event = AsyncMock(return_value=None)
        interceptor = make_interceptor(state, GovernanceConfig(), client=client)

        mock_input = MagicMock()
        mock_input.args = []
        mock_input.headers = headers

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()
            mock_activity.payload_converter.return_value = conv

            await interceptor.execute_activity(mock_input)

        event_payloads = {
            call.args[0]["event_type"]: call.args[0]
            for call in client.evaluate_event.call_args_list
        }
        return event_payloads

    @pytest.mark.asyncio
    async def test_events_tagged_when_header_present(self, state, mock_activity_info):
        """Header present -> ActivityStarted + ActivityCompleted carry the session id."""
        conv = self._converter()
        from openbox.multi_agent import HEADER_KEY

        headers = {HEADER_KEY: conv.to_payload("sess-act")}
        payloads = await self._run(state, mock_activity_info, headers)

        assert payloads["ActivityStarted"]["multi_agent_session_id"] == "sess-act"
        assert payloads["ActivityCompleted"]["multi_agent_session_id"] == "sess-act"

    @pytest.mark.asyncio
    async def test_events_not_tagged_when_header_absent(
        self, state, mock_activity_info
    ):
        """Header absent -> multi_agent_session_id omitted from both events."""
        payloads = await self._run(state, mock_activity_info, {})

        assert "multi_agent_session_id" not in payloads["ActivityStarted"]
        assert "multi_agent_session_id" not in payloads["ActivityCompleted"]

    @pytest.mark.asyncio
    async def test_core_context_carries_session_id_for_hooks(
        self, state, mock_activity_info
    ):
        """The session id from the header is bound onto the core ActivityContext
        so base hook events inherit it (replaces the legacy span-processor
        activity-context stash)."""
        conv = self._converter()
        from openbox.multi_agent import HEADER_KEY

        observed = {}

        async def capture_context(input):
            ctx = get_core_context_store().current_activity_context()
            observed["session_id"] = ctx.multi_agent_session_id
            return "activity_result"

        client = make_verdict_client()
        client.evaluate_event = AsyncMock(return_value=None)
        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(side_effect=capture_context)
        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=GovernanceConfig(),
            client=client,
        )

        mock_input = MagicMock()
        mock_input.args = []
        mock_input.headers = {HEADER_KEY: conv.to_payload("sess-hook")}

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()
            mock_activity.payload_converter.return_value = conv

            await interceptor.execute_activity(mock_input)

        assert observed["session_id"] == "sess-hook"


class TestCoreContextBindingCarriesPolicyFields:
    """The core ActivityContext bound around execution must carry the policy
    context base hook payloads read (activity_input / multi_agent_session_id)."""

    @pytest.fixture
    def mock_activity_info(self):
        info = MagicMock()
        info.workflow_id = "test-workflow-id"
        info.workflow_run_id = "test-run-id"
        info.workflow_type = "TestWorkflow"
        info.activity_id = "test-activity-id"
        info.activity_type = "test_activity"
        info.task_queue = "test-queue"
        info.attempt = 1
        return info

    @pytest.mark.asyncio
    async def test_run_activity_binds_input_and_session_id(
        self, state, mock_activity_info
    ):
        observed = {}

        async def capture_context(input):
            ctx = get_core_context_store().current_activity_context()
            observed["activity_input"] = ctx.activity_input
            observed["session_id"] = ctx.multi_agent_session_id
            return "ok"

        mock_next = AsyncMock()
        mock_next.execute_activity = AsyncMock(side_effect=capture_context)
        interceptor = _ActivityInterceptor(
            next_interceptor=mock_next,
            api_url="http://localhost:8086",
            api_key="obx_test_key123",
            span_processor=WorkflowSpanProcessor(),
            state=state,
            config=GovernanceConfig(),
            client=make_verdict_client(),
        )

        with patch("openbox.activity_interceptor.activity") as mock_activity:
            mock_activity.info.return_value = mock_activity_info
            mock_activity.logger = MagicMock()
            result, status, *_ = await interceptor._run_activity(
                MagicMock(args=["a1"]),
                mock_activity_info,
                activity_input=["serialized-input"],
                session_id="sid-42",
            )

        assert result == "ok"
        assert status == "completed"
        assert observed["activity_input"] == ["serialized-input"]
        assert observed["session_id"] == "sid-42"
        # Reset after the scope exits — no leak.
        assert get_core_context_store().current_activity_context() is None


# Completed-hook HALT must terminate even when the activity also raises (H1)


class TestCompletedHaltReachesTerminateOnActivityRaise:
    """A completed-hook HALT is a kill-switch: it must reach the terminate path
    even when the activity itself raises AFTER the hook recorded the stop (so
    _handle_completion is skipped). Failing the activity does not halt the run."""

    _WF, _RUN, _ACT = "test-workflow-id", "test-run-id", "test-activity-id"

    async def test_completed_halt_terminates_despite_activity_exception(
        self, mock_activity_info
    ):
        state = TemporalGovernanceState()
        interceptor = make_interceptor(state)  # ActivityStarted -> ALLOW

        async def raising_next(_input):
            # Simulate a completed hook recording HALT during user code, then the
            # activity raising for an unrelated reason.
            state.record_completed_stop(
                self._WF, self._RUN, self._ACT, Verdict.HALT, "kill switch"
            )
            raise RuntimeError("activity boom")

        interceptor.next.execute_activity = raising_next
        terminate = AsyncMock()
        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch(
                "openbox.activity_interceptor._terminate_workflow_for_halt", terminate
            ):
                with pytest.raises(RuntimeError, match="activity boom"):
                    await interceptor.execute_activity(make_input())
        finally:
            ctx.stop()

        terminate.assert_awaited_once_with(self._WF, "kill switch")
        # Consumed on the exception path — nothing stranded.
        assert state.take_completed_stop(self._WF, self._RUN, self._ACT) is None

    async def test_completed_block_on_raise_does_not_terminate_but_is_cleared(
        self, mock_activity_info
    ):
        state = TemporalGovernanceState()
        interceptor = make_interceptor(state)

        async def raising_next(_input):
            state.record_completed_stop(
                self._WF, self._RUN, self._ACT, Verdict.BLOCK, "no"
            )
            raise RuntimeError("boom")

        interceptor.next.execute_activity = raising_next
        terminate = AsyncMock()
        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch(
                "openbox.activity_interceptor._terminate_workflow_for_halt", terminate
            ):
                with pytest.raises(RuntimeError):
                    await interceptor.execute_activity(make_input())
        finally:
            ctx.stop()

        terminate.assert_not_awaited()  # BLOCK never terminates
        assert state.take_completed_stop(self._WF, self._RUN, self._ACT) is None


class TestCompletedHookPatchPriority:
    """Completed-hook priority: HALT > BLOCK with patch > original activity
    exception > plain completed BLOCK — on both the success path
    (_handle_completion) and the exception path (_consume_completed_halt)."""

    _WF, _RUN, _ACT = "test-workflow-id", "test-run-id", "test-activity-id"

    # ---- success path (_handle_completion) --------------------------------

    @pytest.mark.asyncio
    async def test_success_patch_request_raises_after_user_code_returns(
        self, state, mock_activity_info
    ):
        """A patch completed-hook stop raises GovernancePatch
        AFTER the activity returned, replacing the skip-completed-event path."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()
        interceptor = make_interceptor(state, config, client=client)

        req = make_patch_request()
        state.record_completed_stop(
            self._WF, self._RUN, self._ACT, Verdict.BLOCK, "post-hoc patch", req
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernancePatch"
        assert exc_info.value.non_retryable is True
        assert exc_info.value.details[0] == req.to_dict()
        # No duplicate ActivityCompleted event sent — the operation already ran.
        client.evaluate_event.assert_not_called()
        # Consumed on the success path — nothing stranded.
        assert state.take_completed_stop(self._WF, self._RUN, self._ACT) is None

    @pytest.mark.asyncio
    async def test_success_halt_wins_over_a_synthetic_patch_request(
        self, state, mock_activity_info
    ):
        """HALT is checked before the patch request regardless of what else
        the stop entry carries — the real normalizer never pairs the two (HALT
        never yields a patch directive), but the interceptor's own priority
        order must not depend on that."""
        config = GovernanceConfig(send_activity_start_event=False)
        client = make_verdict_client()
        interceptor = make_interceptor(state, config, client=client)

        state.record_completed_stop(
            self._WF,
            self._RUN,
            self._ACT,
            Verdict.HALT,
            "kill switch",
            make_patch_request(),
        )

        mock_input = make_input([])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceHalt"

    # ---- exception path (_consume_completed_halt) --------------------------

    async def test_exception_path_patch_request_replaces_original_exception(
        self, mock_activity_info
    ):
        """The activity itself raises, but a patch completed-hook stop was
        recorded during user code — the patch request wins and REPLACES
        the original exception (governance's remediation route takes over)."""
        state = TemporalGovernanceState()
        interceptor = make_interceptor(state)  # ActivityStarted -> ALLOW

        req = make_patch_request()

        async def raising_next(_input):
            state.record_completed_stop(
                self._WF, self._RUN, self._ACT, Verdict.BLOCK, "patch me", req
            )
            raise RuntimeError("activity boom")

        interceptor.next.execute_activity = raising_next
        ctx, _ = patched_activity(mock_activity_info)
        try:
            from temporalio.exceptions import ApplicationError

            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(make_input())
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernancePatch"
        assert exc_info.value.non_retryable is True
        assert exc_info.value.details[0] == req.to_dict()
        # Consumed on the exception path — nothing stranded.
        assert state.take_completed_stop(self._WF, self._RUN, self._ACT) is None

    async def test_exception_path_halt_wins_over_synthetic_patch_request(
        self, mock_activity_info
    ):
        """Same defense-in-depth as the success-path equivalent: HALT still
        terminates (and the original exception still re-raises after) even if
        the stop entry also carries a request."""
        state = TemporalGovernanceState()
        interceptor = make_interceptor(state)

        async def raising_next(_input):
            state.record_completed_stop(
                self._WF,
                self._RUN,
                self._ACT,
                Verdict.HALT,
                "kill switch",
                make_patch_request(),
            )
            raise RuntimeError("activity boom")

        interceptor.next.execute_activity = raising_next
        terminate = AsyncMock()
        ctx, _ = patched_activity(mock_activity_info)
        try:
            with patch(
                "openbox.activity_interceptor._terminate_workflow_for_halt", terminate
            ):
                with pytest.raises(RuntimeError, match="activity boom"):
                    await interceptor.execute_activity(make_input())
        finally:
            ctx.stop()

        terminate.assert_awaited_once_with(self._WF, "kill switch")
        assert state.take_completed_stop(self._WF, self._RUN, self._ACT) is None


# ─── CONSTRAIN sandbox routing ───────────────────────────────────────────────

# The private governed-dispatcher package is not a dependency of this repo, so
# the routing tests inject a minimal stand-in into sys.modules. The interceptor
# only needs the enums for disposition/directive comparisons, the decision
# parser, and the command value object.


class _FakeDirective(_enum.Enum):
    NONE = "none"
    HALT = "halt"


class _FakeDisposition(_enum.Enum):
    EXECUTED_ON_HOST = "executed_on_host"
    EXECUTED_IN_SANDBOX = "executed_in_sandbox"
    EXECUTION_INDETERMINATE = "execution_indeterminate"
    NOT_EXECUTED = "not_executed"


class _FakeGovernanceDecision:
    @staticmethod
    def parse(value):
        return value


class _FakeGovernedCommand:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_governed_dispatcher(monkeypatch):
    """Install a fake ``openbox_sandbox.dispatcher`` for sandbox-routing tests."""
    package = ModuleType("openbox_sandbox")
    package.__path__ = []
    dispatcher = ModuleType("openbox_sandbox.dispatcher")
    dispatcher.Directive = _FakeDirective
    dispatcher.Disposition = _FakeDisposition
    dispatcher.GovernanceDecision = _FakeGovernanceDecision
    dispatcher.GovernedCommand = _FakeGovernedCommand
    monkeypatch.setitem(sys.modules, "openbox_sandbox", package)
    monkeypatch.setitem(sys.modules, "openbox_sandbox.dispatcher", dispatcher)
    return dispatcher


def make_mock_sandbox(dispatch_result=None):
    """Build a mock TemporalSandboxConfig-like object for the interceptor."""
    from unittest.mock import MagicMock

    sandbox = MagicMock()
    sandbox.timeout_seconds = 30
    sandbox.heartbeat_interval_seconds = 0.05
    sandbox.completion_events = True
    sandbox.otel_bridge = None
    sandbox.profiles.derive.return_value = ("/usr/bin/reconcile", "b-1")
    sandbox.profiles.profile_fingerprint.return_value = "f" * 64
    sandbox.profiles.parse_result.return_value = None
    sandbox.heartbeat_sink.bind.return_value.__aenter__ = AsyncMock(
        return_value=None
    )
    sandbox.heartbeat_sink.bind.return_value.__aexit__ = AsyncMock(
        return_value=None
    )
    sandbox.dispatcher.dispatch_with_decision = AsyncMock(
        return_value=dispatch_result
    )
    # A real dispatcher with a governance config posts its own completion;
    # the test mock has no dispatcher config, so the interceptor wrapper owns
    # the ActivityCompleted event.
    sandbox.dispatcher._config = None
    return sandbox


def make_sandbox_dispatch_result():
    """Build the minimal terminal dispatch result the interceptor maps."""
    execution = SimpleNamespace(
        sandbox_id="sandbox-1",
        exit_code=0,
        stdout=b'{"rows": 7}',
        stderr=b"",
        timeout_status=SimpleNamespace(value="completed_within_timeout"),
        cleanup_status=SimpleNamespace(value="deleted"),
    )
    return SimpleNamespace(
        disposition=_FakeDisposition.EXECUTED_IN_SANDBOX,
        directive=_FakeDirective.NONE,
        error=None,
        execution=execution,
    )


def make_constrain_response(*, profile_id=None):
    return GovernanceVerdictResponse(
        verdict=Verdict.CONSTRAIN,
        reason="run in sandbox",
        policy_id="policy-1",
        risk_score=0.7,
        profile_id=profile_id,
    )


class TestConstrainedActivityRouting:
    """CONSTRAIN routes the user's activity into the sandbox transparently."""

    @pytest.mark.asyncio
    async def test_constrain_with_sandbox_routes_and_returns_sandbox_result(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        """A CONSTRAIN ActivityStarted verdict executes the derived command
        through the sandbox dispatcher and returns the bounded result without
        running the user's activity on the host."""
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        client = make_verdict_client(make_constrain_response())
        interceptor = make_interceptor(
            state, GovernanceConfig(), client=client
        )
        interceptor._sandbox = sandbox
        mock_input = make_input(
            [{"profile_id": "reconcile", "arguments": {"batch_id": "b-1"}}]
        )

        ctx, mock_activity = patched_activity(mock_activity_info)
        try:
            mock_activity.wait_for_cancelled.return_value = asyncio.Future()
            from openbox.sandbox.types import GovernedCommandActivityResult

            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        # Host execution never ran.
        interceptor.next.execute_activity.assert_not_called()
        # The command was derived from the activity input and dispatched with
        # the CONSTRAIN decision.
        sandbox.dispatcher.dispatch_with_decision.assert_awaited_once()
        call_args = sandbox.dispatcher.dispatch_with_decision.await_args
        command, decision = call_args.args
        assert command.kwargs["profile_id"] == "reconcile"
        assert command.kwargs["argv"] == ("/usr/bin/reconcile", "b-1")
        assert command.kwargs["arguments"] == {"batch_id": "b-1"}
        assert command.kwargs["workflow_id"] == "test-workflow-id"
        assert command.kwargs["run_id"] == "test-run-id"
        assert command.kwargs["activity_id"] == "test-activity-id"
        assert command.kwargs["timeout_seconds"] == 30
        assert decision["verdict"] == "constrain"
        assert decision["policy_id"] == "policy-1"
        # The bounded sandbox result is returned to the caller.
        assert isinstance(result, GovernedCommandActivityResult)
        assert result.profile_id == "reconcile"
        assert result.disposition == "executed_in_sandbox"
        assert result.exit_code == 0
        assert result.stdout_bytes == len(b'{"rows": 7}')
        # ActivityStarted + the wrapper's ActivityCompleted were both posted.
        assert client.evaluate_event.await_count == 2

    @pytest.mark.asyncio
    async def test_constrain_without_sandbox_keeps_unsupported_error(
        self, state, mock_activity_info
    ):
        """Without a sandbox configuration the CONSTRAIN error contract is
        preserved (the interceptor cannot route anywhere)."""
        from temporalio.exceptions import ApplicationError

        client = make_verdict_client(make_constrain_response())
        interceptor = make_interceptor(state, GovernanceConfig(), client=client)
        mock_input = make_input(["anything"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            with pytest.raises(ApplicationError) as exc_info:
                await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert exc_info.value.type == "GovernanceConstrainUnsupported"
        assert exc_info.value.non_retryable is True
        interceptor.next.execute_activity.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_with_sandbox_runs_activity_on_host(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        """An ALLOW verdict keeps the natural path: the user's activity runs
        on the host and the sandbox dispatcher is never consulted."""
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        interceptor = make_interceptor(
            state, GovernanceConfig(), next_result="host_result", client=client
        )
        interceptor._sandbox = sandbox
        mock_input = make_input(["arg1"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "host_result"
        interceptor.next.execute_activity.assert_awaited_once_with(mock_input)
        sandbox.dispatcher.dispatch_with_decision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_started_hook_constrain_dispatches_through_adapter_once(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        client = make_verdict_client(GovernanceVerdictResponse(verdict=Verdict.ALLOW))
        client.evaluate_event = AsyncMock(
            side_effect=[
                GovernanceVerdictResponse(verdict=Verdict.ALLOW),
                make_constrain_response(profile_id="post-batch"),
            ]
        )
        interceptor = make_interceptor(state, GovernanceConfig(), client=client)
        interceptor._sandbox = sandbox
        adapter = TemporalFrameworkAdapter(state, constrain_handler=interceptor)
        started_result = EvaluationResult.from_dict(
            {
                "verdict": "constrain",
                "reason": "behavior rule matched",
                "policy_id": "policy-started",
                "age_result": {"profile_id": "post-batch"},
            }
        )

        transport_attempts = []

        async def transport(request):
            transport_attempts.append(request)
            return httpx.Response(200)

        http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))

        async def run_host_activity(input):
            await adapter.handle_constrain(
                started_result,
                build_core_activity_context(mock_activity_info, list(input.args)),
            )
            adapter.raise_hook_blocked(started_result)
            # The intercepted host request is after the hook stop and must never
            # reach even an in-memory transport.
            await http_client.get("https://host-action.test/intercepted")
            return {"activity": "host_result"}

        interceptor.next.execute_activity.side_effect = run_host_activity
        mock_input = make_input(["arg1"])

        ctx, mock_activity = patched_activity(mock_activity_info)
        try:
            mock_activity.wait_for_cancelled.return_value = asyncio.Future()
            result = await interceptor.execute_activity(mock_input)
        finally:
            await http_client.aclose()
            ctx.stop()

        interceptor.next.execute_activity.assert_awaited_once_with(mock_input)
        assert transport_attempts == []
        sandbox.dispatcher.dispatch_with_decision.assert_awaited_once()
        command, decision = sandbox.dispatcher.dispatch_with_decision.await_args.args
        assert command.kwargs["profile_id"] == "post-batch"
        assert decision["verdict"] == "constrain"
        assert decision["policy_id"] == "policy-started"
        assert result["activity_result"] is None
        assert result["sandbox_execution"]["status"] == "completed"
        assert result["sandbox_execution"]["profile_id"] == "post-batch"
        # ActivityCompleted returned the same profile, but the activity-scoped
        # once-guard reused the started-hook dispatch and its outcome.
        assert client.evaluate_event.await_count == 2

    @pytest.mark.asyncio
    async def test_started_sync_hook_constrain_dispatches_through_adapter_once(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        interceptor = make_interceptor(state, GovernanceConfig())
        interceptor._sandbox = sandbox
        adapter = TemporalFrameworkAdapter(state, constrain_handler=interceptor)
        started_result = EvaluationResult.from_dict(
            {
                "verdict": "constrain",
                "age_result": {"profile_id": "post-batch"},
            }
        )

        host_attempts = []

        async def run_host_activity(input):
            context = build_core_activity_context(mock_activity_info, list(input.args))

            def intercept():
                adapter.handle_constrain_sync(started_result, context)
                adapter.raise_hook_blocked(started_result)
                host_attempts.append("host action ran")

            await asyncio.to_thread(intercept)
            return "host_result"

        interceptor.next.execute_activity.side_effect = run_host_activity
        mock_input = make_input(["arg1"])
        ctx, mock_activity = patched_activity(mock_activity_info)
        try:
            mock_activity.wait_for_cancelled.return_value = asyncio.Future()
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        sandbox.dispatcher.dispatch_with_decision.assert_awaited_once()
        assert host_attempts == []
        assert result["activity_result"] is None
        assert result["sandbox_execution"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_completed_behavioral_constrain_dispatches_profile_once(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        client.evaluate_event = AsyncMock(
            side_effect=[
                GovernanceVerdictResponse(verdict=Verdict.ALLOW),
                make_constrain_response(profile_id="post-batch"),
            ]
        )
        interceptor = make_interceptor(
            state,
            GovernanceConfig(),
            next_result={"activity": "host_result"},
            client=client,
        )
        interceptor._sandbox = sandbox
        mock_input = make_input(["arg1"])

        ctx, mock_activity = patched_activity(mock_activity_info)
        try:
            mock_activity.wait_for_cancelled.return_value = asyncio.Future()
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        interceptor.next.execute_activity.assert_awaited_once_with(mock_input)
        sandbox.dispatcher.dispatch_with_decision.assert_awaited_once()
        command, decision = (
            sandbox.dispatcher.dispatch_with_decision.await_args.args
        )
        assert command.kwargs["profile_id"] == "post-batch"
        assert command.kwargs["argv"] == ("/usr/bin/reconcile", "b-1")
        assert command.kwargs["arguments"] == {}
        assert decision["verdict"] == "constrain"
        assert decision["policy_id"] == "policy-1"
        assert decision["reason"] == "run in sandbox"
        assert decision["constraints"] == ["run_in_sandbox"]
        request = sandbox.profiles.derive.call_args.args[0]
        assert request.profile_id == "post-batch"
        assert request.arguments == ()
        assert result["activity"] == "host_result"
        assert result["sandbox_execution"] == {
            "status": "completed",
            "profile_id": "post-batch",
            "disposition": "executed_in_sandbox",
            "exit_code": 0,
            "timeout_status": "completed_within_timeout",
            "cleanup_status": "deleted",
            "stdout_bytes": len(b'{"rows": 7}'),
            "stderr_bytes": 0,
            "typed_result": None,
        }
        # The behavioral dispatch does not post another completion evaluation,
        # preventing a completion verdict from recursively dispatching again.
        assert client.evaluate_event.await_count == 2

    @pytest.mark.asyncio
    async def test_completed_behavioral_constrain_attaches_dispatch_failure(
        self, state, mock_activity_info
    ):
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        client.evaluate_event = AsyncMock(
            side_effect=[
                GovernanceVerdictResponse(verdict=Verdict.ALLOW),
                make_constrain_response(profile_id="post-batch"),
            ]
        )
        interceptor = make_interceptor(
            state, GovernanceConfig(), next_result="host_result", client=client
        )

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(make_input(["arg1"]))
        finally:
            ctx.stop()

        assert result["activity_result"] == "host_result"
        assert result["sandbox_execution"]["status"] == "failed"
        assert result["sandbox_execution"]["profile_id"] == "post-batch"
        assert result["sandbox_execution"]["error"] == {
            "type": "ApplicationError",
            "message": (
                "GovernedCommandConfigurationRequired: "
                "Governed command support is not configured"
            ),
        }
        assert client.evaluate_event.await_count == 2

    @pytest.mark.asyncio
    async def test_completed_constrain_without_sandbox_is_noop(
        self, state, mock_activity_info
    ):
        """A completion-stage CONSTRAIN cannot reroute an activity that already
        ran and must not fail solely because no sandbox is configured."""
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        client.evaluate_event = AsyncMock(
            side_effect=[
                GovernanceVerdictResponse(verdict=Verdict.ALLOW),
                make_constrain_response(),
            ]
        )
        interceptor = make_interceptor(
            state, GovernanceConfig(), next_result="host_result", client=client
        )
        mock_input = make_input(["arg1"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "host_result"
        interceptor.next.execute_activity.assert_awaited_once_with(mock_input)
        assert client.evaluate_event.await_count == 2

    @pytest.mark.asyncio
    async def test_completed_constrain_with_sandbox_is_noop(
        self, state, mock_activity_info, fake_governed_dispatcher
    ):
        """A CONSTRAIN verdict at ActivityCompleted (the activity already
        executed) is accepted as a no-op when a sandbox is configured."""
        sandbox = make_mock_sandbox(make_sandbox_dispatch_result())
        client = make_verdict_client(
            GovernanceVerdictResponse(verdict=Verdict.ALLOW)
        )
        client.evaluate_event = AsyncMock(
            side_effect=[
                GovernanceVerdictResponse(verdict=Verdict.ALLOW),
                make_constrain_response(),
            ]
        )
        interceptor = make_interceptor(
            state, GovernanceConfig(), next_result="host_result", client=client
        )
        interceptor._sandbox = sandbox
        mock_input = make_input(["arg1"])

        ctx, _ = patched_activity(mock_activity_info)
        try:
            result = await interceptor.execute_activity(mock_input)
        finally:
            ctx.stop()

        assert result == "host_result"
        interceptor.next.execute_activity.assert_awaited_once_with(mock_input)
        sandbox.dispatcher.dispatch_with_decision.assert_not_awaited()
        assert client.evaluate_event.await_count == 2
