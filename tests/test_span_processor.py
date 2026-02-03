# tests/test_span_processor.py
"""
Comprehensive tests for the WorkflowSpanProcessor class.

Tests cover:
1. Initialization with/without fallback_processor and ignored_url_prefixes
2. Workflow buffer management (register, get, remove, unregister)
3. Trace mapping (trace_id to workflow_id, with activity_id)
4. Verdict storage (set, get, clear, buffer updates)
5. Body storage (store_body, get_pending_body)
6. URL filtering (_should_ignore_span)
7. SpanProcessor interface (on_start, on_end, shutdown, force_flush)
8. Span data extraction (_extract_span_data)
"""

import pytest
from unittest.mock import MagicMock, Mock
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

from openbox.span_processor import WorkflowSpanProcessor
from openbox.types import WorkflowSpanBuffer, Verdict


# =============================================================================
# Mock OTel Types for Testing
# =============================================================================


class MockStatusCode(Enum):
    """Mock OTel StatusCode enum."""
    UNSET = 0
    OK = 1
    ERROR = 2


@dataclass
class MockStatus:
    """Mock OTel Status."""
    status_code: MockStatusCode = MockStatusCode.UNSET
    description: Optional[str] = None


@dataclass
class MockSpanContext:
    """Mock OTel SpanContext."""
    trace_id: int = 0x123456789ABCDEF0123456789ABCDEF0
    span_id: int = 0xABCDEF0123456789
    is_valid: bool = True


@dataclass
class MockSpanKind(Enum):
    """Mock OTel SpanKind."""
    INTERNAL = 0
    SERVER = 1
    CLIENT = 2
    PRODUCER = 3
    CONSUMER = 4


@dataclass
class MockEvent:
    """Mock OTel Event."""
    name: str
    timestamp: int
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class MockParentSpan:
    """Mock parent span for testing."""
    span_id: int = 0x1111111111111111


class MockReadableSpan:
    """Mock OTel ReadableSpan for testing."""

    def __init__(
        self,
        name: str = "test_span",
        trace_id: int = 0x123456789ABCDEF0123456789ABCDEF0,
        span_id: int = 0xABCDEF0123456789,
        parent_span_id: Optional[int] = None,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: int = 1000000000,
        end_time: int = 2000000000,
        status: Optional[MockStatus] = None,
        events: Optional[List[MockEvent]] = None,
        kind: MockSpanKind = MockSpanKind.INTERNAL,
    ):
        self.name = name
        self.context = MockSpanContext(trace_id=trace_id, span_id=span_id)
        self.attributes = attributes
        self.start_time = start_time
        self.end_time = end_time
        self.status = status or MockStatus()
        self.events = events or []
        self.kind = kind

        # Set parent
        if parent_span_id is not None:
            self.parent = MockParentSpan(span_id=parent_span_id)
        else:
            self.parent = None


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def processor():
    """Create a basic WorkflowSpanProcessor."""
    return WorkflowSpanProcessor()


@pytest.fixture
def mock_fallback():
    """Create a mock fallback processor."""
    fallback = MagicMock()
    fallback.on_end = MagicMock()
    fallback.shutdown = MagicMock()
    fallback.force_flush = MagicMock(return_value=True)
    return fallback


@pytest.fixture
def processor_with_fallback(mock_fallback):
    """Create a WorkflowSpanProcessor with a fallback processor."""
    return WorkflowSpanProcessor(fallback_processor=mock_fallback)


@pytest.fixture
def processor_with_ignored_urls():
    """Create a WorkflowSpanProcessor with ignored URL prefixes."""
    return WorkflowSpanProcessor(
        ignored_url_prefixes=["https://openbox.internal/", "http://localhost:9090/"]
    )


@pytest.fixture
def sample_buffer():
    """Create a sample WorkflowSpanBuffer."""
    return WorkflowSpanBuffer(
        workflow_id="wf-123",
        run_id="run-456",
        workflow_type="TestWorkflow",
        task_queue="test-queue",
    )


@pytest.fixture
def sample_span():
    """Create a sample mock span."""
    return MockReadableSpan(
        name="test_activity",
        trace_id=0x123456789ABCDEF0123456789ABCDEF0,
        span_id=0xABCDEF0123456789,
        attributes={
            "temporal.workflow_id": "wf-123",
            "temporal.activity_id": "act-789",
        },
    )


# =============================================================================
# 1. Initialization Tests
# =============================================================================


class TestInitialization:
    """Tests for WorkflowSpanProcessor initialization."""

    def test_init_defaults(self, processor):
        """Test initialization with default values."""
        assert processor.fallback is None
        assert processor._ignored_url_prefixes == set()
        assert processor._buffers == {}
        assert processor._trace_to_workflow == {}
        assert processor._trace_to_activity == {}
        assert processor._body_data == {}
        assert processor._verdicts == {}
        assert processor._lock is not None

    def test_init_with_fallback_processor(self, mock_fallback):
        """Test initialization with a fallback processor."""
        processor = WorkflowSpanProcessor(fallback_processor=mock_fallback)
        assert processor.fallback is mock_fallback

    def test_init_without_fallback_processor(self):
        """Test initialization without a fallback processor."""
        processor = WorkflowSpanProcessor(fallback_processor=None)
        assert processor.fallback is None

    def test_init_with_ignored_url_prefixes(self):
        """Test initialization with ignored URL prefixes."""
        prefixes = ["https://api.openbox.com/", "http://internal/"]
        processor = WorkflowSpanProcessor(ignored_url_prefixes=prefixes)
        assert processor._ignored_url_prefixes == set(prefixes)

    def test_init_without_ignored_url_prefixes(self):
        """Test initialization without ignored URL prefixes."""
        processor = WorkflowSpanProcessor(ignored_url_prefixes=None)
        assert processor._ignored_url_prefixes == set()

    def test_init_with_empty_ignored_url_prefixes(self):
        """Test initialization with empty ignored URL prefixes list."""
        processor = WorkflowSpanProcessor(ignored_url_prefixes=[])
        assert processor._ignored_url_prefixes == set()

    def test_init_with_both_options(self, mock_fallback):
        """Test initialization with both fallback and ignored prefixes."""
        prefixes = ["https://openbox.internal/"]
        processor = WorkflowSpanProcessor(
            fallback_processor=mock_fallback,
            ignored_url_prefixes=prefixes,
        )
        assert processor.fallback is mock_fallback
        assert processor._ignored_url_prefixes == set(prefixes)


# =============================================================================
# 2. Workflow Buffer Management Tests
# =============================================================================


class TestWorkflowBufferManagement:
    """Tests for workflow buffer management methods."""

    def test_register_workflow(self, processor, sample_buffer):
        """Test registering a workflow buffer."""
        processor.register_workflow("wf-123", sample_buffer)
        assert "wf-123" in processor._buffers
        assert processor._buffers["wf-123"] is sample_buffer

    def test_register_workflow_overwrites_existing(self, processor, sample_buffer):
        """Test that registering a workflow overwrites existing buffer."""
        old_buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="old-run",
            workflow_type="OldWorkflow",
            task_queue="old-queue",
        )
        processor.register_workflow("wf-123", old_buffer)
        processor.register_workflow("wf-123", sample_buffer)
        assert processor._buffers["wf-123"] is sample_buffer

    def test_get_buffer_existing(self, processor, sample_buffer):
        """Test getting an existing buffer."""
        processor.register_workflow("wf-123", sample_buffer)
        result = processor.get_buffer("wf-123")
        assert result is sample_buffer

    def test_get_buffer_not_found(self, processor):
        """Test getting a non-existent buffer returns None."""
        result = processor.get_buffer("nonexistent")
        assert result is None

    def test_get_buffer_does_not_remove(self, processor, sample_buffer):
        """Test that get_buffer does not remove the buffer."""
        processor.register_workflow("wf-123", sample_buffer)
        processor.get_buffer("wf-123")
        assert "wf-123" in processor._buffers

    def test_remove_buffer_existing(self, processor, sample_buffer):
        """Test removing an existing buffer."""
        processor.register_workflow("wf-123", sample_buffer)
        result = processor.remove_buffer("wf-123")
        assert result is sample_buffer
        assert "wf-123" not in processor._buffers

    def test_remove_buffer_not_found(self, processor):
        """Test removing a non-existent buffer returns None."""
        result = processor.remove_buffer("nonexistent")
        assert result is None

    def test_unregister_workflow_removes_buffer(self, processor, sample_buffer):
        """Test that unregister_workflow removes the buffer."""
        processor.register_workflow("wf-123", sample_buffer)
        processor.unregister_workflow("wf-123")
        assert "wf-123" not in processor._buffers

    def test_unregister_workflow_removes_verdict(self, processor, sample_buffer):
        """Test that unregister_workflow removes the verdict."""
        processor.register_workflow("wf-123", sample_buffer)
        processor.set_verdict("wf-123", Verdict.BLOCK, "test reason")
        processor.unregister_workflow("wf-123")
        assert "wf-123" not in processor._buffers
        assert "wf-123" not in processor._verdicts

    def test_unregister_workflow_nonexistent(self, processor):
        """Test unregistering a non-existent workflow doesn't raise."""
        processor.unregister_workflow("nonexistent")  # Should not raise


# =============================================================================
# 3. Trace Mapping Tests
# =============================================================================


class TestTraceMapping:
    """Tests for trace_id to workflow_id mapping."""

    def test_register_trace_basic(self, processor):
        """Test registering a trace_id to workflow_id mapping."""
        trace_id = 0x123456789ABCDEF
        processor.register_trace(trace_id, "wf-123")
        assert processor._trace_to_workflow[trace_id] == "wf-123"

    def test_register_trace_with_activity_id(self, processor):
        """Test registering a trace with activity_id."""
        trace_id = 0x123456789ABCDEF
        processor.register_trace(trace_id, "wf-123", activity_id="act-456")
        assert processor._trace_to_workflow[trace_id] == "wf-123"
        assert processor._trace_to_activity[trace_id] == "act-456"

    def test_register_trace_without_activity_id(self, processor):
        """Test registering a trace without activity_id."""
        trace_id = 0x123456789ABCDEF
        processor.register_trace(trace_id, "wf-123", activity_id=None)
        assert processor._trace_to_workflow[trace_id] == "wf-123"
        assert trace_id not in processor._trace_to_activity

    def test_register_trace_overwrites_existing(self, processor):
        """Test that registering a trace overwrites existing mapping."""
        trace_id = 0x123456789ABCDEF
        processor.register_trace(trace_id, "wf-old")
        processor.register_trace(trace_id, "wf-new")
        assert processor._trace_to_workflow[trace_id] == "wf-new"

    def test_register_trace_multiple(self, processor):
        """Test registering multiple trace mappings."""
        processor.register_trace(0x111, "wf-1", "act-1")
        processor.register_trace(0x222, "wf-2", "act-2")
        processor.register_trace(0x333, "wf-3")

        assert processor._trace_to_workflow[0x111] == "wf-1"
        assert processor._trace_to_workflow[0x222] == "wf-2"
        assert processor._trace_to_workflow[0x333] == "wf-3"
        assert processor._trace_to_activity[0x111] == "act-1"
        assert processor._trace_to_activity[0x222] == "act-2"
        assert 0x333 not in processor._trace_to_activity


# =============================================================================
# 4. Verdict Storage Tests
# =============================================================================


class TestVerdictStorage:
    """Tests for verdict storage methods."""

    def test_set_verdict(self, processor):
        """Test storing a verdict."""
        processor.set_verdict("wf-123", Verdict.BLOCK, "policy violation")
        assert "wf-123" in processor._verdicts
        assert processor._verdicts["wf-123"]["verdict"] == Verdict.BLOCK
        assert processor._verdicts["wf-123"]["reason"] == "policy violation"

    def test_set_verdict_with_run_id(self, processor):
        """Test storing a verdict with run_id."""
        processor.set_verdict("wf-123", Verdict.HALT, "critical error", run_id="run-456")
        assert processor._verdicts["wf-123"]["run_id"] == "run-456"

    def test_set_verdict_without_reason(self, processor):
        """Test storing a verdict without reason."""
        processor.set_verdict("wf-123", Verdict.ALLOW)
        assert processor._verdicts["wf-123"]["verdict"] == Verdict.ALLOW
        assert processor._verdicts["wf-123"]["reason"] is None

    def test_set_verdict_updates_buffer(self, processor, sample_buffer):
        """Test that set_verdict updates the buffer if it exists."""
        processor.register_workflow("wf-123", sample_buffer)
        processor.set_verdict("wf-123", Verdict.REQUIRE_APPROVAL, "needs review")
        assert sample_buffer.verdict == Verdict.REQUIRE_APPROVAL
        assert sample_buffer.verdict_reason == "needs review"

    def test_set_verdict_no_buffer(self, processor):
        """Test set_verdict when buffer doesn't exist."""
        processor.set_verdict("wf-123", Verdict.BLOCK, "no buffer")
        # Should not raise, verdict still stored
        assert processor._verdicts["wf-123"]["verdict"] == Verdict.BLOCK

    def test_get_verdict_existing(self, processor):
        """Test getting an existing verdict."""
        processor.set_verdict("wf-123", Verdict.HALT, "critical")
        result = processor.get_verdict("wf-123")
        assert result is not None
        assert result["verdict"] == Verdict.HALT
        assert result["reason"] == "critical"

    def test_get_verdict_not_found(self, processor):
        """Test getting a non-existent verdict returns None."""
        result = processor.get_verdict("nonexistent")
        assert result is None

    def test_clear_verdict(self, processor):
        """Test clearing a verdict."""
        processor.set_verdict("wf-123", Verdict.BLOCK, "test")
        processor.clear_verdict("wf-123")
        assert "wf-123" not in processor._verdicts

    def test_clear_verdict_nonexistent(self, processor):
        """Test clearing a non-existent verdict doesn't raise."""
        processor.clear_verdict("nonexistent")  # Should not raise

    def test_verdict_all_types(self, processor):
        """Test all verdict types can be stored."""
        for verdict in Verdict:
            processor.set_verdict(f"wf-{verdict.value}", verdict, f"reason-{verdict.value}")
            assert processor._verdicts[f"wf-{verdict.value}"]["verdict"] == verdict


# =============================================================================
# 5. Body Storage Tests
# =============================================================================


class TestBodyStorage:
    """Tests for body storage methods."""

    def test_store_body_request_only(self, processor):
        """Test storing only request body."""
        processor.store_body(0x123, request_body='{"key": "value"}')
        assert processor._body_data[0x123]["request_body"] == '{"key": "value"}'
        assert "response_body" not in processor._body_data[0x123]

    def test_store_body_response_only(self, processor):
        """Test storing only response body."""
        processor.store_body(0x123, response_body='{"result": "ok"}')
        assert processor._body_data[0x123]["response_body"] == '{"result": "ok"}'
        assert "request_body" not in processor._body_data[0x123]

    def test_store_body_both(self, processor):
        """Test storing both request and response bodies."""
        processor.store_body(
            0x123,
            request_body='{"input": "data"}',
            response_body='{"output": "result"}',
        )
        assert processor._body_data[0x123]["request_body"] == '{"input": "data"}'
        assert processor._body_data[0x123]["response_body"] == '{"output": "result"}'

    def test_store_body_with_headers(self, processor):
        """Test storing bodies with headers."""
        request_headers = {"Content-Type": "application/json"}
        response_headers = {"X-Custom": "header"}
        processor.store_body(
            0x123,
            request_body="request",
            response_body="response",
            request_headers=request_headers,
            response_headers=response_headers,
        )
        assert processor._body_data[0x123]["request_headers"] == request_headers
        assert processor._body_data[0x123]["response_headers"] == response_headers

    def test_store_body_incremental(self, processor):
        """Test storing body data incrementally."""
        processor.store_body(0x123, request_body="request")
        processor.store_body(0x123, response_body="response")
        assert processor._body_data[0x123]["request_body"] == "request"
        assert processor._body_data[0x123]["response_body"] == "response"

    def test_store_body_overwrites(self, processor):
        """Test that subsequent calls overwrite existing data."""
        processor.store_body(0x123, request_body="old")
        processor.store_body(0x123, request_body="new")
        assert processor._body_data[0x123]["request_body"] == "new"

    def test_get_pending_body_existing(self, processor):
        """Test getting pending body data."""
        processor.store_body(0x123, request_body="test", response_body="result")
        result = processor.get_pending_body(0x123)
        assert result is not None
        assert result["request_body"] == "test"
        assert result["response_body"] == "result"

    def test_get_pending_body_not_found(self, processor):
        """Test getting non-existent pending body returns None."""
        result = processor.get_pending_body(0x999)
        assert result is None

    def test_get_pending_body_does_not_remove(self, processor):
        """Test that get_pending_body does not remove the data."""
        processor.store_body(0x123, request_body="test")
        processor.get_pending_body(0x123)
        assert 0x123 in processor._body_data


# =============================================================================
# 6. URL Filtering Tests (_should_ignore_span)
# =============================================================================


class TestUrlFiltering:
    """Tests for _should_ignore_span method."""

    def test_should_ignore_no_prefixes(self, processor):
        """Test span not ignored when no prefixes configured."""
        span = MockReadableSpan(
            attributes={"http.url": "https://api.example.com/endpoint"}
        )
        assert processor._should_ignore_span(span) is False

    def test_should_ignore_matching_prefix(self, processor_with_ignored_urls):
        """Test span ignored when URL matches prefix."""
        span = MockReadableSpan(
            attributes={"http.url": "https://openbox.internal/api/v1/evaluate"}
        )
        assert processor_with_ignored_urls._should_ignore_span(span) is True

    def test_should_not_ignore_non_matching_prefix(self, processor_with_ignored_urls):
        """Test span not ignored when URL doesn't match prefix."""
        span = MockReadableSpan(
            attributes={"http.url": "https://api.example.com/endpoint"}
        )
        assert processor_with_ignored_urls._should_ignore_span(span) is False

    def test_should_ignore_multiple_prefixes(self, processor_with_ignored_urls):
        """Test multiple ignored prefixes work correctly."""
        span1 = MockReadableSpan(
            attributes={"http.url": "https://openbox.internal/api/v1"}
        )
        span2 = MockReadableSpan(
            attributes={"http.url": "http://localhost:9090/metrics"}
        )
        assert processor_with_ignored_urls._should_ignore_span(span1) is True
        assert processor_with_ignored_urls._should_ignore_span(span2) is True

    def test_should_ignore_no_url_attribute(self, processor_with_ignored_urls):
        """Test span not ignored when no http.url attribute."""
        span = MockReadableSpan(attributes={"other.attr": "value"})
        assert processor_with_ignored_urls._should_ignore_span(span) is False

    def test_should_ignore_no_attributes(self, processor_with_ignored_urls):
        """Test span not ignored when no attributes."""
        span = MockReadableSpan(attributes=None)
        assert processor_with_ignored_urls._should_ignore_span(span) is False

    def test_should_ignore_partial_prefix_match(self, processor_with_ignored_urls):
        """Test that only prefix matches work."""
        span = MockReadableSpan(
            attributes={"http.url": "https://not-openbox.internal/api"}
        )
        assert processor_with_ignored_urls._should_ignore_span(span) is False


# =============================================================================
# 7. on_start Tests
# =============================================================================


class TestOnStart:
    """Tests for on_start method."""

    def test_on_start_is_noop(self, processor):
        """Test that on_start is a no-op."""
        span = MockReadableSpan()
        # Should not raise and should not modify anything
        processor.on_start(span, parent_context=None)
        assert processor._buffers == {}
        assert processor._body_data == {}

    def test_on_start_with_fallback(self, processor_with_fallback, mock_fallback):
        """Test that on_start doesn't call fallback."""
        span = MockReadableSpan()
        processor_with_fallback.on_start(span, parent_context=None)
        mock_fallback.on_start.assert_not_called()


# =============================================================================
# 8. on_end Tests
# =============================================================================


class TestOnEnd:
    """Tests for on_end method."""

    def test_on_end_buffers_span_with_workflow_id_attribute(self, processor, sample_buffer):
        """Test buffering spans that have workflow_id attribute."""
        processor.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="test_activity",
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)
        assert len(sample_buffer.spans) == 1
        assert sample_buffer.spans[0]["name"] == "test_activity"

    def test_on_end_buffers_span_via_trace_mapping(self, processor, sample_buffer):
        """Test buffering spans via trace_id mapping."""
        processor.register_workflow("wf-123", sample_buffer)
        trace_id = 0xABCDEF123456
        processor.register_trace(trace_id, "wf-123")

        span = MockReadableSpan(
            name="http_call",
            trace_id=trace_id,
            attributes={},  # No workflow_id attribute
        )
        processor.on_end(span)
        assert len(sample_buffer.spans) == 1
        assert sample_buffer.spans[0]["name"] == "http_call"

    def test_on_end_sets_activity_id_from_attribute(self, processor, sample_buffer):
        """Test that activity_id from span attribute is set."""
        processor.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="activity",
            attributes={
                "temporal.workflow_id": "wf-123",
                "temporal.activity_id": "act-456",
            },
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["activity_id"] == "act-456"

    def test_on_end_sets_activity_id_from_trace_mapping(self, processor, sample_buffer):
        """Test that activity_id from trace mapping is set for child spans."""
        processor.register_workflow("wf-123", sample_buffer)
        trace_id = 0xABCDEF123456
        processor.register_trace(trace_id, "wf-123", activity_id="act-789")

        span = MockReadableSpan(
            name="child_span",
            trace_id=trace_id,
            attributes={},  # No workflow_id or activity_id
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["activity_id"] == "act-789"

    def test_on_end_merges_body_data(self, processor, sample_buffer):
        """Test that body data is merged on span end."""
        processor.register_workflow("wf-123", sample_buffer)
        span_id = 0xABCDEF0123456789
        processor.store_body(
            span_id,
            request_body='{"input": "test"}',
            response_body='{"output": "result"}',
        )

        span = MockReadableSpan(
            name="http_span",
            span_id=span_id,
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)

        assert sample_buffer.spans[0]["request_body"] == '{"input": "test"}'
        assert sample_buffer.spans[0]["response_body"] == '{"output": "result"}'
        # Body data should be removed after merge
        assert span_id not in processor._body_data

    def test_on_end_merges_body_headers(self, processor, sample_buffer):
        """Test that header data is merged on span end."""
        processor.register_workflow("wf-123", sample_buffer)
        span_id = 0xABCDEF0123456789
        processor.store_body(
            span_id,
            request_headers={"Content-Type": "application/json"},
            response_headers={"X-Request-Id": "abc123"},
        )

        span = MockReadableSpan(
            name="http_span",
            span_id=span_id,
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)

        assert sample_buffer.spans[0]["request_headers"] == {"Content-Type": "application/json"}
        assert sample_buffer.spans[0]["response_headers"] == {"X-Request-Id": "abc123"}

    def test_on_end_ignores_span_to_ignored_url(self, processor_with_ignored_urls):
        """Test that spans to ignored URLs are not buffered."""
        buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="run-456",
            workflow_type="Test",
            task_queue="test",
        )
        processor_with_ignored_urls.register_workflow("wf-123", buffer)

        span = MockReadableSpan(
            name="openbox_call",
            attributes={
                "temporal.workflow_id": "wf-123",
                "http.url": "https://openbox.internal/api/v1/evaluate",
            },
        )
        processor_with_ignored_urls.on_end(span)
        assert len(buffer.spans) == 0

    def test_on_end_forwards_to_fallback(self, processor_with_fallback, mock_fallback, sample_buffer):
        """Test that spans are forwarded to fallback processor."""
        processor_with_fallback.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="test_span",
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor_with_fallback.on_end(span)
        mock_fallback.on_end.assert_called_once_with(span)

    def test_on_end_forwards_ignored_span_to_fallback(
        self, mock_fallback
    ):
        """Test that ignored spans are still forwarded to fallback."""
        processor = WorkflowSpanProcessor(
            fallback_processor=mock_fallback,
            ignored_url_prefixes=["https://openbox.internal/"],
        )
        buffer = WorkflowSpanBuffer(
            workflow_id="wf-123",
            run_id="run-456",
            workflow_type="Test",
            task_queue="test",
        )
        processor.register_workflow("wf-123", buffer)

        span = MockReadableSpan(
            name="openbox_call",
            attributes={
                "temporal.workflow_id": "wf-123",
                "http.url": "https://openbox.internal/api/v1",
            },
        )
        processor.on_end(span)
        # Should not be buffered
        assert len(buffer.spans) == 0
        # Should still be forwarded to fallback
        mock_fallback.on_end.assert_called_once_with(span)

    def test_on_end_no_workflow_id_no_buffer(self, processor):
        """Test span without workflow_id and no trace mapping is not buffered."""
        span = MockReadableSpan(
            name="orphan_span",
            attributes={},
        )
        processor.on_end(span)  # Should not raise

    def test_on_end_no_buffer_for_workflow(self, processor):
        """Test span with workflow_id but no registered buffer."""
        span = MockReadableSpan(
            name="unregistered_span",
            attributes={"temporal.workflow_id": "wf-unknown"},
        )
        processor.on_end(span)  # Should not raise

    def test_on_end_without_fallback(self, processor, sample_buffer):
        """Test on_end works without a fallback processor."""
        processor.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="test_span",
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)  # Should not raise
        assert len(sample_buffer.spans) == 1


# =============================================================================
# 9. _extract_span_data Tests
# =============================================================================


class TestExtractSpanData:
    """Tests for _extract_span_data method."""

    def test_extract_basic_span_data(self, processor):
        """Test extracting basic span data."""
        span = MockReadableSpan(
            name="test_span",
            trace_id=0x123456789ABCDEF0123456789ABCDEF0,
            span_id=0xABCDEF0123456789,
            start_time=1000000000,
            end_time=2000000000,
        )
        data = processor._extract_span_data(span)

        assert data["name"] == "test_span"
        assert data["span_id"] == "abcdef0123456789"  # 16 hex chars
        # Note: trace_id format depends on the actual integer value
        assert data["trace_id"] == "123456789abcdef0123456789abcdef0"  # 32 hex chars (leading zero stripped if not needed)
        assert data["start_time"] == 1000000000
        assert data["end_time"] == 2000000000
        assert data["duration_ns"] == 1000000000  # end - start

    def test_extract_span_data_with_parent(self, processor):
        """Test extracting span data with parent span."""
        span = MockReadableSpan(
            name="child_span",
            parent_span_id=0x1111111111111111,
        )
        data = processor._extract_span_data(span)
        assert data["parent_span_id"] == "1111111111111111"

    def test_extract_span_data_without_parent(self, processor):
        """Test extracting span data without parent span."""
        span = MockReadableSpan(
            name="root_span",
            parent_span_id=None,
        )
        data = processor._extract_span_data(span)
        assert data["parent_span_id"] is None

    def test_extract_span_data_with_attributes(self, processor):
        """Test extracting span data with attributes."""
        span = MockReadableSpan(
            name="span_with_attrs",
            attributes={
                "http.method": "POST",
                "http.url": "https://api.example.com",
                "custom.attr": 123,
            },
        )
        data = processor._extract_span_data(span)
        assert data["attributes"]["http.method"] == "POST"
        assert data["attributes"]["http.url"] == "https://api.example.com"
        assert data["attributes"]["custom.attr"] == 123

    def test_extract_span_data_without_attributes(self, processor):
        """Test extracting span data without attributes."""
        span = MockReadableSpan(name="no_attrs", attributes=None)
        data = processor._extract_span_data(span)
        assert data["attributes"] == {}

    def test_extract_span_data_with_status(self, processor):
        """Test extracting span data with status."""
        span = MockReadableSpan(
            name="span_with_status",
            status=MockStatus(status_code=MockStatusCode.ERROR, description="Something went wrong"),
        )
        data = processor._extract_span_data(span)
        assert data["status"]["code"] == "ERROR"
        assert data["status"]["description"] == "Something went wrong"

    def test_extract_span_data_with_ok_status(self, processor):
        """Test extracting span data with OK status."""
        span = MockReadableSpan(
            name="span_ok",
            status=MockStatus(status_code=MockStatusCode.OK),
        )
        data = processor._extract_span_data(span)
        assert data["status"]["code"] == "OK"

    def test_extract_span_data_with_events(self, processor):
        """Test extracting span data with events."""
        events = [
            MockEvent(name="event1", timestamp=1500000000, attributes={"key": "value"}),
            MockEvent(name="event2", timestamp=1600000000, attributes=None),
        ]
        span = MockReadableSpan(name="span_with_events", events=events)
        data = processor._extract_span_data(span)

        assert len(data["events"]) == 2
        assert data["events"][0]["name"] == "event1"
        assert data["events"][0]["timestamp"] == 1500000000
        assert data["events"][0]["attributes"] == {"key": "value"}
        assert data["events"][1]["name"] == "event2"
        assert data["events"][1]["attributes"] == {}

    def test_extract_span_data_with_kind(self, processor):
        """Test extracting span data with span kind."""
        span = MockReadableSpan(name="client_span", kind=MockSpanKind.CLIENT)
        data = processor._extract_span_data(span)
        assert data["kind"] == "CLIENT"

    def test_extract_span_data_duration_calculation(self, processor):
        """Test duration calculation in span data."""
        span = MockReadableSpan(
            name="timed_span",
            start_time=1000000000,
            end_time=1500000000,
        )
        data = processor._extract_span_data(span)
        assert data["duration_ns"] == 500000000

    def test_extract_span_data_missing_times(self, processor):
        """Test span data when times are missing."""
        span = MockReadableSpan(
            name="timeless_span",
            start_time=None,
            end_time=None,
        )
        data = processor._extract_span_data(span)
        assert data["duration_ns"] is None


# =============================================================================
# 10. shutdown and force_flush Tests
# =============================================================================


class TestShutdownAndForceFlush:
    """Tests for shutdown and force_flush methods."""

    def test_shutdown_delegates_to_fallback(self, processor_with_fallback, mock_fallback):
        """Test that shutdown delegates to fallback processor."""
        processor_with_fallback.shutdown()
        mock_fallback.shutdown.assert_called_once()

    def test_shutdown_without_fallback(self, processor):
        """Test shutdown without fallback processor doesn't raise."""
        processor.shutdown()  # Should not raise

    def test_force_flush_delegates_to_fallback(self, processor_with_fallback, mock_fallback):
        """Test that force_flush delegates to fallback processor."""
        result = processor_with_fallback.force_flush(timeout_millis=5000)
        mock_fallback.force_flush.assert_called_once_with(5000)
        assert result is True

    def test_force_flush_without_fallback(self, processor):
        """Test force_flush without fallback returns True."""
        result = processor.force_flush(timeout_millis=5000)
        assert result is True

    def test_force_flush_default_timeout(self, processor_with_fallback, mock_fallback):
        """Test force_flush with default timeout."""
        processor_with_fallback.force_flush()
        mock_fallback.force_flush.assert_called_once_with(30000)

    def test_force_flush_returns_fallback_result(self, mock_fallback):
        """Test force_flush returns result from fallback."""
        mock_fallback.force_flush.return_value = False
        processor = WorkflowSpanProcessor(fallback_processor=mock_fallback)
        result = processor.force_flush()
        assert result is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestThreadSafety:
    """Tests for thread safety of the processor."""

    def test_concurrent_buffer_registration(self, processor):
        """Test concurrent buffer registration."""
        import threading

        buffers = []
        for i in range(10):
            buffer = WorkflowSpanBuffer(
                workflow_id=f"wf-{i}",
                run_id=f"run-{i}",
                workflow_type="ConcurrentWorkflow",
                task_queue="test",
            )
            buffers.append((f"wf-{i}", buffer))

        threads = []
        for wf_id, buffer in buffers:
            t = threading.Thread(
                target=processor.register_workflow,
                args=(wf_id, buffer),
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(processor._buffers) == 10

    def test_concurrent_store_body(self, processor):
        """Test concurrent body storage."""
        import threading

        def store_body(span_id):
            processor.store_body(span_id, request_body=f"body-{span_id}")

        threads = []
        for i in range(10):
            t = threading.Thread(target=store_body, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(processor._body_data) == 10

    def test_concurrent_on_end(self, processor, sample_buffer):
        """Test concurrent span processing."""
        import threading

        processor.register_workflow("wf-123", sample_buffer)

        def process_span(i):
            span = MockReadableSpan(
                name=f"span_{i}",
                span_id=i,
                attributes={"temporal.workflow_id": "wf-123"},
            )
            processor.on_end(span)

        threads = []
        for i in range(10):
            t = threading.Thread(target=process_span, args=(i,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(sample_buffer.spans) == 10


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_workflow_id_in_attribute_not_buffered(self, processor):
        """Test that empty workflow_id in span attribute is treated as falsy (not buffered).

        In Python, empty string is falsy, so spans with empty workflow_id attribute
        will not be buffered directly. They would need trace_id mapping.
        """
        buffer = WorkflowSpanBuffer(
            workflow_id="",
            run_id="run",
            workflow_type="Test",
            task_queue="test",
        )
        processor.register_workflow("", buffer)
        span = MockReadableSpan(
            name="empty_wf_span",
            attributes={"temporal.workflow_id": ""},
        )
        processor.on_end(span)
        # Empty string is falsy in Python, so span is not buffered via workflow_id attribute
        assert len(buffer.spans) == 0

    def test_empty_workflow_id_via_trace_mapping_not_buffered(self, processor):
        """Test that empty workflow_id from trace mapping is also not buffered.

        Empty string is falsy in Python, so even if trace mapping returns empty
        string workflow_id, the span will not be buffered. This is intentional -
        empty string is not a valid workflow ID.
        """
        buffer = WorkflowSpanBuffer(
            workflow_id="",
            run_id="run",
            workflow_type="Test",
            task_queue="test",
        )
        processor.register_workflow("", buffer)
        trace_id = 0xABCDEF123456
        processor.register_trace(trace_id, "")  # Map to empty workflow_id

        span = MockReadableSpan(
            name="empty_wf_span",
            trace_id=trace_id,
            attributes={},  # No workflow_id attribute
        )
        processor.on_end(span)
        # Empty string workflow_id is falsy, so span is not buffered
        assert len(buffer.spans) == 0

    def test_large_body_data(self, processor, sample_buffer):
        """Test handling large body data."""
        processor.register_workflow("wf-123", sample_buffer)
        large_body = "x" * 1000000  # 1MB body
        processor.store_body(0x123, request_body=large_body)

        span = MockReadableSpan(
            name="large_body_span",
            span_id=0x123,
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["request_body"] == large_body

    def test_special_characters_in_body(self, processor, sample_buffer):
        """Test handling special characters in body."""
        processor.register_workflow("wf-123", sample_buffer)
        special_body = '{"key": "value with \n newlines \t tabs and \"quotes\""}'
        processor.store_body(0x123, request_body=special_body)

        span = MockReadableSpan(
            name="special_span",
            span_id=0x123,
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["request_body"] == special_body

    def test_unicode_in_span_name(self, processor, sample_buffer):
        """Test handling unicode in span name."""
        processor.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="span_with_unicode_\u4e2d\u6587",
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["name"] == "span_with_unicode_\u4e2d\u6587"

    def test_zero_span_id(self, processor, sample_buffer):
        """Test handling zero span_id."""
        processor.register_workflow("wf-123", sample_buffer)
        processor.store_body(0, request_body="zero_id_body")

        span = MockReadableSpan(
            name="zero_span",
            span_id=0,
            attributes={"temporal.workflow_id": "wf-123"},
        )
        processor.on_end(span)
        assert sample_buffer.spans[0]["request_body"] == "zero_id_body"

    def test_very_long_workflow_id(self, processor):
        """Test handling very long workflow_id."""
        long_wf_id = "wf-" + "x" * 10000
        buffer = WorkflowSpanBuffer(
            workflow_id=long_wf_id,
            run_id="run",
            workflow_type="Test",
            task_queue="test",
        )
        processor.register_workflow(long_wf_id, buffer)
        assert processor.get_buffer(long_wf_id) is buffer

    def test_span_with_none_attributes_value(self, processor, sample_buffer):
        """Test span where attributes dict exists but values may be None."""
        processor.register_workflow("wf-123", sample_buffer)
        span = MockReadableSpan(
            name="null_attr_span",
            attributes={
                "temporal.workflow_id": "wf-123",
                "some.attr": None,
            },
        )
        processor.on_end(span)
        assert len(sample_buffer.spans) == 1
