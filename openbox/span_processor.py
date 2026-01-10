# openbox/span_processor.py
"""
OpenTelemetry SpanProcessor for workflow-boundary governance.

WorkflowSpanProcessor buffers spans per-workflow for batch submission
to OpenBox Core. Bodies are stored separately via store_body() and merged
on span end - this keeps bodies OUT of OTel spans but IN the OpenBox buffer.
"""

from typing import TYPE_CHECKING, Dict, Optional
import threading
import logging

# Logger for debugging HITL flow (outside workflow sandbox)
_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

from .types import WorkflowSpanBuffer, Verdict


class WorkflowSpanProcessor:
    """
    SpanProcessor that buffers spans per-workflow for batch submission.

    Bodies are stored separately via store_body() and merged on span end.
    This keeps bodies OUT of OTel spans but IN the OpenBox buffer.

    Thread-safe: Uses workflow_id from span attributes as key, with trace_id
    as fallback for child spans (like HTTP spans) that don't have workflow_id.

    Usage:
        processor = WorkflowSpanProcessor(fallback_processor=batch_processor)

        # Register buffer before workflow starts
        processor.register_workflow(workflow_id, buffer)

        # Spans with temporal.workflow_id attribute are buffered
        # Child spans (same trace_id) are also buffered via trace_id mapping
        # Bodies stored via store_body() are merged on span end

        # Get buffer after workflow completes, spans are in buffer.spans
        buffer = processor.get_buffer(workflow_id)
        spans = buffer.spans  # List of span dicts
    """

    def __init__(
        self,
        fallback_processor: Optional["SpanProcessor"] = None,
        ignored_url_prefixes: Optional[list] = None,
    ):
        """
        Initialize the span processor.

        Args:
            fallback_processor: Optional processor to forward spans to (e.g., Jaeger exporter).
                               Spans are forwarded WITHOUT body data for privacy.
            ignored_url_prefixes: List of URL prefixes to ignore (e.g., OpenBox Core API)
        """
        self.fallback = fallback_processor
        self._ignored_url_prefixes = set(ignored_url_prefixes or [])
        self._buffers: Dict[str, WorkflowSpanBuffer] = {}  # workflow_id -> buffer
        self._trace_to_workflow: Dict[int, str] = {}  # trace_id (int) -> workflow_id
        self._trace_to_activity: Dict[int, str] = {}  # trace_id (int) -> activity_id
        self._body_data: Dict[int, dict] = {}  # span_id (int) -> {request_body, response_body}
        self._verdicts: Dict[str, dict] = {}  # workflow_id -> {"verdict": Verdict, "reason": str}
        self._lock = threading.Lock()

    def _should_ignore_span(self, span: "ReadableSpan") -> bool:
        """Check if span should be ignored based on URL."""
        if not self._ignored_url_prefixes:
            return False

        # Check http.url attribute
        url = span.attributes.get("http.url") if span.attributes else None
        if url:
            for prefix in self._ignored_url_prefixes:
                if url.startswith(prefix):
                    return True
        return False

    # ═══════════════════════════════════════════════════════════════════════════
    # Workflow Buffer Management (called by GovernanceWorkflowInterceptor)
    # ═══════════════════════════════════════════════════════════════════════════

    def register_workflow(self, workflow_id: str, buffer: WorkflowSpanBuffer) -> None:
        """
        Register buffer for a workflow.

        Called by ActivityGovernanceInterceptor when first activity starts.

        Args:
            workflow_id: Temporal workflow ID
            buffer: Buffer to collect spans for this workflow
        """
        with self._lock:
            self._buffers[workflow_id] = buffer

    def register_trace(self, trace_id: int, workflow_id: str, activity_id: str = None) -> None:
        """
        Register trace_id to workflow_id (and optionally activity_id) mapping.

        Called when creating an activity span to enable child span buffering.
        Child spans (like HTTP calls) don't have temporal.workflow_id attribute,
        but share the same trace_id with the parent activity span.

        Args:
            trace_id: OTel trace ID (integer form)
            workflow_id: Temporal workflow ID
            activity_id: Temporal activity ID (optional, for filtering)
        """
        with self._lock:
            self._trace_to_workflow[trace_id] = workflow_id
            if activity_id:
                self._trace_to_activity[trace_id] = activity_id

    def get_buffer(self, workflow_id: str) -> Optional[WorkflowSpanBuffer]:
        """
        Retrieve buffer without removing it.

        Args:
            workflow_id: Temporal workflow ID

        Returns:
            Buffer if found, None otherwise
        """
        with self._lock:
            return self._buffers.get(workflow_id)

    def remove_buffer(self, workflow_id: str) -> Optional[WorkflowSpanBuffer]:
        """
        Remove and return buffer.

        Called by GovernanceWorkflowInterceptor after submission.

        Args:
            workflow_id: Temporal workflow ID

        Returns:
            Buffer if found, None otherwise
        """
        with self._lock:
            return self._buffers.pop(workflow_id, None)

    def unregister_workflow(self, workflow_id: str) -> None:
        """
        Remove buffer for a workflow (alias for remove_buffer).

        Called when clearing stale buffers from previous workflow runs.

        Args:
            workflow_id: Temporal workflow ID
        """
        with self._lock:
            self._buffers.pop(workflow_id, None)
            self._verdicts.pop(workflow_id, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Verdict Storage (called by workflow interceptor for SignalReceived stop)
    # ═══════════════════════════════════════════════════════════════════════════

    def set_verdict(self, workflow_id: str, verdict: Verdict, reason: str = None, run_id: str = None) -> None:
        """Store governance verdict for a workflow. Called when SignalReceived returns BLOCK/HALT."""
        with self._lock:
            self._verdicts[workflow_id] = {"verdict": verdict, "reason": reason, "run_id": run_id}
            if workflow_id in self._buffers:
                self._buffers[workflow_id].verdict = verdict
                self._buffers[workflow_id].verdict_reason = reason

    def get_verdict(self, workflow_id: str) -> Optional[dict]:
        """Get stored verdict for a workflow. Returns dict with 'verdict' (Verdict) and 'reason' keys."""
        with self._lock:
            return self._verdicts.get(workflow_id)

    def clear_verdict(self, workflow_id: str) -> None:
        """Clear stored verdict for a workflow."""
        with self._lock:
            self._verdicts.pop(workflow_id, None)

    # ═══════════════════════════════════════════════════════════════════════════
    # Body Storage (called by HTTP hooks in otel_setup.py)
    # ═══════════════════════════════════════════════════════════════════════════

    def store_body(
        self,
        span_id: int,
        request_body: Optional[str] = None,
        response_body: Optional[str] = None,
        request_headers: Optional[dict] = None,
        response_headers: Optional[dict] = None,
    ) -> None:
        """
        Store body and header data for a span (called from HTTP hooks).

        Bodies and headers are stored here, NOT in OTel span attributes.
        They will be merged with span data in on_end().

        Args:
            span_id: OTel span ID (integer form)
            request_body: HTTP request body (if available)
            response_body: HTTP response body (if available)
            request_headers: HTTP request headers (if available)
            response_headers: HTTP response headers (if available)
        """
        with self._lock:
            if span_id not in self._body_data:
                self._body_data[span_id] = {}
            if request_body is not None:
                self._body_data[span_id]["request_body"] = request_body
            if response_body is not None:
                self._body_data[span_id]["response_body"] = response_body
            if request_headers is not None:
                self._body_data[span_id]["request_headers"] = request_headers
            if response_headers is not None:
                self._body_data[span_id]["response_headers"] = response_headers

    def get_pending_body(self, span_id: int) -> Optional[dict]:
        """
        Get pending body data for a span (not yet merged).

        Used by activity interceptor to propagate body data to child spans
        before the activity span has ended (and on_end merged the data).

        Args:
            span_id: OTel span ID (integer form)

        Returns:
            Dict with request_body and/or response_body, or None
        """
        with self._lock:
            return self._body_data.get(span_id)

    # ═══════════════════════════════════════════════════════════════════════════
    # SpanProcessor Interface
    # ═══════════════════════════════════════════════════════════════════════════

    def on_start(self, span, parent_context=None) -> None:
        """Called when span starts. No-op for this processor."""
        pass

    def on_end(self, span: "ReadableSpan") -> None:
        """
        Called when span ends. Buffer by workflow_id.

        Spans with temporal.workflow_id attribute are buffered directly.
        Child spans (like HTTP calls) without workflow_id are buffered via
        trace_id -> workflow_id mapping.
        Body data stored via store_body() is merged here.
        """
        # Skip spans to ignored URLs (e.g., OpenBox Core API)
        if self._should_ignore_span(span):
            if self.fallback:
                self.fallback.on_end(span)
            return

        # Get workflow_id from span attributes (direct)
        workflow_id = span.attributes.get("temporal.workflow_id") if span.attributes else None
        activity_id = span.attributes.get("temporal.activity_id") if span.attributes else None

        # Fallback: look up by trace_id (for child spans like HTTP calls)
        if not workflow_id:
            with self._lock:
                workflow_id = self._trace_to_workflow.get(span.context.trace_id)
                # Also get activity_id from trace mapping for child spans
                if not activity_id:
                    activity_id = self._trace_to_activity.get(span.context.trace_id)

        if workflow_id:
            with self._lock:
                buffer = self._buffers.get(workflow_id)

            if buffer:
                span_data = self._extract_span_data(span)

                # Set activity_id for filtering later
                if activity_id:
                    span_data["activity_id"] = activity_id

                # Merge body data (stored separately, NOT in OTel span)
                span_id = span.context.span_id
                with self._lock:
                    if span_id in self._body_data:
                        body_data = self._body_data.pop(span_id)
                        span_data.update(body_data)

                buffer.spans.append(span_data)

        # Always forward to fallback (OTel exporter) - WITHOUT body
        if self.fallback:
            self.fallback.on_end(span)

    def _extract_span_data(self, span: "ReadableSpan") -> dict:
        """
        Extract span data for OpenBox API.

        Args:
            span: OTel ReadableSpan

        Returns:
            Dictionary matching SpanData structure
        """
        # Format span_id and trace_id as hex strings
        span_id_hex = format(span.context.span_id, "016x")
        trace_id_hex = format(span.context.trace_id, "032x")

        # Format parent span ID if present
        parent_span_id = None
        if span.parent and span.parent.span_id:
            parent_span_id = format(span.parent.span_id, "016x")

        # Extract status
        status = None
        if span.status:
            status = {
                "code": span.status.status_code.name if span.status.status_code else "UNSET",
                "description": span.status.description,
            }

        # Extract events
        events = []
        if span.events:
            for event in span.events:
                events.append(
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {},
                    }
                )

        # Calculate duration
        duration_ns = None
        if span.end_time and span.start_time:
            duration_ns = span.end_time - span.start_time

        return {
            "span_id": span_id_hex,
            "trace_id": trace_id_hex,
            "parent_span_id": parent_span_id,
            "name": span.name,
            "kind": span.kind.name if span.kind else None,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ns": duration_ns,
            "attributes": dict(span.attributes) if span.attributes else {},
            "status": status,
            "events": events,
            # request_body and response_body will be merged from _body_data
        }

    def shutdown(self) -> None:
        """Shutdown the processor."""
        if self.fallback:
            self.fallback.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush any buffered spans."""
        if self.fallback:
            return self.fallback.force_flush(timeout_millis)
        return True
